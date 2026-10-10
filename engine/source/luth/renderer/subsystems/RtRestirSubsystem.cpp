#include "luthpch.h"
#include "luth/renderer/subsystems/RtRestirSubsystem.h"
#include "luth/renderer/subsystems/RtSubsystem.h"
#include "luth/renderer/RenderPipeline.h"
#include "luth/renderer/Renderer.h"
#include "luth/renderer/FrameTargets.h"
#include "luth/renderer/settings/RestirSettings.h"
#include "luth/renderer/shader/ShaderLibrary.h"
#include "luth/scene/systems/RenderingSystem.h"
#include "luth/renderer/backend/vulkan/VulkanContext.h"
#include "luth/renderer/backend/vulkan/VulkanTexture.h"
#include "luth/renderer/material/MaterialSystem.h"
#include "luth/core/FrameData.h"
#include "luth/core/types/LuthMath.h"

namespace Luth
{
    namespace {
        // Sizes the reservoir allocation only; the GPU layout lives in restir_common.slang's
        // Reservoir struct; any field change must update both. see arch/rendering-pipeline.md
        struct GPUReservoir {
            u32 lightIndex;   // bit31 = light type (0 point / 1 triangle)
            f32 W, wSum;
            u32 M;
            f32 targetPdf;
            f32 histDepth;    // temporal-validation self-carry (raw depth)
            u32 histOctN;     // origin-pixel oct normal, f16x2
            u32 uvPacked;     // triangle barycentric uv, unorm16x2
        };
        static_assert(sizeof(GPUReservoir) == 32, "GPUReservoir must match restir_common.slang Reservoir (32 B)");

        struct RestirPC {
            Mat4 invViewProj;
            u32  candidateCount;
            u32  frameSeed;
            i32  gbufferScale;   // 1 = full-res; 2 = half-res DI (G-buffer reads remap to full)
            i32  dispatchW;      // DI working (dispatch) resolution
            i32  dispatchH;
            f32  diSpecClamp;    // fills the pre-pointer pad (offset 84); read only by the shade pass
            u64  geomTableBDA;   // cutout alpha-test material fetch; stays 8-aligned at offset 88
            f32  confidenceNorm; // SVGF confidence normalizer at offset 96; read only by the shade pass
        };
        static_assert(sizeof(RestirPC) == 104, "RestirPC must be 104 B (matches restir_shade.slang push_constant)");

        // Temporal-pass push constants. Same 80 B footprint + COMPUTE range as RestirPC, so the two
        // share the existing pcRange; the field meanings differ (M-cap + validation thresholds).
        struct RestirTemporalPC {
            Mat4 invViewProj;
            u32  mCap;
            u32  frameSeed;
            f32  depthThreshold;
            f32  normalThreshold;
            i32  gbufferScale;
            i32  dispatchW;
            i32  dispatchH;
        };
        static_assert(sizeof(RestirTemporalPC) == 92, "RestirTemporalPC must match restir_temporal.slang push_constant");

        // Spatial-pass push constants. Shares the fixed COMPUTE pcRange with the other three pipelines;
        // only the field meanings differ (neighbour disk + reject + final-visibility geometry table).
        struct RestirSpatialPC {
            Mat4 invViewProj;
            u32  neighbourCount;
            u32  radius;
            u32  frameSeed;
            f32  depthThreshold;
            i32  gbufferScale;
            i32  dispatchW;
            i32  dispatchH;
            f32  normalThreshold;      // min dot(neighbourN, currN)
            f32  roughnessThreshold;   // max |neighbourRough - rough| (spec reuse gate)
            f32  boilingStrength;      // boiling-filter kill knob, 0 disables (fills the ex-pad; geomTable stays at 104)
            u64  geomTableBDA;         // final-visibility alpha-test material fetch
        };
        static_assert(sizeof(RestirSpatialPC) == 112, "RestirSpatialPC must match restir_spatial.slang push_constant");

        // Fixed push-constant range shared by the four DI pipelines; 128 B (Vulkan min) leaves headroom
        // for the largest struct (spatial 100 B) plus later growth without touching the pipeline layout.
        constexpr u32 k_RestirPCSize = 128;

        struct UpscalePC {
            i32 fullW;
            i32 fullH;
            i32 halfW;
            i32 halfH;
            f32 phiDepth;
            f32 phiNormal;
        };
        static_assert(sizeof(UpscalePC) == 24, "UpscalePC must match bilateral_upscale.slang push_constant");
    }

    bool RtRestirSubsystem::IsEnabled() const
    {
        return m_Pipeline && m_Pipeline->GetSystem().GetRestirSettings().enabled;
    }

    void RtRestirSubsystem::SetEnabled(bool e)
    {
        if (m_Pipeline) m_Pipeline->GetSystem().GetRestirSettings().enabled = e;
    }

    void RtRestirSubsystem::Init(RenderPipeline& pipeline)
    {
        LH_PROFILE_FUNCTION();
        m_Pipeline = &pipeline;
        VkDevice device = VulkanContext::Get().GetDevice();

        // Linear clamp-to-edge; same shape as RtSubsystem's pass sampler for SceneDepth/SlimNormal.
        VkSamplerCreateInfo sampCI{ VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO };
        sampCI.magFilter    = VK_FILTER_LINEAR;
        sampCI.minFilter    = VK_FILTER_LINEAR;
        sampCI.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        sampCI.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        sampCI.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        sampCI.mipmapMode   = VK_SAMPLER_MIPMAP_MODE_NEAREST;
        vkCreateSampler(device, &sampCI, nullptr, &m_Sampler);

        // Set 2 (pass-local): b0 depth sampler, b1 slimNormal sampler, b2 reservoir SCRATCH
        // (initial -> temporal in-place, r/w SSBO), b3 DI storage image, b4 reservoir HISTORY = the
        // spatial buffer (read SSBO), b5 motion sampler, b6 spatial-output reservoir (write SSBO,
        // same buffer as b4). initial uses b0/b1/b2; temporal uses b0/b1/b2/b4/b5; spatial uses
        // b0/b1/b2(read)/b6(write); shade uses b0/b1/b6(read)/b3. All stable per-view.
        VkDescriptorSetLayoutBinding bindings[9]{};
        bindings[0].binding         = 0;
        bindings[0].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        bindings[0].descriptorCount = 1;
        bindings[0].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;
        bindings[1].binding         = 1;
        bindings[1].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        bindings[1].descriptorCount = 1;
        bindings[1].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;
        bindings[2].binding         = 2;
        bindings[2].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[2].descriptorCount = 1;
        bindings[2].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;
        bindings[3].binding         = 3;
        bindings[3].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        bindings[3].descriptorCount = 1;
        bindings[3].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;
        bindings[4].binding         = 4;
        bindings[4].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[4].descriptorCount = 1;
        bindings[4].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;
        bindings[5].binding         = 5;
        bindings[5].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        bindings[5].descriptorCount = 1;
        bindings[5].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;
        bindings[6].binding         = 6;
        bindings[6].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[6].descriptorCount = 1;
        bindings[6].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;
        bindings[7].binding         = 7;   // slimRoughness sampler (combined diffuse+spec target + spec shade)
        bindings[7].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        bindings[7].descriptorCount = 1;
        bindings[7].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;
        bindings[8].binding         = 8;   // restirDISpec storage image (demodulated specular out)
        bindings[8].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        bindings[8].descriptorCount = 1;
        bindings[8].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;

        // b2/b4 are now stable per-view (written at WriteView time like the rest); the UAB flags stay
        // so a resize-time rewrite while older cycled slots are still in flight remains legal
        // (VUID-vkUpdateDescriptorSets-None-03047).
        VkDescriptorBindingFlags bindingFlags[9] = {
            0,                                            // b0 depth sampler
            0,                                            // b1 normal sampler
            VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT,  // b2 reservoir scratch
            0,                                            // b3 DI storage image
            VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT,  // b4 reservoir history (spatial buffer)
            0,                                            // b5 motion sampler
            0,                                            // b6 spatial output reservoir
            0,                                            // b7 slimRoughness sampler
            0,                                            // b8 restirDISpec storage image
        };
        VkDescriptorSetLayoutBindingFlagsCreateInfo bindingFlagsCI{
            VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO };
        bindingFlagsCI.bindingCount  = 9;
        bindingFlagsCI.pBindingFlags = bindingFlags;

        VkDescriptorSetLayoutCreateInfo layoutCI{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
        layoutCI.pNext        = &bindingFlagsCI;
        layoutCI.flags        = VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT;
        layoutCI.bindingCount = 9;
        layoutCI.pBindings    = bindings;
        vkCreateDescriptorSetLayout(device, &layoutCI, nullptr, &m_SetLayout);

        if (auto sh = ShaderLibrary::LoadEngine("shaders/restir_initial.slang"))
            m_InitialSpv = sh->GetSpirV();
        if (auto sh = ShaderLibrary::LoadEngine("shaders/restir_temporal.slang"))
            m_TemporalSpv = sh->GetSpirV();
        if (auto sh = ShaderLibrary::LoadEngine("shaders/restir_spatial.slang"))
            m_SpatialSpv = sh->GetSpirV();
        if (auto sh = ShaderLibrary::LoadEngine("shaders/restir_shade.slang"))
            m_ShadeSpv = sh->GetSpirV();
        if (m_InitialSpv.empty() || m_TemporalSpv.empty() || m_SpatialSpv.empty() || m_ShadeSpv.empty())
        {
            LH_LOG(Renderer, error, "RtRestirSubsystem: failed to load restir_initial/temporal/spatial/shade.comp SPIR-V");
            return;
        }

        // Sets: 0 = global (UBO b0 + TLAS b6), 1 = light SSBO, 2 = pass-local. The initial AND spatial
        // passes add Set 3 (Material SSBO) + Set 4 (bindless) for the cutout alpha-test on their
        // visibility rays (material_bindings_rt.slang); temporal/shade trace no rays, 3-set layout.
        const std::vector<VkDescriptorSetLayout> layouts = {
            m_Pipeline->GetGlobal().GetSetLayout(),
            m_Pipeline->GetLighting().GetSetLayout(),
            m_SetLayout,
        };
        std::vector<VkDescriptorSetLayout> layoutsInitial = layouts;
        layoutsInitial.push_back(MaterialSystem::GetDescriptorSetLayout());
        layoutsInitial.push_back(VulkanContext::Get().GetBindlessSet().GetLayout());
        VkPushConstantRange pcRange{ VK_SHADER_STAGE_COMPUTE_BIT, 0, k_RestirPCSize };

        m_InitialPipeline = std::make_unique<VKComputePipeline>(
            m_InitialSpv, layoutsInitial, std::vector<VkPushConstantRange>{ pcRange });
        m_TemporalPipeline = std::make_unique<VKComputePipeline>(
            m_TemporalSpv, layouts, std::vector<VkPushConstantRange>{ pcRange });
        m_SpatialPipeline = std::make_unique<VKComputePipeline>(
            m_SpatialSpv, layoutsInitial, std::vector<VkPushConstantRange>{ pcRange });
        m_ShadePipeline = std::make_unique<VKComputePipeline>(
            m_ShadeSpv, layouts, std::vector<VkPushConstantRange>{ pcRange });

        // Half-res DI bilateral-upscale pipeline (shared bilateral_upscale.slang). Set 0 = global UBO;
        // Set 1 = b0 half-res signal sampler, b1 depth sampler, b2 normal sampler, b3 full-res storage.
        {
            VkDescriptorSetLayoutBinding ub[4]{};
            ub[0].binding = 0; ub[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; ub[0].descriptorCount = 1; ub[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
            ub[1].binding = 1; ub[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; ub[1].descriptorCount = 1; ub[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
            ub[2].binding = 2; ub[2].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; ub[2].descriptorCount = 1; ub[2].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
            ub[3].binding = 3; ub[3].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;          ub[3].descriptorCount = 1; ub[3].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
            VkDescriptorSetLayoutCreateInfo uci{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
            uci.bindingCount = 4; uci.pBindings = ub;
            vkCreateDescriptorSetLayout(device, &uci, nullptr, &m_UpscaleSetLayout);

            if (auto sh = ShaderLibrary::LoadEngine("shaders/bilateral_upscale.slang")) m_UpscaleSpv = sh->GetSpirV();
            if (!m_UpscaleSpv.empty())
            {
                const std::vector<VkDescriptorSetLayout> ulayouts = {
                    m_Pipeline->GetGlobal().GetSetLayout(),
                    m_UpscaleSetLayout,
                };
                VkPushConstantRange upc{ VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(UpscalePC) };
                m_UpscalePipeline = std::make_unique<VKComputePipeline>(
                    m_UpscaleSpv, ulayouts, std::vector<VkPushConstantRange>{ upc });
            }
        }
    }

    void RtRestirSubsystem::Shutdown()
    {
        LH_PROFILE_FUNCTION();
        m_UpscaleViews.ReleaseAll([] { Renderer::WaitForGPU(); });
        m_Views.ReleaseAll([] { Renderer::WaitForGPU(); });
        VkDevice device = VulkanContext::Get().GetDevice();
        m_InitialPipeline.reset();
        m_TemporalPipeline.reset();
        m_SpatialPipeline.reset();
        m_ShadePipeline.reset();
        m_UpscalePipeline.reset();
        if (m_Sampler)          vkDestroySampler(device, m_Sampler, nullptr);
        if (m_SetLayout)        vkDestroyDescriptorSetLayout(device, m_SetLayout, nullptr);
        if (m_UpscaleSetLayout) vkDestroyDescriptorSetLayout(device, m_UpscaleSetLayout, nullptr);
        m_Sampler          = VK_NULL_HANDLE;
        m_SetLayout        = VK_NULL_HANDLE;
        m_UpscaleSetLayout = VK_NULL_HANDLE;
        m_InitialSpv.clear();
        m_TemporalSpv.clear();
        m_SpatialSpv.clear();
        m_ShadeSpv.clear();
        m_UpscaleSpv.clear();
        m_Pipeline = nullptr;
    }

    bool RtRestirSubsystem::OnShaderReloaded(const std::string& name, const std::vector<u32>& spv)
    {
        LH_PROFILE_FUNCTION();
        if (m_SetLayout == VK_NULL_HANDLE || !m_Pipeline) return false;

        if (name == "bilateral_upscale.slang" && m_UpscaleSetLayout != VK_NULL_HANDLE)
        {
            m_UpscaleSpv = spv;
            if (auto* raw = m_UpscalePipeline.release(); raw)
                VulkanContext::Get().PushDeletion([raw]() { delete raw; });
            const std::vector<VkDescriptorSetLayout> ulayouts = {
                m_Pipeline->GetGlobal().GetSetLayout(), m_UpscaleSetLayout };
            VkPushConstantRange upc{ VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(UpscalePC) };
            m_UpscalePipeline = std::make_unique<VKComputePipeline>(
                m_UpscaleSpv, ulayouts, std::vector<VkPushConstantRange>{ upc });
            return true;
        }

        const bool isInitial  = (name == "restir_initial.slang");
        const bool isTemporal = (name == "restir_temporal.slang");
        const bool isSpatial  = (name == "restir_spatial.slang");
        const bool isShade    = (name == "restir_shade.slang");
        if (!isInitial && !isTemporal && !isSpatial && !isShade) return false;

        const std::vector<VkDescriptorSetLayout> layouts = {
            m_Pipeline->GetGlobal().GetSetLayout(),
            m_Pipeline->GetLighting().GetSetLayout(),
            m_SetLayout,
        };
        std::vector<VkDescriptorSetLayout> layoutsInitial = layouts;
        layoutsInitial.push_back(MaterialSystem::GetDescriptorSetLayout());
        layoutsInitial.push_back(VulkanContext::Get().GetBindlessSet().GetLayout());
        VkPushConstantRange pcRange{ VK_SHADER_STAGE_COMPUTE_BIT, 0, k_RestirPCSize };

        auto deferComp = [](std::unique_ptr<VKComputePipeline>& p) {
            if (auto* raw = p.release(); raw)
                VulkanContext::Get().PushDeletion([raw]() { delete raw; });
        };

        if (isInitial)
        {
            m_InitialSpv = spv;
            deferComp(m_InitialPipeline);
            m_InitialPipeline = std::make_unique<VKComputePipeline>(
                m_InitialSpv, layoutsInitial, std::vector<VkPushConstantRange>{ pcRange });
        }
        else if (isTemporal)
        {
            m_TemporalSpv = spv;
            deferComp(m_TemporalPipeline);
            m_TemporalPipeline = std::make_unique<VKComputePipeline>(
                m_TemporalSpv, layouts, std::vector<VkPushConstantRange>{ pcRange });
        }
        else if (isSpatial)
        {
            m_SpatialSpv = spv;
            deferComp(m_SpatialPipeline);
            m_SpatialPipeline = std::make_unique<VKComputePipeline>(
                m_SpatialSpv, layoutsInitial, std::vector<VkPushConstantRange>{ pcRange });
        }
        else
        {
            m_ShadeSpv = spv;
            deferComp(m_ShadePipeline);
            m_ShadePipeline = std::make_unique<VKComputePipeline>(
                m_ShadeSpv, layouts, std::vector<VkPushConstantRange>{ pcRange });
        }
        m_Views.ForEach([](auto& state) { state->history.Invalidate(); });
        return true;
    }

    std::shared_ptr<RestirDiViewState> RtRestirSubsystem::EnsureView(RenderViewId id,
        const FrameTargets& targets, bool half)
    {
        if (!m_SetLayout) return {};
        std::array<std::shared_ptr<Texture>, 4> sources{
            targets.GetSceneDepth(), targets.GetSlimNormal(), targets.GetSlimMotion(), targets.GetSlimRoughness()};
        std::array<VkImageView, 4> views{};
        if (!sources[0]) throw std::invalid_argument("ReSTIR DI: missing depth source");
        const auto width = sources[0]->GetWidth(), height = sources[0]->GetHeight();
        for (u32 i = 0; i < sources.size(); ++i) {
            if (!sources[i] || sources[i]->GetWidth() != width || sources[i]->GetHeight() != height)
                throw std::invalid_argument("ReSTIR DI: incompatible sampled sources");
            views[i] = std::static_pointer_cast<VKTexture>(sources[i])->GetImageView();
            if (!views[i]) throw std::invalid_argument("ReSTIR DI: missing sampled image view");
        }
        const auto* prior = m_Views.Find(id);
        const u64 generation = prior && (*prior)->sourceViews == views
            ? (*prior)->sourceGeneration : m_NextSourceGeneration++;
        const auto config = RestirDiViewState::Config(width, height, half, generation);
        return m_Views.Ensure(id, config, [&](const ViewStateConfig& requested) {
            const u32 scratchTag = NextReservoirTag(), spatialTag = NextReservoirTag();
            auto state = RestirDiViewState::Create(id, requested, m_SetLayout, scratchTag, spatialTag);
            state->sources = sources; state->sourceViews = views;
            WriteView(*state, targets);
            return state;
        }, [] { Renderer::WaitForGPU(); });
    }

    void RtRestirSubsystem::ReleaseView(RenderViewId id)
    {
        m_UpscaleViews.Release(id, [] { Renderer::WaitForGPU(); });
        m_Views.Release(id, [] { Renderer::WaitForGPU(); });
    }

    void RtRestirSubsystem::WriteView(RestirDiViewState& state, const FrameTargets& targets)
    {
        LH_PROFILE_FUNCTION();
        if (state.restirDescSet[0] == VK_NULL_HANDLE) return;
        if (!targets.GetSceneDepth() || !targets.GetSlimNormal() || !targets.GetSlimMotion()
            || !targets.GetSlimRoughness() || !state.restirDI || !state.restirDISpec) return;
        if (!state.restirSpatial.buffer || !state.restirReservoir.buffer) return;

        VkDevice device = VulkanContext::Get().GetDevice();

        const VkImageView depthView  = std::static_pointer_cast<VKTexture>(targets.GetSceneDepth())->GetImageView();
        const VkImageView normalView = std::static_pointer_cast<VKTexture>(targets.GetSlimNormal())->GetImageView();
        const VkImageView motionView = std::static_pointer_cast<VKTexture>(targets.GetSlimMotion())->GetImageView();
        const VkImageView diView     = std::static_pointer_cast<VKTexture>(state.restirDI)->GetImageView();
        const VkImageView roughView  = std::static_pointer_cast<VKTexture>(targets.GetSlimRoughness())->GetImageView();
        const VkImageView specView   = std::static_pointer_cast<VKTexture>(state.restirDISpec)->GetImageView();

        VkDescriptorImageInfo depthInfo{};
        depthInfo.sampler     = m_Sampler;
        depthInfo.imageView   = depthView;
        depthInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

        VkDescriptorImageInfo normalInfo{};
        normalInfo.sampler     = m_Sampler;
        normalInfo.imageView   = normalView;
        normalInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

        VkDescriptorImageInfo motionInfo{};
        motionInfo.sampler     = m_Sampler;
        motionInfo.imageView   = motionView;
        motionInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

        VkDescriptorImageInfo diInfo{};
        diInfo.imageView   = diView;
        diInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

        VkDescriptorImageInfo roughInfo{};
        roughInfo.sampler     = m_Sampler;
        roughInfo.imageView   = roughView;
        roughInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

        VkDescriptorImageInfo specInfo{};
        specInfo.imageView   = specView;   // restirDISpec: GENERAL (storage write from shade)
        specInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

        // b2 scratch reservoir (initial -> temporal in-place, same-frame lifetime only) + b4 temporal
        // history + b6 spatial output. b4 and b6 alias the SAME per-view buffer: temporal reads last
        // frame's spatial result (b4) before spatial overwrites it (b6); the RG emits the WAR barrier.
        VkDescriptorBufferInfo scratchInfo{
            state.restirReservoir.buffer, state.restirReservoir.offset, state.restirReservoir.size };
        VkDescriptorBufferInfo spatialInfo{
            state.restirSpatial.buffer, state.restirSpatial.offset, state.restirSpatial.size };

        // All Set 2 bindings are stable per-view now (b2/b4 stopped ping-ponging with the post-spatial
        // history topology); rewritten only on view alloc/resize.
        VkWriteDescriptorSet writes[9 * MAX_FRAMES_IN_FLIGHT]{};
        u32 n = 0;
        for (u32 slot = 0; slot < MAX_FRAMES_IN_FLIGHT; ++slot)
        {
            VkDescriptorSet set = state.restirDescSet[slot];
            if (set == VK_NULL_HANDLE) continue;

            writes[n] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
            writes[n].dstSet          = set;
            writes[n].dstBinding      = 0;
            writes[n].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            writes[n].descriptorCount = 1;
            writes[n].pImageInfo      = &depthInfo;
            ++n;

            writes[n] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
            writes[n].dstSet          = set;
            writes[n].dstBinding      = 2;
            writes[n].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[n].descriptorCount = 1;
            writes[n].pBufferInfo     = &scratchInfo;
            ++n;

            writes[n] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
            writes[n].dstSet          = set;
            writes[n].dstBinding      = 4;
            writes[n].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[n].descriptorCount = 1;
            writes[n].pBufferInfo     = &spatialInfo;
            ++n;

            writes[n] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
            writes[n].dstSet          = set;
            writes[n].dstBinding      = 1;
            writes[n].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            writes[n].descriptorCount = 1;
            writes[n].pImageInfo      = &normalInfo;
            ++n;

            writes[n] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
            writes[n].dstSet          = set;
            writes[n].dstBinding      = 3;
            writes[n].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            writes[n].descriptorCount = 1;
            writes[n].pImageInfo      = &diInfo;
            ++n;

            writes[n] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
            writes[n].dstSet          = set;
            writes[n].dstBinding      = 5;
            writes[n].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            writes[n].descriptorCount = 1;
            writes[n].pImageInfo      = &motionInfo;
            ++n;

            writes[n] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
            writes[n].dstSet          = set;
            writes[n].dstBinding      = 6;
            writes[n].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[n].descriptorCount = 1;
            writes[n].pBufferInfo     = &spatialInfo;
            ++n;

            writes[n] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
            writes[n].dstSet          = set;
            writes[n].dstBinding      = 7;
            writes[n].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            writes[n].descriptorCount = 1;
            writes[n].pImageInfo      = &roughInfo;
            ++n;

            writes[n] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
            writes[n].dstSet          = set;
            writes[n].dstBinding      = 8;
            writes[n].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            writes[n].descriptorCount = 1;
            writes[n].pImageInfo      = &specInfo;
            ++n;
        }
        vkUpdateDescriptorSets(device, n, writes, 0, nullptr);
    }

    RestirDiBindings RtRestirSubsystem::PrepareBindings(const ViewResources& vr, u64 frameIndex,
        RenderViewId view, u64 generation, const PreparedRtScene* scene, const RestirSettings& settings,
        const Mat4& inverseViewProjection, const Memory::GPUSubRegion& lights) const
    {
        RestirDiBindings packet;
        packet.frameIndex = frameIndex; packet.view = view; packet.generation = generation;
        packet.settings = settings; packet.inverseViewProjection = inverseViewProjection;
        packet.lights = lights;
        packet.fullWidth = vr.width; packet.fullHeight = vr.height;
        if (!m_InitialPipeline || !m_TemporalPipeline || !m_SpatialPipeline || !m_ShadePipeline || !vr.restirDi)
            return packet;
        const u32 slot = static_cast<u32>(frameIndex % MAX_FRAMES_IN_FLIGHT);
        const std::array pipelines{m_InitialPipeline.get(), m_TemporalPipeline.get(),
            m_SpatialPipeline.get(), m_ShadePipeline.get()};
        for (u32 i = 0; i < pipelines.size(); ++i) {
            packet.pipelines[i] = pipelines[i]->GetHandle(); packet.layouts[i] = pipelines[i]->GetLayout();
        }
        packet.retained = vr.restirDi;
        packet.historyValid = vr.restirDi->history.CanReuse(frameIndex, generation,
            vr.cameraHistory.CanReuse(frameIndex, generation));
        packet.width = vr.restirDi->width; packet.height = vr.restirDi->height;
        packet.scratch = vr.restirDi->restirReservoir; packet.spatial = vr.restirDi->restirSpatial;
        packet.sets = {vr.globalDescriptorSet[slot], vr.lightDescSet[slot], vr.restirDi->restirDescSet[slot],
            MaterialSystem::GetDescriptorSet(slot), VulkanContext::Get().GetBindlessSet().GetSet()};
        for (u32 i = 0; i < packet.sources.size(); ++i) {
            auto source = std::static_pointer_cast<VKTexture>(vr.restirDi->sources[i]);
            packet.sources[i] = source.get();
            if (source) { packet.sourceImages[i] = source->GetImage(); packet.sourceViews[i] = source->GetImageView(); }
        }
        const std::array outputs{vr.restirDi->restirDI, vr.restirDi->restirDISpec};
        for (u32 i = 0; i < outputs.size(); ++i) {
            if (!outputs[i]) continue;
            auto texture = std::static_pointer_cast<VKTexture>(outputs[i]);
            packet.images[i] = texture->GetImage(); packet.imageViews[i] = texture->GetImageView();
            packet.outputs[i] = {texture.get()};
        }
        if (scene) { packet.tlas = scene->GetTlas(); packet.geometryTable = scene->GetGeometryTableBDA(); }
        return packet;
    }

    RtRestirSubsystem::Outputs RtRestirSubsystem::AddPasses(RG::RenderGraph& rg,
                                                    RG::ResourceHandle sceneDepth,
                                                    RG::ResourceHandle slimNormal,
                                                    RG::ResourceHandle slimMotion,
                                                    RG::ResourceHandle slimRoughness, const RestirDiBindings& native, RG::BufferHandle lights)
    {
        LH_PROFILE_FUNCTION();
        const auto& settings = native.settings;
        if (!settings.enabled || std::any_of(native.pipelines.begin(), native.pipelines.end(), [](auto p) { return !p; })) return {};
        const u32 frameAbs = static_cast<u32>(native.frameIndex); // Shader seed ABI stays 32-bit.
        const auto scratchRes = native.scratch, spatialRes = native.spatial;
        const Mat4 invVP = native.inverseViewProjection;
        const i32 diW2 = static_cast<i32>(native.width), diH2 = static_cast<i32>(native.height);
        const i32 diScale = native.width == native.fullWidth && native.height == native.fullHeight ? 1 : 2;
        RestirPC pc{};
        pc.invViewProj    = invVP;
        pc.candidateCount = settings.candidateCount;
        pc.diSpecClamp    = settings.diSpecClamp;
        pc.confidenceNorm = settings.confidenceNorm;
        pc.frameSeed      = frameAbs;
        pc.gbufferScale   = diScale;
        pc.dispatchW      = diW2;
        pc.dispatchH      = diH2;
        pc.geomTableBDA   = native.geometryTable;

        RestirTemporalPC tpc{};
        tpc.invViewProj     = invVP;
        tpc.mCap            = native.historyValid ? settings.temporalMCap : 0;
        tpc.frameSeed       = frameAbs;
        tpc.depthThreshold  = settings.temporalDepthThreshold;
        tpc.normalThreshold = settings.temporalNormalThreshold;
        tpc.gbufferScale    = diScale;
        tpc.dispatchW       = diW2;
        tpc.dispatchH       = diH2;

        RestirSpatialPC spc{};
        spc.invViewProj    = invVP;
        spc.neighbourCount = settings.spatialNeighbours;
        spc.radius         = settings.spatialRadius;
        spc.frameSeed      = frameAbs;
        spc.depthThreshold = settings.spatialDepthThreshold;
        spc.normalThreshold    = settings.spatialNormalThreshold;
        spc.roughnessThreshold = settings.roughnessThreshold;
        spc.boilingStrength    = settings.boilingStrength;
        spc.gbufferScale   = diScale;
        spc.dispatchW      = diW2;
        spc.dispatchH      = diH2;
        spc.geomTableBDA   = native.geometryTable;

        // Initial pass: RIS over point lights + one visibility ray, writes the SCRATCH reservoir.
        // The scratch buffer is imported ONCE here; its handle threads through temporal (read+write)
        // and spatial (read) so the RG chains the barriers across all three (re-importing would
        // alias distinct nodes). Undefined import is correct: fully overwritten, no cross-frame read.
        struct RestirInitialData {
            RG::ResourceHandle depth;
            RG::ResourceHandle normal;
            RG::ResourceHandle rough;
            RG::BufferHandle   reservoir;
        };
        RG::BufferHandle reservoirHandle{};
        rg.AddComputePass<RestirInitialData>(
            "RestirInitial",
            RG::QueueFamily::AsyncCompute,
            [&](RestirInitialData& data, RG::RenderPassBuilder& builder) {
                builder.ReadBuffer(lights);
                if (sceneDepth.IsValid())    data.depth  = builder.ReadStorageImage(sceneDepth);
                if (slimNormal.IsValid())    data.normal = builder.ReadStorageImage(slimNormal);
                if (slimRoughness.IsValid()) data.rough  = builder.ReadStorageImage(slimRoughness);

                RG::BufferDesc bd{ "RestirReservoirScratch", scratchRes.size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT };
                data.reservoir  = rg.ImportBuffer(bd, (void*)scratchRes.buffer, RG::ResourceState::Undefined);
                data.reservoir  = builder.WriteBuffer(data.reservoir);
                reservoirHandle = data.reservoir;
            },
            [native, pc](RestirInitialData&, RG::RenderPassContext& ctx) {
                VkCommandBuffer cmd = ctx.commandBuffer;

                // AS-build -> AS-read barrier. dstStageMask is COMPUTE_SHADER (NOT RAY_TRACING):
                // rayQuery executes in the compute stage; a RAY_TRACING dst here is a TDR trap.
                VkMemoryBarrier2 asBarrier{ VK_STRUCTURE_TYPE_MEMORY_BARRIER_2 };
                asBarrier.srcStageMask  = VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR;
                asBarrier.srcAccessMask = VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
                asBarrier.dstStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
                asBarrier.dstAccessMask = VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR;
                VkDependencyInfo asDep{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
                asDep.memoryBarrierCount = 1;
                asDep.pMemoryBarriers    = &asBarrier;
                vkCmdPipelineBarrier2(cmd, &asDep);
                vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, native.pipelines[0]);
                VkDescriptorSet sets[5] = {
                    native.sets[0],
                    native.sets[1],
                    native.sets[2],
                    native.sets[3],
                    native.sets[4],
                };
                vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                    native.layouts[0], 0, 5, sets, 0, nullptr);
                vkCmdPushConstants(cmd, native.layouts[0],
                    VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(RestirPC), &pc);

                const u32 groupX = (static_cast<u32>(pc.dispatchW) + 7) / 8;
                const u32 groupY = (static_cast<u32>(pc.dispatchH) + 7) / 8;
                vkCmdDispatch(cmd, groupX, groupY, 1);
            });

        // Temporal pass: reprojects via motion + merges last frame's SPATIAL output (the history) into
        // the scratch RIS candidate in-place. No AS barrier (traces no rays). The history is the
        // per-view spatial buffer, imported ONCE here in its true last-left state (StorageBufferWrite,
        // NOT Undefined: Undefined -> srcAccess=0 -> no cross-frame availability -> stale temporal
        // read; see arch/rendering-pipeline.md); its handle threads into spatial's WriteBuffer so the
        // RG emits the temporal-read -> spatial-write WAR barrier on the same node. SCRATCH threads
        // through reservoirHandle as read+write (initial->temporal RAW barrier).
        struct RestirTemporalData {
            RG::ResourceHandle depth;
            RG::ResourceHandle normal;
            RG::ResourceHandle motion;
            RG::ResourceHandle rough;
            RG::BufferHandle   reservoirCurr;
            RG::BufferHandle   reservoirPrev;
        };
        RG::BufferHandle spatialHandle{};
        rg.AddComputePass<RestirTemporalData>(
            "RestirTemporal",
            RG::QueueFamily::AsyncCompute,
            [&](RestirTemporalData& data, RG::RenderPassBuilder& builder) {
                builder.ReadBuffer(lights);
                if (sceneDepth.IsValid())    data.depth  = builder.ReadStorageImage(sceneDepth);
                if (slimNormal.IsValid())    data.normal = builder.ReadStorageImage(slimNormal);
                if (slimMotion.IsValid())    data.motion = builder.ReadStorageImage(slimMotion);
                if (slimRoughness.IsValid()) data.rough  = builder.ReadStorageImage(slimRoughness);

                RG::BufferDesc histBd{ "RestirReservoirSpatial", spatialRes.size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT };
                spatialHandle      = rg.ImportBuffer(histBd, (void*)spatialRes.buffer, RG::ResourceState::StorageBufferWrite);
                data.reservoirPrev = builder.ReadBuffer(spatialHandle);

                data.reservoirCurr = builder.ReadBuffer(reservoirHandle);
                data.reservoirCurr = builder.WriteBuffer(data.reservoirCurr);
                reservoirHandle    = data.reservoirCurr;
            },
            [native, tpc](RestirTemporalData&, RG::RenderPassContext& ctx) {
                VkCommandBuffer cmd = ctx.commandBuffer;
                vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, native.pipelines[1]);
                VkDescriptorSet sets[3] = {
                    native.sets[0],
                    native.sets[1],
                    native.sets[2],
                };
                vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                    native.layouts[1], 0, 3, sets, 0, nullptr);
                vkCmdPushConstants(cmd, native.layouts[1],
                    VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(RestirTemporalPC), &tpc);

                const u32 groupX = (static_cast<u32>(tpc.dispatchW) + 7) / 8;
                const u32 groupY = (static_cast<u32>(tpc.dispatchH) + 7) / 8;
                vkCmdDispatch(cmd, groupX, groupY, 1);
            });

        // Spatial pass: merges each pixel's temporal-output reservoir (b2) with a few random disk
        // neighbours, rejecting dissimilar geometry, into the spatial/history buffer (b6), then traces
        // one final-visibility ray on the selected sample (5-set layout: Material + Bindless for the
        // alpha test; the initial pass's AS-build -> COMPUTE barrier covers this same-queue trace).
        // Reads b2 read-only (neighbour reads must see un-modified values; never in-place). The scratch
        // handle ends here: ReadBuffer(reservoirHandle) is its last consumer (temporal->spatial RAW
        // barrier). WriteBuffer on the SAME node temporal read (spatialHandle) yields the WAR barrier;
        // the result persists as next frame's history.
        struct RestirSpatialData {
            RG::ResourceHandle depth;
            RG::ResourceHandle normal;
            RG::ResourceHandle rough;
            RG::BufferHandle   reservoirIn;
            RG::BufferHandle   reservoirOut;
        };
        rg.AddComputePass<RestirSpatialData>(
            "RestirSpatial",
            RG::QueueFamily::AsyncCompute,
            [&](RestirSpatialData& data, RG::RenderPassBuilder& builder) {
                builder.ReadBuffer(lights);
                if (sceneDepth.IsValid())    data.depth  = builder.ReadStorageImage(sceneDepth);
                if (slimNormal.IsValid())    data.normal = builder.ReadStorageImage(slimNormal);
                if (slimRoughness.IsValid()) data.rough  = builder.ReadStorageImage(slimRoughness);

                data.reservoirIn  = builder.ReadBuffer(reservoirHandle);
                data.reservoirOut = builder.WriteBuffer(spatialHandle);
                spatialHandle     = data.reservoirOut;
            },
            [native, spc](RestirSpatialData&, RG::RenderPassContext& ctx) {
                VkCommandBuffer cmd = ctx.commandBuffer;
                vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, native.pipelines[2]);
                VkDescriptorSet sets[5] = {
                    native.sets[0],
                    native.sets[1],
                    native.sets[2],
                    native.sets[3],
                    native.sets[4],
                };
                vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                    native.layouts[2], 0, 5, sets, 0, nullptr);
                vkCmdPushConstants(cmd, native.layouts[2],
                    VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(RestirSpatialPC), &spc);

                const u32 groupX = (static_cast<u32>(spc.dispatchW) + 7) / 8;
                const u32 groupY = (static_cast<u32>(spc.dispatchH) + 7) / 8;
                vkCmdDispatch(cmd, groupX, groupY, 1);
            });

        // Shade pass: reads the SPATIAL-output reservoir (b6) + depth/normal, writes demodulated DI
        // image. Reads spatialHandle (not the temporal output); the shader's b6 is the spatial result.
        struct RestirShadeData {
            RG::ResourceHandle depth;
            RG::ResourceHandle normal;
            RG::ResourceHandle rough;
            RG::ResourceHandle di;
            RG::ResourceHandle spec;
            RG::BufferHandle   reservoir;
        };
        RG::ResourceHandle diHandle{};
        RG::ResourceHandle specHandle{};
        rg.AddComputePass<RestirShadeData>(
            "RestirShade",
            RG::QueueFamily::AsyncCompute,
            [&](RestirShadeData& data, RG::RenderPassBuilder& builder) {
                builder.ReadBuffer(lights);
                if (sceneDepth.IsValid())    data.depth  = builder.ReadStorageImage(sceneDepth);
                if (slimNormal.IsValid())    data.normal = builder.ReadStorageImage(slimNormal);
                if (slimRoughness.IsValid()) data.rough  = builder.ReadStorageImage(slimRoughness);
                data.reservoir = builder.ReadBuffer(spatialHandle);
                RG::TextureDesc desc;
                desc.name   = "RestirDI";
                desc.width  = native.width;
                desc.height = native.height;
                desc.format = RG::TextureFormat::RGBA16_Float;
                data.di  = rg.ImportResource(desc,
                    (void*)native.images[0], (void*)native.imageViews[0],
                    RG::ResourceState::Undefined);
                data.di  = builder.WriteStorageImage(data.di);
                diHandle = data.di;

                // Second output: demodulated specular (b8). Imported once here; its handle feeds
                // the DiSpecular SVGF denoiser. Mirrors the DI import above (same shape, GENERAL storage).
                RG::TextureDesc specDesc;
                specDesc.name   = "RestirDISpec";
                specDesc.width  = native.width;
                specDesc.height = native.height;
                specDesc.format = RG::TextureFormat::RGBA16_Float;
                data.spec  = rg.ImportResource(specDesc,
                    (void*)native.images[1], (void*)native.imageViews[1],
                    RG::ResourceState::Undefined);
                data.spec  = builder.WriteStorageImage(data.spec);
                specHandle = data.spec;
            },
            [native, pc](RestirShadeData&, RG::RenderPassContext& ctx) {
                VkCommandBuffer cmd = ctx.commandBuffer;
                vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, native.pipelines[3]);
                VkDescriptorSet sets[3] = {
                    native.sets[0],
                    native.sets[1],
                    native.sets[2],
                };
                vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                    native.layouts[3], 0, 3, sets, 0, nullptr);
                vkCmdPushConstants(cmd, native.layouts[3],
                    VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(RestirPC), &pc);

                const u32 groupX = (static_cast<u32>(pc.dispatchW) + 7) / 8;
                const u32 groupY = (static_cast<u32>(pc.dispatchH) + 7) / 8;
                vkCmdDispatch(cmd, groupX, groupY, 1);
            });

        return { diHandle, specHandle };
    }

    std::shared_ptr<DiUpscaleViewState> RtRestirSubsystem::EnsureUpscaleView(RenderViewId id,
        const FrameTargets& targets, const std::shared_ptr<DiDenoiserViewState>& diffuse,
        const std::shared_ptr<DiDenoiserViewState>& specular)
    {
        if (!m_UpscaleSetLayout || !m_Sampler || !diffuse || !specular) return {};
        if (!targets.GetSceneColor()) throw std::invalid_argument("DI upscale: missing view output");
        const std::array owners{diffuse, specular};
        const std::array sources{targets.GetSceneDepth(), targets.GetSlimNormal()};
        std::array<VkImageView, 2> views{};
        for (u32 i = 0; i < sources.size(); ++i) {
            if (!sources[i] || sources[i]->GetWidth() != targets.GetSceneColor()->GetWidth() ||
                sources[i]->GetHeight() != targets.GetSceneColor()->GetHeight())
                throw std::invalid_argument("DI upscale: incompatible full-resolution source");
            views[i] = std::static_pointer_cast<VKTexture>(sources[i])->GetImageView();
            if (!views[i]) throw std::invalid_argument("DI upscale: missing source view");
        }
        const auto* prior = m_UpscaleViews.Find(id);
        const u64 generation = prior && (*prior)->denoisers == owners && (*prior)->sources == sources &&
            (*prior)->sourceViews == views ? (*prior)->sourceGeneration : m_NextUpscaleGeneration++;
        const ViewStateConfig config{targets.GetSceneColor()->GetWidth(), targets.GetSceneColor()->GetHeight(), 0, generation};
        return m_UpscaleViews.Ensure(id, config, [&](const ViewStateConfig& requested) {
            return DiUpscaleViewState::Create(id, requested, m_UpscaleSetLayout, m_Sampler, owners, sources);
        }, [] { Renderer::WaitForGPU(); });
    }

    DiUpscaleBindings RtRestirSubsystem::PrepareUpscaleBindings(const ViewResources& vr, u64 frame,
        RenderViewId id, u64 generation, DiDenoiserSignal signal, const RestirSettings& settings) const
    {
        if (signal != DiDenoiserSignal::Diffuse && signal != DiDenoiserSignal::Specular)
            throw std::invalid_argument("DI upscale: invalid signal");
        DiUpscaleBindings native;
        if (!vr.diUpscale) return native;
        const auto& state = *vr.diUpscale;
        const u32 channel = signal == DiDenoiserSignal::Specular ? 1u : 0u;
        const auto& owner = state.denoisers[channel];
        if (state.id != id || !owner || owner->id != id || owner->signal != signal)
            throw std::invalid_argument("DI upscale: incompatible view owner");
        native.signal = signal; native.view = id; native.generation = generation; native.frameIndex = frame;
        native.retained = vr.diUpscale;
        if (m_UpscalePipeline) {
            native.pipeline = m_UpscalePipeline->GetHandle(); native.layout = m_UpscalePipeline->GetLayout();
        }
        native.set = state.sets[channel];
        native.globalSet = vr.globalDescriptorSet[frame % MAX_FRAMES_IN_FLIGHT];
        native.width = owner->width; native.height = owner->height;
        native.fullWidth = vr.width; native.fullHeight = vr.height;
        native.phiDepth = settings.spatialDepthThreshold;
        auto freeze = [](const std::shared_ptr<Texture>& texture, TextureBindingRef& binding, VkImage& image, VkImageView& view) {
            if (!texture) return;
            const auto vk = std::static_pointer_cast<VKTexture>(texture);
            binding = {texture.get()}; image = vk->GetImage(); view = vk->GetImageView();
        };
        freeze(owner->svgfDiHalf, native.sources[0], native.sourceImages[0], native.sourceViews[0]);
        for (u32 i = 0; i < 2; ++i) {
            freeze(state.sources[i], native.sources[i + 1], native.sourceImages[i + 1], native.sourceViews[i + 1]);
            native.sourceViews[i + 1] = state.sourceViews[i];
        }
        freeze(owner->svgfDenoised, native.output, native.outputImage, native.outputView);
        return native;
    }

    RG::ResourceHandle RtRestirSubsystem::AddUpscalePass(RG::RenderGraph& graph,
        const std::array<RG::ResourceHandle, 3>& inputs, const DiUpscaleBindings& native)
    {
        if (!inputs[0].IsValid() || !native.Ready()) return inputs[0];
        // At a 1x1 extent, half resolution is already full resolution. The denoiser
        // writes the final image directly; do not sample its unused working image.
        if (inputs[0].index <= graph.GetResources().size() && native.outputImage &&
            graph.GetResources()[inputs[0].index - 1].image == native.outputImage)
            return inputs[0];
        struct Data { RG::ResourceHandle half, depth, normal, out; };
        RG::ResourceHandle output;
        const bool specular = native.signal == DiDenoiserSignal::Specular;
        const UpscalePC pc{static_cast<i32>(native.fullWidth), static_cast<i32>(native.fullHeight),
            static_cast<i32>(native.width), static_cast<i32>(native.height), native.phiDepth, native.phiNormal};
        graph.AddComputePass<Data>(specular ? "DiSpecUpscale" : "DiUpscale", RG::QueueFamily::AsyncCompute,
            [&](Data& data, RG::RenderPassBuilder& builder) {
                data.half = builder.ReadStorageImageGeneral(inputs[0]);
                if (inputs[1].IsValid()) data.depth = builder.ReadStorageImage(inputs[1]);
                if (inputs[2].IsValid()) data.normal = builder.ReadStorageImage(inputs[2]);
                RG::TextureDesc desc; desc.name = specular ? "SvgfDiSpecDenoised" : "SvgfDenoised";
                desc.width = native.fullWidth; desc.height = native.fullHeight; desc.format = RG::TextureFormat::RGBA16_Float;
                data.out = builder.WriteStorageImage(graph.ImportResource(desc, (void*)native.outputImage,
                    (void*)native.outputView, RG::ResourceState::Undefined));
                output = data.out;
            }, [native, pc](Data&, RG::RenderPassContext& ctx) {
                const VkDescriptorSet sets[]{native.globalSet, native.set};
                vkCmdBindPipeline(ctx.commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, native.pipeline);
                vkCmdBindDescriptorSets(ctx.commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, native.layout, 0, 2, sets, 0, nullptr);
                vkCmdPushConstants(ctx.commandBuffer, native.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
                vkCmdDispatch(ctx.commandBuffer, (native.fullWidth + 7) / 8, (native.fullHeight + 7) / 8, 1);
            });
        return output;
    }
}

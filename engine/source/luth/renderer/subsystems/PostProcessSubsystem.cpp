#include "luthpch.h"
#include "luth/renderer/subsystems/PostProcessSubsystem.h"

#include "luth/renderer/Renderer.h"
#include "luth/renderer/FrameTargets.h"
#include "luth/renderer/material/Material.h"
#include "luth/renderer/draw/DrawCommand.h"
#include "luth/renderer/shader/ShaderLibrary.h"
#include "luth/renderer/backend/vulkan/VulkanContext.h"
#include "luth/renderer/backend/vulkan/VulkanTexture.h"
#include "luth/renderer/backend/vulkan/VulkanBuffer.h"
#include "luth/renderer/settings/PostProcessSettings.h"
#include "luth/core/FrameData.h"
#include "luth/core/time/Time.h"
#include "luth/jobs/JobSystem.h"
#include "luth/memory/GPUTaggedPageAllocator.h"

namespace Luth
{
    void PostProcessSubsystem::Init()
    {
        LH_PROFILE_FUNCTION();

        VkDevice device = VulkanContext::Get().GetDevice();

        VkSamplerCreateInfo samplerInfo{ VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO };
        samplerInfo.magFilter    = VK_FILTER_LINEAR;
        samplerInfo.minFilter    = VK_FILTER_LINEAR;
        samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        samplerInfo.mipmapMode   = VK_SAMPLER_MIPMAP_MODE_LINEAR;
        vkCreateSampler(device, &samplerInfo, nullptr, &m_Sampler);

        // Nearest sampler for the slim G-buffer matID binding (R16_UINT: integer formats lack
        // SAMPLED_IMAGE_FILTER_LINEAR_BIT, so binding the LINEAR m_Sampler trips VUID 04553).
        VkSamplerCreateInfo nearestInfo{ VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO };
        nearestInfo.magFilter    = VK_FILTER_NEAREST;
        nearestInfo.minFilter    = VK_FILTER_NEAREST;
        nearestInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        nearestInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        nearestInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        nearestInfo.mipmapMode   = VK_SAMPLER_MIPMAP_MODE_NEAREST;
        vkCreateSampler(device, &nearestInfo, nullptr, &m_NearestSampler);

        // Composite descriptor layout:
        //   binding 0 = sampler2D (HDR), binding 1 = sampler2D (bloom mip0), binding 2 = UBO.
        VkDescriptorSetLayoutBinding bindings[3] = {};
        bindings[0].binding = 0;
        bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        bindings[0].descriptorCount = 1;
        bindings[0].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        bindings[1].binding = 1;
        bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        bindings[1].descriptorCount = 1;
        bindings[1].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        bindings[2].binding = 2;
        bindings[2].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        bindings[2].descriptorCount = 1;
        bindings[2].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

        // invariant: binding 2 (PP UBO) is rewritten per render-stage; cycling alone
        // doesn't avoid the in-pending-cmdbuf race in practice. UAB needed.
        VkDescriptorBindingFlags bindingFlags[3] = {
            VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT,
            VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT,
            VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT,
        };
        VkDescriptorSetLayoutBindingFlagsCreateInfo bindingFlagsCI{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO };
        bindingFlagsCI.bindingCount  = 3;
        bindingFlagsCI.pBindingFlags = bindingFlags;

        VkDescriptorSetLayoutCreateInfo layoutInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
        layoutInfo.pNext        = &bindingFlagsCI;
        layoutInfo.flags        = VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT;
        layoutInfo.bindingCount = 3;
        layoutInfo.pBindings    = bindings;
        vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &m_DescSetLayout);

        // Bloom pyramid compute layout: b0 = source mip (COMBINED_IMAGE_SAMPLER), b1 = dest mip
        // (STORAGE_IMAGE), both COMPUTE. b0 is UAB so the prefilter's per-frame source rebind
        // (PrepareBloomBindings) is race-safe; the single down/up sets bind stable per-view mips.
        VkDescriptorSetLayoutBinding bloomBindings[2] = {};
        bloomBindings[0].binding         = 0;
        bloomBindings[0].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        bloomBindings[0].descriptorCount = 1;
        bloomBindings[0].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;
        bloomBindings[1].binding         = 1;
        bloomBindings[1].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        bloomBindings[1].descriptorCount = 1;
        bloomBindings[1].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;
        VkDescriptorBindingFlags bloomFlags[2] = {
            VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT,
            VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT,
        };
        VkDescriptorSetLayoutBindingFlagsCreateInfo bloomFlagsCI{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO };
        bloomFlagsCI.bindingCount  = 2;
        bloomFlagsCI.pBindingFlags = bloomFlags;
        VkDescriptorSetLayoutCreateInfo bloomLayoutInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
        bloomLayoutInfo.pNext        = &bloomFlagsCI;
        bloomLayoutInfo.flags        = VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT;
        bloomLayoutInfo.bindingCount = 2;
        bloomLayoutInfo.pBindings    = bloomBindings;
        vkCreateDescriptorSetLayout(device, &bloomLayoutInfo, nullptr, &m_BloomComputeLayout);

        // Slim viz descriptor set layout: 4 sampler bindings (normal/roughness/motion/matID).
        // Written once on domain state creation. Replacement waits for GPU completion;
        // stable bindings need no UAB and retire with their local descriptor pool.
        VkDescriptorSetLayoutBinding slimBindings[4] = {};
        for (u32 i = 0; i < 4; ++i)
        {
            slimBindings[i].binding         = i;
            slimBindings[i].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            slimBindings[i].descriptorCount = 1;
            slimBindings[i].stageFlags      = VK_SHADER_STAGE_FRAGMENT_BIT;
        }
        VkDescriptorSetLayoutCreateInfo slimLayoutInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
        slimLayoutInfo.bindingCount = 4;
        slimLayoutInfo.pBindings    = slimBindings;
        vkCreateDescriptorSetLayout(device, &slimLayoutInfo, nullptr, &m_SlimVizDescSetLayout);

        // TAA Resolve descriptor set layout (Karis14 YCoCg-clip recipe).
        //   0 = sceneColor (sampler2D, current HDR after volumetric composite)
        //   1 = motion vectors (sampler2D, RG16F NDC delta from SlimGBufferPass)
        //   2 = history-prev (sampler2D, cycled UAB; parity-picked taa->historyA/B each frame)
        //   3 = sceneDepth (sampler2D, for closest-depth velocity dilation in resolve)
        //   4 = PP UBO (shared with bloom/composite sets; rewritten per render-stage)
        // All bindings UAB so binding 2's per-frame parity-rewrite is race-safe.
        VkDescriptorSetLayoutBinding taaBindings[5] = {};
        for (u32 i = 0; i < 4; ++i)
        {
            taaBindings[i].binding         = i;
            taaBindings[i].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            taaBindings[i].descriptorCount = 1;
            taaBindings[i].stageFlags      = VK_SHADER_STAGE_FRAGMENT_BIT;
        }
        taaBindings[4].binding         = 4;
        taaBindings[4].descriptorType  = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        taaBindings[4].descriptorCount = 1;
        taaBindings[4].stageFlags      = VK_SHADER_STAGE_FRAGMENT_BIT;
        VkDescriptorBindingFlags taaFlags[5] = {
            VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT,
            VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT,
            VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT,
            VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT,
            VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT,
        };
        VkDescriptorSetLayoutBindingFlagsCreateInfo taaFlagsCI{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO };
        taaFlagsCI.bindingCount  = 5;
        taaFlagsCI.pBindingFlags = taaFlags;
        VkDescriptorSetLayoutCreateInfo taaLayoutInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
        taaLayoutInfo.pNext        = &taaFlagsCI;
        taaLayoutInfo.flags        = VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT;
        taaLayoutInfo.bindingCount = 5;
        taaLayoutInfo.pBindings    = taaBindings;
        vkCreateDescriptorSetLayout(device, &taaLayoutInfo, nullptr, &m_TaaResolveDescSetLayout);

        auto loadSpv = [](const char* relPath) -> std::vector<u32> {
            auto sh = ShaderLibrary::LoadEngine(relPath);
            return sh ? sh->GetSpirV() : std::vector<u32>{};
        };
        m_FullscreenVertSpv   = loadSpv("shaders/fullscreen.slang");
        m_BloomDownSpv        = loadSpv("shaders/bloom_downsample.slang");
        m_BloomUpSpv          = loadSpv("shaders/bloom_upsample.slang");
        m_PostProcessFragSpv  = loadSpv("shaders/postprocess.slang");
        m_SlimVizFragSpv      = loadSpv("shaders/slim_viz.slang");
        m_TaaResolveFragSpv   = loadSpv("shaders/taa_resolve.slang");

        if (m_FullscreenVertSpv.empty() || m_BloomDownSpv.empty() ||
            m_BloomUpSpv.empty() || m_PostProcessFragSpv.empty() ||
            m_SlimVizFragSpv.empty() || m_TaaResolveFragSpv.empty())
        {
            LH_LOG(Renderer, error, "PostProcessSubsystem: shader SPIR-V empty after asset load!");
            return;
        }

        BuildPipelines();
    }

    void PostProcessSubsystem::BuildPipelines()
    {
        LH_PROFILE_FUNCTION();
        std::vector<VkDescriptorSetLayout> ppLayouts = { m_DescSetLayout };

        // Bloom pyramid compute pipelines (shared 2-binding layout). Downsample doubles as the
        // prefilter via its push-constant flag; upsample tent-blends additively back up.
        std::vector<VkDescriptorSetLayout> bloomLayouts = { m_BloomComputeLayout };
        if (!m_BloomDownSpv.empty())
        {
            VkPushConstantRange pc{ VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(BloomDownPC) };
            m_BloomDownPipeline = std::make_unique<VKComputePipeline>(
                m_BloomDownSpv, bloomLayouts, std::vector<VkPushConstantRange>{ pc });
        }
        if (!m_BloomUpSpv.empty())
        {
            VkPushConstantRange pc{ VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(BloomUpPC) };
            m_BloomUpPipeline = std::make_unique<VKComputePipeline>(
                m_BloomUpSpv, bloomLayouts, std::vector<VkPushConstantRange>{ pc });
        }
        if (!m_PostProcessFragSpv.empty())
        {
            PipelineConfig cfg;
            cfg.colorFormats = { VK_FORMAT_R8G8B8A8_UNORM };
            cfg.depthFormat  = VK_FORMAT_UNDEFINED;
            cfg.depthTest    = false; cfg.depthWrite = false;
            cfg.blendEnabled = false;
            cfg.cullMode     = VK_CULL_MODE_NONE;
            m_PostProcessPipeline = std::make_unique<VKPipeline>(
                cfg, m_FullscreenVertSpv, m_PostProcessFragSpv, ppLayouts);
        }

        // Slim G-buffer viz pipeline (live ShadeMode toggle). Push constants: mode + scale = 8B.
        if (!m_SlimVizFragSpv.empty())
        {
            std::vector<VkDescriptorSetLayout> slimLayouts = { m_SlimVizDescSetLayout };
            VkPushConstantRange slimPC{ VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(u32) + sizeof(float) };
            PipelineConfig cfg;
            cfg.colorFormats = { VK_FORMAT_R8G8B8A8_UNORM };
            cfg.depthFormat  = VK_FORMAT_UNDEFINED;
            cfg.depthTest    = false; cfg.depthWrite = false;
            cfg.blendEnabled = false;
            cfg.cullMode     = VK_CULL_MODE_NONE;
            cfg.pushConstantRanges = { slimPC };
            m_SlimVizPipeline = std::make_unique<VKPipeline>(
                cfg, m_FullscreenVertSpv, m_SlimVizFragSpv, slimLayouts);
        }

        // TAA Resolve pipeline. Output to RGBA16F (HDR history texture); push constant carries
        // temporalAlpha (jitter delta moved to slim_gbuffer.slang as source-side de-jitter).
        // No depth, no blend: opaque write.
        if (!m_TaaResolveFragSpv.empty())
        {
            std::vector<VkDescriptorSetLayout> taaLayouts = { m_TaaResolveDescSetLayout };
            VkPushConstantRange taaPC{ VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(TaaResolvePushConstants) };
            PipelineConfig cfg;
            cfg.colorFormats = { VK_FORMAT_R16G16B16A16_SFLOAT };
            cfg.depthFormat  = VK_FORMAT_UNDEFINED;
            cfg.depthTest    = false; cfg.depthWrite = false;
            cfg.blendEnabled = false;
            cfg.cullMode     = VK_CULL_MODE_NONE;
            cfg.pushConstantRanges = { taaPC };
            m_TaaResolvePipeline = std::make_unique<VKPipeline>(
                cfg, m_FullscreenVertSpv, m_TaaResolveFragSpv, taaLayouts);
        }
    }

    void PostProcessSubsystem::Shutdown()
    {
        LH_PROFILE_FUNCTION();
        VkDevice device = VulkanContext::Get().GetDevice();
        m_SlimVizStates.ReleaseAll([] { Renderer::WaitForGPU(); });
        m_CompositeStates.ReleaseAll([] { Renderer::WaitForGPU(); });
        m_TaaStates.ReleaseAll([] { Renderer::WaitForGPU(); });
        m_BloomStates.ReleaseAll([] { Renderer::WaitForGPU(); });
        m_TaaResolvePipeline.reset();
        m_SlimVizPipeline.reset();
        m_PostProcessPipeline.reset();
        m_BloomUpPipeline.reset();
        m_BloomDownPipeline.reset();
        if (m_Sampler)              { vkDestroySampler(device, m_Sampler, nullptr); m_Sampler = VK_NULL_HANDLE; }
        if (m_NearestSampler)       { vkDestroySampler(device, m_NearestSampler, nullptr); m_NearestSampler = VK_NULL_HANDLE; }
        if (m_DescSetLayout)           { vkDestroyDescriptorSetLayout(device, m_DescSetLayout, nullptr); m_DescSetLayout = VK_NULL_HANDLE; }
        if (m_BloomComputeLayout)      { vkDestroyDescriptorSetLayout(device, m_BloomComputeLayout, nullptr); m_BloomComputeLayout = VK_NULL_HANDLE; }
        if (m_SlimVizDescSetLayout)    { vkDestroyDescriptorSetLayout(device, m_SlimVizDescSetLayout, nullptr); m_SlimVizDescSetLayout = VK_NULL_HANDLE; }
        if (m_TaaResolveDescSetLayout) { vkDestroyDescriptorSetLayout(device, m_TaaResolveDescSetLayout, nullptr); m_TaaResolveDescSetLayout = VK_NULL_HANDLE; }
    }

    bool PostProcessSubsystem::OnShaderReloaded(const std::string& name, const std::vector<u32>& spv)
    {
        LH_PROFILE_FUNCTION();
        auto deferGfx = [](std::unique_ptr<VKPipeline>& p) {
            if (auto* raw = p.release(); raw)
                VulkanContext::Get().PushDeletion([raw]() { delete raw; });
        };
        auto deferComp = [](std::unique_ptr<VKComputePipeline>& p) {
            if (auto* raw = p.release(); raw)
                VulkanContext::Get().PushDeletion([raw]() { delete raw; });
        };

        if      (name == "fullscreen.slang")        m_FullscreenVertSpv  = spv;
        else if (name == "bloom_downsample.slang") m_BloomDownSpv       = spv;
        else if (name == "bloom_upsample.slang")   m_BloomUpSpv         = spv;
        else if (name == "postprocess.slang")       m_PostProcessFragSpv = spv;
        else if (name == "slim_viz.slang")          m_SlimVizFragSpv     = spv;
        else if (name == "taa_resolve.slang")       m_TaaResolveFragSpv  = spv;
        else return false;

        deferComp(m_BloomDownPipeline);
        deferComp(m_BloomUpPipeline);
        deferGfx(m_PostProcessPipeline);
        deferGfx(m_SlimVizPipeline);
        deferGfx(m_TaaResolvePipeline);
        BuildPipelines();
        if ((name == "taa_resolve.slang" || name == "fullscreen.slang") &&
            m_TaaResolvePipeline && m_TaaResolvePipeline->GetHandle())
            ++m_TaaShaderGeneration;
        return true;
    }

    void PostProcessSubsystem::WriteSlimVizView(SlimVizViewState& state)
    {
        for (const auto& source : state.sources)
            if (!source) throw std::invalid_argument("SlimViz: missing stable source");
        if (!state.set || !m_Sampler || !m_NearestSampler) return;
        std::array<VkDescriptorImageInfo, 4> infos{};
        std::array<VkWriteDescriptorSet, 4> writes{};
        for (u32 i = 0; i < 4; ++i)
        {
            const auto texture = std::static_pointer_cast<VKTexture>(state.sources[i]);
            infos[i].sampler = i == 3 ? m_NearestSampler : m_Sampler;
            infos[i].imageView = texture->GetImageView(); infos[i].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            writes[i] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET}; writes[i].dstSet = state.set;
            writes[i].dstBinding = i; writes[i].descriptorCount = 1;
            writes[i].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; writes[i].pImageInfo = &infos[i];
        }
        vkUpdateDescriptorSets(VulkanContext::Get().GetDevice(), 4, writes.data(), 0, nullptr);
    }
    void PostProcessSubsystem::WriteBloomView(BloomViewState& state)
    {
        LH_PROFILE_FUNCTION();
        if (!state.mips[0]) return;
        VkDevice device = VulkanContext::Get().GetDevice();

        // Per-mip image infos must outlive the single vkUpdateDescriptorSets; hold them in arrays.
        // sampled[i] = SHADER_READ_ONLY (filtered taps); storage[i] = GENERAL (imageStore dest).
        std::array<VkDescriptorImageInfo, BloomViewState::kMipCount> sampled{};
        std::array<VkDescriptorImageInfo, BloomViewState::kMipCount> storage{};
        for (u32 i = 0; i < BloomViewState::kMipCount; ++i)
        {
            VkImageView view = std::static_pointer_cast<VKTexture>(state.mips[i])->GetImageView();
            sampled[i].sampler     = m_Sampler;
            sampled[i].imageView   = view;
            sampled[i].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            storage[i].imageView   = view;
            storage[i].imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        }

        std::vector<VkWriteDescriptorSet> writes;
        writes.reserve(MAX_FRAMES_IN_FLIGHT + 4 * (BloomViewState::kMipCount - 1));
        auto add = [&](VkDescriptorSet set, u32 binding, VkDescriptorType type, const VkDescriptorImageInfo* info) {
            if (set == VK_NULL_HANDLE) return;
            VkWriteDescriptorSet w{ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
            w.dstSet          = set;
            w.dstBinding      = binding;
            w.descriptorCount = 1;
            w.descriptorType  = type;
            w.pImageInfo      = info;
            writes.push_back(w);
        };

        // Prefilter dest (b1 = mip0 storage); b0 source is rebound per frame by PrepareBloomBindings.
        for (u32 s = 0; s < MAX_FRAMES_IN_FLIGHT; ++s)
            add(state.prefilterSets[s], 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &storage[0]);

        for (u32 i = 0; i < BloomViewState::kMipCount - 1; ++i)
        {
            // Downsample i: mip[i] (sampled) -> mip[i+1] (storage).
            add(state.downSets[i], 0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &sampled[i]);
            add(state.downSets[i], 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,          &storage[i + 1]);
            // Upsample i: mip[i+1] (sampled) -> mip[i] (storage, additive RMW).
            add(state.upSets[i],   0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &sampled[i + 1]);
            add(state.upSets[i],   1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,          &storage[i]);
        }

        if (!writes.empty())
            vkUpdateDescriptorSets(device, static_cast<u32>(writes.size()), writes.data(), 0, nullptr);
    }

    void PostProcessSubsystem::WriteTaaResolveView(TaaViewState& state, FrameTargets& targets)
    {
        LH_PROFILE_FUNCTION();
        // Bindings 0/1/3 are stable per-view-resize; write once across all cycled slots.
        // Binding 2 (history-prev sampler) cycles per-frame in WriteTaaResolvePerFrame.
        // Binding 4 (UBO) is declared in the layout but unused by the current shader.
        if (state.resolveSets[0] == VK_NULL_HANDLE) return;

        auto sceneTex  = std::static_pointer_cast<VKTexture>(targets.GetSceneColor());
        auto motionTex = std::static_pointer_cast<VKTexture>(targets.GetSlimMotion());
        auto depthTex  = std::static_pointer_cast<VKTexture>(targets.GetSceneDepth());
        if (!sceneTex || !motionTex || !depthTex) return;

        auto makeImg = [&](VkImageView v) {
            VkDescriptorImageInfo info{};
            info.sampler     = m_Sampler;
            info.imageView   = v;
            info.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            return info;
        };
        VkDescriptorImageInfo sceneInfo  = makeImg(sceneTex->GetImageView());
        VkDescriptorImageInfo motionInfo = makeImg(motionTex->GetImageView());
        // Depth: aspect is DEPTH only; layout is DEPTH_STENCIL_READ_ONLY_OPTIMAL once the prepass
        // completes. SHADER_READ_ONLY_OPTIMAL works as well because the RG will transition into
        // a read-compatible state when the pass declares the depth Read.
        VkDescriptorImageInfo depthInfo  = makeImg(depthTex->GetImageView());

        VkWriteDescriptorSet writes[3 * MAX_FRAMES_IN_FLIGHT] = {};
        u32 idx = 0;
        auto addImg = [&](VkDescriptorSet set, u32 binding, VkDescriptorImageInfo* info) {
            writes[idx] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
            writes[idx].dstSet          = set;
            writes[idx].dstBinding      = binding;
            writes[idx].descriptorCount = 1;
            writes[idx].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            writes[idx].pImageInfo      = info;
            ++idx;
        };
        for (u32 s = 0; s < MAX_FRAMES_IN_FLIGHT; ++s)
        {
            addImg(state.resolveSets[s], 0, &sceneInfo);
            addImg(state.resolveSets[s], 1, &motionInfo);
            addImg(state.resolveSets[s], 3, &depthInfo);
        }
        vkUpdateDescriptorSets(VulkanContext::Get().GetDevice(), idx, writes, 0, nullptr);
    }

    void PostProcessSubsystem::WriteTaaResolvePerFrame(TaaViewState& state, u64 frameAbs)
    {
        LH_PROFILE_FUNCTION();
        // Binding 2 = history-prev sampler. Even frames read HistB and write HistA;
        // odd frames read HistA and write HistB. The write target is a color attachment
        // via the RG (not in this descriptor set), so we only rebind the READ side here.
        if (!state.historyA || !state.historyB) return;
        const u32 slot = frameAbs % MAX_FRAMES_IN_FLIGHT;
        if (state.resolveSets[slot] == VK_NULL_HANDLE) return;

        const bool parity = (frameAbs & 1u) != 0u;
        auto vkPrev = std::static_pointer_cast<VKTexture>(parity ? state.historyA : state.historyB);

        VkDescriptorImageInfo prevInfo{};
        prevInfo.sampler     = m_Sampler;
        prevInfo.imageView   = vkPrev->GetImageView();
        prevInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

        VkWriteDescriptorSet write{ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
        write.dstSet          = state.resolveSets[slot];
        write.dstBinding      = 2;
        write.descriptorCount = 1;
        write.descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        write.pImageInfo      = &prevInfo;
        vkUpdateDescriptorSets(VulkanContext::Get().GetDevice(), 1, &write, 0, nullptr);
    }

}

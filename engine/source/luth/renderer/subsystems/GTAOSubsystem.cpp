#include "luthpch.h"
#include "luth/renderer/subsystems/GTAOSubsystem.h"
#include "luth/renderer/FrameDebugger.h"
#include "luth/renderer/CameraParams.h"
#include "luth/renderer/Renderer.h"
#include "luth/renderer/shader/ShaderLibrary.h"
#include "luth/renderer/backend/vulkan/VulkanContext.h"
#include "luth/renderer/backend/vulkan/VulkanTexture.h"
#include "luth/renderer/backend/vulkan/VulkanBuffer.h"
#include "luth/renderer/settings/GTAOSettings.h"
#include "luth/core/FrameData.h"
#include "luth/jobs/JobSystem.h"
#include "luth/memory/GPUTaggedPageAllocator.h"

#include <cmath>

namespace Luth
{
    void GTAOSubsystem::Init()
    {
        LH_PROFILE_FUNCTION();
        VkDevice device = VulkanContext::Get().GetDevice();

        VkSamplerCreateInfo sampCI{ VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO };
        sampCI.magFilter    = VK_FILTER_LINEAR;
        sampCI.minFilter    = VK_FILTER_LINEAR;
        sampCI.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        sampCI.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        sampCI.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        sampCI.mipmapMode   = VK_SAMPLER_MIPMAP_MODE_NEAREST;
        vkCreateSampler(device, &sampCI, nullptr, &m_Sampler);

        // Prefilter layout: [sampler2D sceneDepth, image2D linearDepth]
        {
            VkDescriptorSetLayoutBinding bindings[2]{};
            bindings[0].binding         = 0;
            bindings[0].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            bindings[0].descriptorCount = 1;
            bindings[0].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;
            bindings[1].binding         = 1;
            bindings[1].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            bindings[1].descriptorCount = 1;
            bindings[1].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;

            VkDescriptorSetLayoutCreateInfo layoutCI{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
            layoutCI.bindingCount = 2;
            layoutCI.pBindings    = bindings;
            vkCreateDescriptorSetLayout(device, &layoutCI, nullptr, &m_PrefilterDescLayout);

            VkPushConstantRange pcRange{ VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(i32) * 2 + sizeof(float) * 6 };

            if (auto sh = ShaderLibrary::LoadEngine("shaders/gtao_depth_prefilter.slang"))
                m_PrefilterSpv = sh->GetSpirV();
            if (m_PrefilterSpv.empty())
            {
                LH_LOG(Renderer, error, "GTAOSubsystem: failed to load gtao_depth_prefilter.slang!");
                return;
            }
            m_PrefilterPipeline = std::make_unique<VKComputePipeline>(
                m_PrefilterSpv,
                std::vector<VkDescriptorSetLayout>{ m_PrefilterDescLayout },
                std::vector<VkPushConstantRange>{ pcRange });
        }

        // Main layout: [sampler2D linearDepth, image2D rawAO, UBO]
        {
            VkDescriptorSetLayoutBinding bindings[3]{};
            bindings[0].binding         = 0;
            bindings[0].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            bindings[0].descriptorCount = 1;
            bindings[0].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;
            bindings[1].binding         = 1;
            bindings[1].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            bindings[1].descriptorCount = 1;
            bindings[1].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;
            bindings[2].binding         = 2;
            bindings[2].descriptorType  = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            bindings[2].descriptorCount = 1;
            bindings[2].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;

            // invariant: cycling alone doesn't avoid the in-pending-cmdbuf race for
            // these per-render-stage rewrites; UAB needed (validation 03047).
            VkDescriptorBindingFlags bindingFlags[3] = {
                VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT,
                VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT,
                VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT,
            };
            VkDescriptorSetLayoutBindingFlagsCreateInfo bindingFlagsCI{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO };
            bindingFlagsCI.bindingCount  = 3;
            bindingFlagsCI.pBindingFlags = bindingFlags;

            VkDescriptorSetLayoutCreateInfo layoutCI{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
            layoutCI.pNext        = &bindingFlagsCI;
            layoutCI.flags        = VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT;
            layoutCI.bindingCount = 3;
            layoutCI.pBindings    = bindings;
            vkCreateDescriptorSetLayout(device, &layoutCI, nullptr, &m_MainDescLayout);

            VkPushConstantRange pcRange{ VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(float) * 4 + sizeof(u32) * 4 };

            if (auto sh = ShaderLibrary::LoadEngine("shaders/gtao_main.slang"))
                m_MainSpv = sh->GetSpirV();
            if (m_MainSpv.empty())
            {
                LH_LOG(Renderer, error, "GTAOSubsystem: failed to load gtao_main.slang!");
                return;
            }
            m_MainPipeline = std::make_unique<VKComputePipeline>(
                m_MainSpv,
                std::vector<VkDescriptorSetLayout>{ m_MainDescLayout },
                std::vector<VkPushConstantRange>{ pcRange });
        }

        // Denoise layout: [sampler2D rawAO, sampler2D linDepth, image2D finalAO]
        {
            VkDescriptorSetLayoutBinding bindings[3]{};
            bindings[0].binding         = 0;
            bindings[0].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            bindings[0].descriptorCount = 1;
            bindings[0].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;
            bindings[1].binding         = 1;
            bindings[1].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            bindings[1].descriptorCount = 1;
            bindings[1].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;
            bindings[2].binding         = 2;
            bindings[2].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            bindings[2].descriptorCount = 1;
            bindings[2].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;

            VkDescriptorSetLayoutCreateInfo layoutCI{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
            layoutCI.bindingCount = 3;
            layoutCI.pBindings    = bindings;
            vkCreateDescriptorSetLayout(device, &layoutCI, nullptr, &m_DenoiseDescLayout);

            if (auto sh = ShaderLibrary::LoadEngine("shaders/gtao_denoise.slang"))
                m_DenoiseSpv = sh->GetSpirV();
            if (m_DenoiseSpv.empty())
            {
                LH_LOG(Renderer, error, "GTAOSubsystem: failed to load gtao_denoise.slang!");
                return;
            }
            m_DenoisePipeline = std::make_unique<VKComputePipeline>(
                m_DenoiseSpv,
                std::vector<VkDescriptorSetLayout>{ m_DenoiseDescLayout },
                std::vector<VkPushConstantRange>{});
        }
    }

    void GTAOSubsystem::Shutdown()
    {
        LH_PROFILE_FUNCTION();
        VkDevice device = VulkanContext::Get().GetDevice();
        m_PrefilterPipeline.reset();
        m_MainPipeline.reset();
        m_DenoisePipeline.reset();
        if (m_Sampler)             vkDestroySampler(device, m_Sampler, nullptr);
        if (m_PrefilterDescLayout) vkDestroyDescriptorSetLayout(device, m_PrefilterDescLayout, nullptr);
        if (m_MainDescLayout)      vkDestroyDescriptorSetLayout(device, m_MainDescLayout, nullptr);
        if (m_DenoiseDescLayout)   vkDestroyDescriptorSetLayout(device, m_DenoiseDescLayout, nullptr);
        m_Sampler             = VK_NULL_HANDLE;
        m_PrefilterDescLayout = VK_NULL_HANDLE;
        m_MainDescLayout      = VK_NULL_HANDLE;
        m_DenoiseDescLayout   = VK_NULL_HANDLE;
    }

    bool GTAOSubsystem::OnShaderReloaded(const std::string& name, const std::vector<u32>& spv)
    {
        LH_PROFILE_FUNCTION();
        auto deferComp = [](std::unique_ptr<VKComputePipeline>& p) {
            if (auto* raw = p.release(); raw)
                VulkanContext::Get().PushDeletion([raw]() { delete raw; });
        };

        if (name == "gtao_depth_prefilter.slang" && m_PrefilterDescLayout)
        {
            m_PrefilterSpv = spv;
            deferComp(m_PrefilterPipeline);
            VkPushConstantRange pc{ VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(i32) * 2 + sizeof(float) * 6 };
            m_PrefilterPipeline = std::make_unique<VKComputePipeline>(m_PrefilterSpv,
                std::vector<VkDescriptorSetLayout>{ m_PrefilterDescLayout },
                std::vector<VkPushConstantRange>{ pc });
            return true;
        }
        if (name == "gtao_main.slang" && m_MainDescLayout)
        {
            m_MainSpv = spv;
            deferComp(m_MainPipeline);
            VkPushConstantRange pc{ VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(float) * 4 + sizeof(u32) * 4 };
            m_MainPipeline = std::make_unique<VKComputePipeline>(m_MainSpv,
                std::vector<VkDescriptorSetLayout>{ m_MainDescLayout },
                std::vector<VkPushConstantRange>{ pc });
            return true;
        }
        if (name == "gtao_denoise.slang" && m_DenoiseDescLayout)
        {
            m_DenoiseSpv = spv;
            deferComp(m_DenoisePipeline);
            m_DenoisePipeline = std::make_unique<VKComputePipeline>(m_DenoiseSpv,
                std::vector<VkDescriptorSetLayout>{ m_DenoiseDescLayout },
                std::vector<VkPushConstantRange>{});
            return true;
        }
        return false;
    }

    void GTAOSubsystem::UpdateUBO(GtaoViewState& vr,
        const std::array<VkDescriptorSet, MAX_FRAMES_IN_FLIGHT>& globalSets,
        const GTAOSettings& s, u64 renderFrameIndex)
    {
        LH_PROFILE_FUNCTION();
        if (globalSets[0] == VK_NULL_HANDLE) return;
        GTAOUBO ubo{};
        ubo.intensity      = s.intensity;
        ubo.radius         = s.radius;
        ubo.falloff        = s.falloff;
        ubo.power          = s.power;
        ubo.sliceCount     = s.sliceCount;
        ubo.stepsPerSlice  = s.stepsPerSlice;
        ubo.enabled        = s.enabled  ? 1 : 0;
        vr.uniformEnabled = s.enabled;
        ubo.visualize      = s.visualize ? 1 : 0;

        // Derive full-res from the view's half-res GTAO textures.
        const auto& lin = vr.linearDepth;
        const u32 halfW = lin ? lin->GetWidth()  : 1u;
        const u32 halfH = lin ? lin->GetHeight() : 1u;
        const u32 fullW = halfW * 2;
        const u32 fullH = halfH * 2;
        ubo.invResolution[0]     = 1.0f / float(halfW);
        ubo.invResolution[1]     = 1.0f / float(halfH);
        ubo.invFullResolution[0] = 1.0f / float(fullW);
        ubo.invFullResolution[1] = 1.0f / float(fullH);

        // invariant: Set 0 binding 5 + GTAO main set binding 2 share the same per-frame region AND the
        // same per-frame slot. The two writes MUST stay in one batched call to avoid double-allocating,
        // and both must use the same `slot` so the next frame's allocator doesn't overwrite a region the
        // previous frame's binding still references.
        auto* jobCtx = JobSystem::GetCurrentJobContext();
        if (!jobCtx) return;
        const u32 frameAbs = static_cast<u32>(renderFrameIndex);
        const u32 slot     = frameAbs % MAX_FRAMES_IN_FLIGHT;
        jobCtx->GpuCache.CurrentTag = frameAbs;

        auto& heap   = Memory::GPUTaggedPageAllocator::Get();
        const u64 al = VulkanContext::Get().GetMinUniformBufferAlignment();
        Memory::GPUSubRegion region = heap.Allocate(jobCtx->GpuCache, sizeof(GTAOUBO), al);
        if (!region.buffer) return;

        memcpy(region.mappedPtr, &ubo, sizeof(GTAOUBO));
        heap.FlushRegion(region);

        VkDescriptorBufferInfo bi{};
        bi.buffer = region.buffer;
        bi.offset = region.offset;
        bi.range  = region.size;

        VkWriteDescriptorSet writes[2] = {};
        writes[0] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
        writes[0].dstSet          = globalSets[slot];
        writes[0].dstBinding      = 5;
        writes[0].descriptorType  = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        writes[0].descriptorCount = 1;
        writes[0].pBufferInfo     = &bi;

        u32 n = 1;
        if (vr.mainSets[0] != VK_NULL_HANDLE)
        {
            writes[1] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
            writes[1].dstSet          = vr.mainSets[slot];
            writes[1].dstBinding      = 2;
            writes[1].descriptorType  = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            writes[1].descriptorCount = 1;
            writes[1].pBufferInfo     = &bi;
            ++n;
        }

        vkUpdateDescriptorSets(VulkanContext::Get().GetDevice(), n, writes, 0, nullptr);
    }

    void GTAOSubsystem::EnsureView(GtaoViewStateStore& states, RenderViewId id,
        u32 width, u32 height, const Texture& depth)
    {
        LH_PROFILE_FUNCTION();
        if (!width || !height || depth.GetWidth() != width || depth.GetHeight() != height)
            throw std::invalid_argument("GTAO: depth extent does not match the view");
        auto& owned = states.Ensure(id, {width, height}, [&](const ViewStateConfig&) {
            auto state = std::make_shared<GtaoViewState>();
            state->width = width; state->height = height;
            state->halfWidth = std::max(width / 2, 1u);
            state->halfHeight = std::max(height / 2, 1u);
            auto makeImage = [&](TextureFormat format) {
                return std::make_shared<VKTexture>(state->halfWidth, state->halfHeight,
                    format, 1, 0u, 1, VK_IMAGE_USAGE_STORAGE_BIT);
            };
            state->linearDepth = makeImage(TextureFormat::R32_Float);
            state->rawAO = makeImage(TextureFormat::R8);
            state->edges = makeImage(TextureFormat::R8); // Preserve the legacy allocation set.
            state->finalAO = makeImage(TextureFormat::R8);
            auto binding = [](const std::shared_ptr<Texture>& texture) {
                const auto* native = static_cast<const VKTexture*>(texture.get());
                return GtaoImageBinding{native->GetImage(), native->GetImageView(), {texture.get()}};
            };
            state->linearBinding = binding(state->linearDepth);
            state->rawBinding = binding(state->rawAO);
            state->finalBinding = binding(state->finalAO);
            state->device = VulkanContext::Get().GetDevice();
            VkDescriptorPoolSize sizes[] = {
                {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, MAX_FRAMES_IN_FLIGHT},
                {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, MAX_FRAMES_IN_FLIGHT + 2},
                {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, MAX_FRAMES_IN_FLIGHT + 3}
            };
            VkDescriptorPoolCreateInfo pool{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
            pool.flags = VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT;
            pool.maxSets = MAX_FRAMES_IN_FLIGHT + 2;
            pool.poolSizeCount = 3; pool.pPoolSizes = sizes;
            if (vkCreateDescriptorPool(state->device, &pool, nullptr, &state->pool) != VK_SUCCESS)
                throw std::runtime_error("GTAO: descriptor pool allocation failed");
            auto allocate = [&](VkDescriptorSetLayout layout, VkDescriptorSet* sets, u32 count, const char* tag) {
                if (!layout) return;
                std::array<VkDescriptorSetLayout, MAX_FRAMES_IN_FLIGHT> layouts;
                layouts.fill(layout);
                VkDescriptorSetAllocateInfo alloc{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
                alloc.descriptorPool = state->pool;
                alloc.descriptorSetCount = count; alloc.pSetLayouts = layouts.data();
                if (vkAllocateDescriptorSets(state->device, &alloc, sets) != VK_SUCCESS)
                    throw std::runtime_error("GTAO: descriptor set allocation failed");
                for (u32 i = 0; i < count; ++i)
                {
                    const auto name = std::string(tag) + ".View" + std::to_string(id.value) + ".Slot" + std::to_string(i);
                    VulkanContext::SetDebugName(sets[i], name.c_str());
                }
            };
            allocate(m_PrefilterDescLayout, &state->prefilterSet, 1, "GTAO.Prefilter");
            allocate(m_MainDescLayout, state->mainSets.data(), MAX_FRAMES_IN_FLIGHT, "GTAO.Main");
            allocate(m_DenoiseDescLayout, &state->denoiseSet, 1, "GTAO.Denoise");
            WriteView(*state, depth);
            return state;
        }, [] { Renderer::WaitForGPU(); });
        if (owned->depthSource != &depth)
        {
            // A changed physical depth binding also requires a descriptor-safe point.
            Renderer::WaitForGPU();
            WriteView(*owned, depth);
        }
    }

    void GTAOSubsystem::WriteView(GtaoViewState& vr, const Texture& depth)
    {
        LH_PROFILE_FUNCTION();
        vr.depthSource = &depth;
        if (vr.prefilterSet == VK_NULL_HANDLE) return;

        VkDevice device = VulkanContext::Get().GetDevice();

        const auto* vkSceneDepth = static_cast<const VKTexture*>(&depth);
        auto vkLinDepth   = std::static_pointer_cast<VKTexture>(vr.linearDepth);
        auto vkRawAO      = std::static_pointer_cast<VKTexture>(vr.rawAO);
        auto vkFinalAO    = std::static_pointer_cast<VKTexture>(vr.finalAO);

        VkDescriptorImageInfo sceneDepthInfo{};
        sceneDepthInfo.sampler     = m_Sampler;
        sceneDepthInfo.imageView   = vkSceneDepth->GetImageView();
        sceneDepthInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

        VkDescriptorImageInfo linDepthSampledInfo{};
        linDepthSampledInfo.sampler     = m_Sampler;
        linDepthSampledInfo.imageView   = vkLinDepth->GetImageView();
        linDepthSampledInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

        VkDescriptorImageInfo linDepthStorageInfo{};
        linDepthStorageInfo.imageView   = vkLinDepth->GetImageView();
        linDepthStorageInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

        VkDescriptorImageInfo rawAOStorageInfo{};
        rawAOStorageInfo.imageView   = vkRawAO->GetImageView();
        rawAOStorageInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

        // Prefilter: [sceneDepth (sampler), linDepth (storage)].
        VkWriteDescriptorSet preWrites[2]{};
        preWrites[0] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
        preWrites[0].dstSet          = vr.prefilterSet;
        preWrites[0].dstBinding      = 0;
        preWrites[0].descriptorCount = 1;
        preWrites[0].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        preWrites[0].pImageInfo      = &sceneDepthInfo;
        preWrites[1] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
        preWrites[1].dstSet          = vr.prefilterSet;
        preWrites[1].dstBinding      = 1;
        preWrites[1].descriptorCount = 1;
        preWrites[1].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        preWrites[1].pImageInfo      = &linDepthStorageInfo;
        vkUpdateDescriptorSets(device, 2, preWrites, 0, nullptr);

        if (vr.mainSets[0] == VK_NULL_HANDLE) return;
        // Bindings 0 + 1 stable; binding 2 (UBO) rebound per render-stage in UpdateUBO.
        VkWriteDescriptorSet mainWrites[2 * MAX_FRAMES_IN_FLIGHT]{};
        for (u32 s = 0; s < MAX_FRAMES_IN_FLIGHT; ++s)
        {
            mainWrites[s * 2 + 0] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
            mainWrites[s * 2 + 0].dstSet          = vr.mainSets[s];
            mainWrites[s * 2 + 0].dstBinding      = 0;
            mainWrites[s * 2 + 0].descriptorCount = 1;
            mainWrites[s * 2 + 0].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            mainWrites[s * 2 + 0].pImageInfo      = &linDepthSampledInfo;
            mainWrites[s * 2 + 1] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
            mainWrites[s * 2 + 1].dstSet          = vr.mainSets[s];
            mainWrites[s * 2 + 1].dstBinding      = 1;
            mainWrites[s * 2 + 1].descriptorCount = 1;
            mainWrites[s * 2 + 1].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            mainWrites[s * 2 + 1].pImageInfo      = &rawAOStorageInfo;
        }
        vkUpdateDescriptorSets(device, 2 * MAX_FRAMES_IN_FLIGHT, mainWrites, 0, nullptr);

        if (vr.denoiseSet == VK_NULL_HANDLE) return;
        VkDescriptorImageInfo rawAOSampledInfo{};
        rawAOSampledInfo.sampler     = m_Sampler;
        rawAOSampledInfo.imageView   = vkRawAO->GetImageView();
        rawAOSampledInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        VkDescriptorImageInfo finalAOStorageInfo{};
        finalAOStorageInfo.imageView   = vkFinalAO->GetImageView();
        finalAOStorageInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

        VkWriteDescriptorSet denoiseWrites[3]{};
        denoiseWrites[0] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
        denoiseWrites[0].dstSet          = vr.denoiseSet;
        denoiseWrites[0].dstBinding      = 0;
        denoiseWrites[0].descriptorCount = 1;
        denoiseWrites[0].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        denoiseWrites[0].pImageInfo      = &rawAOSampledInfo;
        denoiseWrites[1] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
        denoiseWrites[1].dstSet          = vr.denoiseSet;
        denoiseWrites[1].dstBinding      = 1;
        denoiseWrites[1].descriptorCount = 1;
        denoiseWrites[1].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        denoiseWrites[1].pImageInfo      = &linDepthSampledInfo;
        denoiseWrites[2] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
        denoiseWrites[2].dstSet          = vr.denoiseSet;
        denoiseWrites[2].dstBinding      = 2;
        denoiseWrites[2].descriptorCount = 1;
        denoiseWrites[2].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        denoiseWrites[2].pImageInfo      = &finalAOStorageInfo;
        vkUpdateDescriptorSets(device, 3, denoiseWrites, 0, nullptr);
    }

    namespace {
        struct GTAOPrefilterPC {
            IVec2 halfResSize;     // 0
            Vec2  invFullRes;      // 8
            float nearZ;           // 16
            float farZ;            // 20
            float _pad0;           // 24
            float _pad1;           // 28
        };
        static_assert(sizeof(GTAOPrefilterPC) == 32, "GTAOPrefilterPC layout mismatch");

        struct GTAOMainPC {
            Vec2  projParams;   // 0  (P[0][0], |P[1][1]|)
            float nearZ;        // 8
            float farZ;         // 12
            u32   frameIndex;   // 16
            u32   _pad0;        // 20
            u32   _pad1;        // 24
            u32   _pad2;        // 28
        };
        static_assert(sizeof(GTAOMainPC) == 32, "GTAOMainPC layout mismatch");
    }

    RG::ResourceHandle GTAOSubsystem::AddPrefilterPass(RG::RenderGraph& rg, RG::ResourceHandle sceneDepth,
        const GtaoViewState& state, const CameraParams& camera, FrameDebugger* debugger)
    {
        LH_PROFILE_FUNCTION();
        struct Data { RG::ResourceHandle depth, linear; };
        RG::ResourceHandle output;
        rg.AddComputePass<Data>("GTAODepthPrefilter", RG::QueueFamily::AsyncCompute,
            [&](Data& data, RG::RenderPassBuilder& builder) {
                data.depth = builder.ReadStorageImage(sceneDepth);
                RG::TextureDesc desc;
                desc.name = "GTAOLinearDepth";
                desc.width = state.halfWidth; desc.height = state.halfHeight;
                desc.format = RG::TextureFormat::R32_Float;
                data.linear = rg.ImportResource(desc, (void*)state.linearBinding.image,
                    (void*)state.linearBinding.view, RG::ResourceState::Undefined);
                output = data.linear = builder.WriteStorageImage(data.linear);
            },
            [pipeline = m_PrefilterPipeline.get(), set = state.prefilterSet,
             halfW = state.halfWidth, halfH = state.halfHeight, fullW = state.width, fullH = state.height,
             nearZ = camera.nearZ, farZ = camera.farZ, debugger](Data&, RG::RenderPassContext& ctx) {
                if (debugger) debugger->BeginCapturePass(ctx.passIndex, "GTAODepthPrefilter", "GTAOLinearDepth", false,
                    { "gtao_depth_prefilter", 0, 0, VK_POLYGON_MODE_FILL, false, false, false, false });
                if (pipeline && set)
                {
                    pipeline->Bind(ctx.commandBuffer);
                    vkCmdBindDescriptorSets(ctx.commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                        pipeline->GetLayout(), 0, 1, &set, 0, nullptr);
                    GTAOPrefilterPC pc{};
                    pc.halfResSize = { (i32)halfW, (i32)halfH };
                    pc.invFullRes = { 1.0f / float(fullW), 1.0f / float(fullH) };
                    pc.nearZ = nearZ; pc.farZ = farZ;
                    vkCmdPushConstants(ctx.commandBuffer, pipeline->GetLayout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
                    const u32 groupX = (halfW + 7) / 8, groupY = (halfH + 7) / 8;
                    vkCmdDispatch(ctx.commandBuffer, groupX, groupY, 1);
                    if (debugger) debugger->CaptureComputeDispatch("GTAODepthPrefilter", "gtao_depth_prefilter", groupX, groupY, 1);
                }
                if (debugger) debugger->EndCapturePass();
            });
        return output;
    }

    RG::ResourceHandle GTAOSubsystem::AddMainPass(RG::RenderGraph& rg, RG::ResourceHandle linearDepth,
        const GtaoViewState& state, const CameraParams& camera, u64 renderFrameIndex, u32 shaderFrameIndex, FrameDebugger* debugger)
    {
        LH_PROFILE_FUNCTION();
        struct Data { RG::ResourceHandle linear, raw; };
        RG::ResourceHandle output;
        rg.AddComputePass<Data>("GTAOMain", RG::QueueFamily::AsyncCompute,
            [&](Data& data, RG::RenderPassBuilder& builder) {
                data.linear = builder.ReadStorageImage(linearDepth);
                RG::TextureDesc desc;
                desc.name = "GTAORawAO";
                desc.width = state.halfWidth; desc.height = state.halfHeight;
                desc.format = RG::TextureFormat::R8_Unorm;
                data.raw = rg.ImportResource(desc, (void*)state.rawBinding.image,
                    (void*)state.rawBinding.view, RG::ResourceState::Undefined);
                output = data.raw = builder.WriteStorageImage(data.raw);
            },
            [pipeline = m_MainPipeline.get(), set = state.mainSets[static_cast<u32>(renderFrameIndex) % MAX_FRAMES_IN_FLIGHT],
             halfW = state.halfWidth, halfH = state.halfHeight,
             projection = Vec2(camera.projection[0][0], std::abs(camera.projection[1][1])),
             nearZ = camera.nearZ, farZ = camera.farZ, shaderFrameIndex, debugger](Data&, RG::RenderPassContext& ctx) {
                if (debugger) debugger->BeginCapturePass(ctx.passIndex, "GTAOMain", "GTAORawAO", false,
                    { "gtao_main", 0, 0, VK_POLYGON_MODE_FILL, false, false, false, false });
                if (pipeline && set)
                {
                    pipeline->Bind(ctx.commandBuffer);
                    vkCmdBindDescriptorSets(ctx.commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                        pipeline->GetLayout(), 0, 1, &set, 0, nullptr);
                    GTAOMainPC pc{};
                    pc.projParams = projection; // Preserve Vulkan Y-flip correction.
                    pc.nearZ = nearZ; pc.farZ = farZ; pc.frameIndex = shaderFrameIndex;
                    vkCmdPushConstants(ctx.commandBuffer, pipeline->GetLayout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
                    const u32 groupX = (halfW + 7) / 8, groupY = (halfH + 7) / 8;
                    vkCmdDispatch(ctx.commandBuffer, groupX, groupY, 1);
                    if (debugger) debugger->CaptureComputeDispatch("GTAOMain", "gtao_main", groupX, groupY, 1);
                }
                if (debugger) debugger->EndCapturePass();
            });
        return output;
    }

    RG::ResourceHandle GTAOSubsystem::AddDenoisePass(RG::RenderGraph& rg, RG::ResourceHandle rawAO,
        RG::ResourceHandle linearDepth, const GtaoViewState& state, FrameDebugger* debugger)
    {
        LH_PROFILE_FUNCTION();
        struct Data { RG::ResourceHandle raw, linear, final; };
        RG::ResourceHandle output;
        rg.AddComputePass<Data>("GTAODenoise", RG::QueueFamily::AsyncCompute,
            [&](Data& data, RG::RenderPassBuilder& builder) {
                data.raw = builder.ReadStorageImage(rawAO);
                data.linear = builder.ReadStorageImage(linearDepth);
                RG::TextureDesc desc;
                desc.name = "GTAOFinal";
                desc.width = state.halfWidth; desc.height = state.halfHeight;
                desc.format = RG::TextureFormat::R8_Unorm;
                data.final = rg.ImportResource(desc, (void*)state.finalBinding.image,
                    (void*)state.finalBinding.view, RG::ResourceState::Undefined);
                output = data.final = builder.WriteStorageImage(data.final);
            },
            [pipeline = m_DenoisePipeline.get(), set = state.denoiseSet,
             halfW = state.halfWidth, halfH = state.halfHeight, debugger](Data&, RG::RenderPassContext& ctx) {
                if (debugger) debugger->BeginCapturePass(ctx.passIndex, "GTAODenoise", "GTAOFinal", false,
                    { "gtao_denoise", 0, 0, VK_POLYGON_MODE_FILL, false, false, false, false });
                if (pipeline && set)
                {
                    pipeline->Bind(ctx.commandBuffer);
                    vkCmdBindDescriptorSets(ctx.commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                        pipeline->GetLayout(), 0, 1, &set, 0, nullptr);
                    const u32 groupX = (halfW + 7) / 8, groupY = (halfH + 7) / 8;
                    vkCmdDispatch(ctx.commandBuffer, groupX, groupY, 1);
                    if (debugger) debugger->CaptureComputeDispatch("GTAODenoise", "gtao_denoise", groupX, groupY, 1);
                }
                if (debugger) debugger->EndCapturePass();
            });
        return output;
    }
}

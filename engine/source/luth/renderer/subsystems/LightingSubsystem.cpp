#include "luthpch.h"
#include "luth/renderer/subsystems/LightingSubsystem.h"
#include "luth/renderer/RenderPipeline.h"
#include "luth/renderer/Renderer.h"
#include "luth/scene/systems/RenderingSystem.h"
#include "luth/scene/systems/LightingSystem.h"
#include "luth/scene/systems/SystemRegistry.h"
#include "luth/renderer/material/MaterialSystem.h"
#include "luth/renderer/material/Material.h"
#include "luth/renderer/resources/BoneMatrixBuffer.h"
#include "luth/renderer/resources/Model.h"
#include "luth/renderer/lighting/IBLPrecompute.h"
#include "luth/renderer/shader/ShaderLibrary.h"
#include "luth/renderer/draw/DrawCommand.h"
#include "luth/renderer/backend/vulkan/VulkanContext.h"
#include "luth/renderer/backend/vulkan/VulkanLightBindings.h"
#include "luth/renderer/backend/vulkan/VulkanTexture.h"
#include "luth/renderer/backend/vulkan/VulkanBuffer.h"
#include "luth/core/FrameData.h"
#include "luth/core/RenderSnapshot.h"
#include "luth/jobs/JobSystem.h"
#include "luth/memory/GPUTaggedPageAllocator.h"

namespace Luth
{
    void LightingSubsystem::Init(RenderPipeline& pipeline, const fs::path& hdrPath)
    {
        LH_PROFILE_FUNCTION();
        m_Pipeline = &pipeline;
        VkDevice device = VulkanContext::Get().GetDevice();

        auto loadSpv = [](const char* relPath) -> std::vector<u32> {
            auto sh = ShaderLibrary::LoadEngine(relPath);
            return sh ? sh->GetSpirV() : std::vector<u32>{};
        };
        m_ShadowVertSpv        = loadSpv("shaders/shadowDepth_vert.slang");
        m_ShadowFragSpv        = loadSpv("shaders/shadowDepth.slang");
        m_ShadowSkinnedVertSpv = loadSpv("shaders/shadowDepth_skinned.slang");

        if (m_ShadowVertSpv.empty() || m_ShadowFragSpv.empty() || m_ShadowSkinnedVertSpv.empty())
        {
            LH_LOG(Renderer, error, "LightingSubsystem: shadow shader SPIR-V empty after asset load!");
            return;
        }

        CreateShadowResources(device);
        LoadIBL(hdrPath);

        // Forward+ cluster build pipeline. Layout has 2 SSBO bindings (AABB write, Grid write).
        // UAB matches GTAO main's pattern: per-render-stage descriptor rewrites need it to
        // dodge validation 03047 when the previous frame's cmd buffer is still pending.
        {
            VkDescriptorSetLayoutBinding bindings[2] = {};
            bindings[0].binding         = 0;
            bindings[0].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            bindings[0].descriptorCount = 1;
            bindings[0].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;
            bindings[1].binding         = 1;
            bindings[1].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            bindings[1].descriptorCount = 1;
            bindings[1].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;

            VkDescriptorBindingFlags bindingFlags[2] = {
                VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT,
                VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT,
            };
            VkDescriptorSetLayoutBindingFlagsCreateInfo bindingFlagsCI{
                VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO };
            bindingFlagsCI.bindingCount  = 2;
            bindingFlagsCI.pBindingFlags = bindingFlags;

            VkDescriptorSetLayoutCreateInfo layoutCI{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
            layoutCI.pNext        = &bindingFlagsCI;
            layoutCI.flags        = VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT;
            layoutCI.bindingCount = 2;
            layoutCI.pBindings    = bindings;
            vkCreateDescriptorSetLayout(device, &layoutCI, nullptr, &m_ClusterBuildSetLayout);

            // Push constant: invProjection + viewportSize + _pad + nearZ + farZ + uvec2 tiles = 96 B.
            VkPushConstantRange pcRange{ VK_SHADER_STAGE_COMPUTE_BIT, 0, 96 };

            m_ClusterBuildSpv = loadSpv("shaders/cluster_build.slang");
            if (m_ClusterBuildSpv.empty())
            {
                LH_LOG(Renderer, error, "LightingSubsystem: failed to load cluster_build.slang!");
                return;
            }
            m_ClusterBuildPipeline = std::make_unique<VKComputePipeline>(
                m_ClusterBuildSpv,
                std::vector<VkDescriptorSetLayout>{ m_ClusterBuildSetLayout },
                std::vector<VkPushConstantRange>{ pcRange });
        }

        // Forward+ light-to-cluster assignment pipeline. 5 SSBO bindings; UAB for the per-frame
        // descriptor rewrites that happen inside RecordView.
        {
            VkDescriptorSetLayoutBinding bindings[5] = {};
            for (u32 i = 0; i < 5; ++i)
            {
                bindings[i].binding         = i;
                bindings[i].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                bindings[i].descriptorCount = 1;
                bindings[i].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;
            }

            VkDescriptorBindingFlags bindingFlags[5] = {
                VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT,
                VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT,
                VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT,
                VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT,
                VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT,
            };
            VkDescriptorSetLayoutBindingFlagsCreateInfo bindingFlagsCI{
                VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO };
            bindingFlagsCI.bindingCount  = 5;
            bindingFlagsCI.pBindingFlags = bindingFlags;

            VkDescriptorSetLayoutCreateInfo layoutCI{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
            layoutCI.pNext        = &bindingFlagsCI;
            layoutCI.flags        = VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT;
            layoutCI.bindingCount = 5;
            layoutCI.pBindings    = bindings;
            vkCreateDescriptorSetLayout(device, &layoutCI, nullptr, &m_LightAssignSetLayout);

            // Push constant: mat4 view + u32 pointLightCount + u32 spotLightCount + u32 maxLightsPerCluster + u32 _pad = 80 B.
            VkPushConstantRange pcRange{ VK_SHADER_STAGE_COMPUTE_BIT, 0, 80 };

            m_LightAssignSpv = loadSpv("shaders/light_assign.slang");
            if (m_LightAssignSpv.empty())
            {
                LH_LOG(Renderer, error, "LightingSubsystem: failed to load light_assign.slang!");
                return;
            }
            m_LightAssignPipeline = std::make_unique<VKComputePipeline>(
                m_LightAssignSpv,
                std::vector<VkDescriptorSetLayout>{ m_LightAssignSetLayout },
                std::vector<VkPushConstantRange>{ pcRange });
        }

        // Cluster debug viz pipeline. Two descriptor sets: set 0 = depth sampler (per-view stable),
        // set 1 = m_LightSetLayout (per-view x per-frame, the existing lightDescSet; only b1 read).
        {
            VkSamplerCreateInfo sampCI{ VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO };
            sampCI.magFilter    = VK_FILTER_NEAREST;
            sampCI.minFilter    = VK_FILTER_NEAREST;
            sampCI.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
            sampCI.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
            sampCI.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
            sampCI.mipmapMode   = VK_SAMPLER_MIPMAP_MODE_NEAREST;
            vkCreateSampler(device, &sampCI, nullptr, &m_ClusterVizDepthSampler);

            VkDescriptorSetLayoutBinding sbinding{};
            sbinding.binding         = 0;
            sbinding.descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            sbinding.descriptorCount = 1;
            sbinding.stageFlags      = VK_SHADER_STAGE_FRAGMENT_BIT;

            VkDescriptorSetLayoutCreateInfo slayoutCI{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
            slayoutCI.bindingCount = 1;
            slayoutCI.pBindings    = &sbinding;
            vkCreateDescriptorSetLayout(device, &slayoutCI, nullptr, &m_ClusterVizDescSetLayout);

            m_FullscreenVertSpv = loadSpv("shaders/fullscreen.slang");
            m_ClusterVizFragSpv = loadSpv("shaders/cluster_viz.slang");
            if (!m_FullscreenVertSpv.empty() && !m_ClusterVizFragSpv.empty())
            {
                std::vector<VkDescriptorSetLayout> layouts = { m_ClusterVizDescSetLayout, m_LightSetLayout };
                VkPushConstantRange pcRange{ VK_SHADER_STAGE_FRAGMENT_BIT, 0, 16 };  // vec2 viewport + nearZ + farZ
                PipelineConfig cfg;
                cfg.colorFormats       = { VK_FORMAT_R8G8B8A8_UNORM };
                cfg.depthFormat        = VK_FORMAT_UNDEFINED;
                cfg.depthTest          = false;
                cfg.depthWrite         = false;
                cfg.blendEnabled       = true;
                cfg.cullMode           = VK_CULL_MODE_NONE;
                cfg.pushConstantRanges = { pcRange };
                m_ClusterVizPipeline = std::make_unique<VKPipeline>(
                    cfg, m_FullscreenVertSpv, m_ClusterVizFragSpv, layouts);
            }
        }
    }

    void LightingSubsystem::BuildPipelines(const std::vector<VkDescriptorSetLayout>& geoLayouts)
    {
        LH_PROFILE_FUNCTION();
        BuildShadowPipelines(geoLayouts);
        BuildSkyboxPipeline(geoLayouts);
    }

    void LightingSubsystem::Shutdown()
    {
        LH_PROFILE_FUNCTION();
        VkDevice device = VulkanContext::Get().GetDevice();

        m_ClusterVizStates.ReleaseAll([] { Renderer::WaitForGPU(); });
        m_SkyboxPipeline.reset();
        m_SkyboxVB.reset();
        m_ShadowSkinnedPipeline.reset();
        m_ShadowPipeline.reset();

        m_HybridSignalsEnabled = false;
        m_IrradianceMap.reset();
        m_PrefilteredMap.reset();
        m_BRDFLut.reset();
        if (m_IBLSampler) { vkDestroySampler(device, m_IBLSampler, nullptr); m_IBLSampler = VK_NULL_HANDLE; }

        if (m_ShadowSampler)        { vkDestroySampler(device, m_ShadowSampler, nullptr);        m_ShadowSampler        = VK_NULL_HANDLE; }
        if (m_SunShadowMaskSampler) { vkDestroySampler(device, m_SunShadowMaskSampler, nullptr); m_SunShadowMaskSampler = VK_NULL_HANDLE; }
        for (u32 i = 0; i < k_ShadowCascadeCount; ++i)
        {
            if (m_ShadowLayerViews[i]) vkDestroyImageView(device, m_ShadowLayerViews[i], nullptr);
            m_ShadowLayerViews[i] = VK_NULL_HANDLE;
        }
        m_ShadowMap.reset();

        if (m_LightSetLayout) { vkDestroyDescriptorSetLayout(device, m_LightSetLayout, nullptr); m_LightSetLayout = VK_NULL_HANDLE; }

        m_ClusterBuildPipeline.reset();
        if (m_ClusterBuildSetLayout)
        {
            vkDestroyDescriptorSetLayout(device, m_ClusterBuildSetLayout, nullptr);
            m_ClusterBuildSetLayout = VK_NULL_HANDLE;
        }

        m_LightAssignPipeline.reset();
        if (m_LightAssignSetLayout)
        {
            vkDestroyDescriptorSetLayout(device, m_LightAssignSetLayout, nullptr);
            m_LightAssignSetLayout = VK_NULL_HANDLE;
        }

        m_ClusterVizPipeline.reset();
        if (m_ClusterVizDescSetLayout)
        {
            vkDestroyDescriptorSetLayout(device, m_ClusterVizDescSetLayout, nullptr);
            m_ClusterVizDescSetLayout = VK_NULL_HANDLE;
        }
        if (m_ClusterVizDepthSampler)
        {
            vkDestroySampler(device, m_ClusterVizDepthSampler, nullptr);
            m_ClusterVizDepthSampler = VK_NULL_HANDLE;
        }
    }

    void LightingSubsystem::ReloadSkybox(const fs::path& hdrPath, const std::vector<VkDescriptorSetLayout>& geoLayouts)
    {
        LH_PROFILE_FUNCTION();
        VkDevice device = VulkanContext::Get().GetDevice();
        vkDeviceWaitIdle(device);

        if (m_IBLSampler) { vkDestroySampler(device, m_IBLSampler, nullptr); m_IBLSampler = VK_NULL_HANDLE; }
        LoadIBL(hdrPath);

        // Skybox pipeline depends on prefiltered mip count; rebuild.
        m_SkyboxPipeline.reset();
        BuildSkyboxPipeline(geoLayouts);

        LH_LOG(Renderer, info, "Skybox reloaded from '{}'", hdrPath.string());
    }

    bool LightingSubsystem::OnShaderReloaded(const std::string& name, const std::vector<u32>& spv,
                                             const std::vector<VkDescriptorSetLayout>& geoLayouts)
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

        if (name == "cluster_build.slang" && m_ClusterBuildSetLayout)
        {
            m_ClusterBuildSpv = spv;
            deferComp(m_ClusterBuildPipeline);
            VkPushConstantRange pc{ VK_SHADER_STAGE_COMPUTE_BIT, 0, 96 };
            m_ClusterBuildPipeline = std::make_unique<VKComputePipeline>(m_ClusterBuildSpv,
                std::vector<VkDescriptorSetLayout>{ m_ClusterBuildSetLayout },
                std::vector<VkPushConstantRange>{ pc });
            return true;
        }
        if (name == "light_assign.slang" && m_LightAssignSetLayout)
        {
            m_LightAssignSpv = spv;
            deferComp(m_LightAssignPipeline);
            VkPushConstantRange pc{ VK_SHADER_STAGE_COMPUTE_BIT, 0, 80 };
            m_LightAssignPipeline = std::make_unique<VKComputePipeline>(m_LightAssignSpv,
                std::vector<VkDescriptorSetLayout>{ m_LightAssignSetLayout },
                std::vector<VkPushConstantRange>{ pc });
            return true;
        }
        if ((name == "cluster_viz.slang" || name == "fullscreen.slang") && m_ClusterVizDescSetLayout)
        {
            if (name == "cluster_viz.slang") m_ClusterVizFragSpv = spv;
            else                            m_FullscreenVertSpv = spv;
            deferGfx(m_ClusterVizPipeline);
            if (!m_FullscreenVertSpv.empty() && !m_ClusterVizFragSpv.empty())
            {
                std::vector<VkDescriptorSetLayout> layouts = { m_ClusterVizDescSetLayout, m_LightSetLayout };
                VkPushConstantRange pcRange{ VK_SHADER_STAGE_FRAGMENT_BIT, 0, 16 };
                PipelineConfig cfg;
                cfg.colorFormats = { VK_FORMAT_R8G8B8A8_UNORM };
                cfg.depthFormat  = VK_FORMAT_UNDEFINED;
                cfg.depthTest    = false; cfg.depthWrite = false;
                cfg.blendEnabled = true;
                cfg.cullMode     = VK_CULL_MODE_NONE;
                cfg.pushConstantRanges = { pcRange };
                m_ClusterVizPipeline = std::make_unique<VKPipeline>(
                    cfg, m_FullscreenVertSpv, m_ClusterVizFragSpv, layouts);
            }
            return true;
        }

        if (name == "shadowDepth_vert.slang")           m_ShadowVertSpv        = spv;
        else if (name == "shadowDepth.slang")      m_ShadowFragSpv        = spv;
        else if (name == "shadowDepth_skinned.slang") m_ShadowSkinnedVertSpv = spv;
        else if (name == "skybox_vert.slang")           m_SkyboxVertSpv        = spv;
        else if (name == "skybox.slang")           m_SkyboxFragSpv        = spv;
        else return false;

        if (name == "skybox_vert.slang" || name == "skybox.slang")
        {
            deferGfx(m_SkyboxPipeline);
            BuildSkyboxPipeline(geoLayouts);
        }
        else
        {
            deferGfx(m_ShadowPipeline);
            deferGfx(m_ShadowSkinnedPipeline);
            BuildShadowPipelines(geoLayouts);
        }
        return true;
    }

    VkDescriptorSet LightingSubsystem::GetLightDescSet(u32 slot) const
    {
        // Set 3 lives on the active ViewResources. Replay paths with a null view return VK_NULL_HANDLE.
        auto* vr = m_Pipeline ? m_Pipeline->GetCurrentViewResources() : nullptr;
        if (!vr) return VK_NULL_HANDLE;
        return vr->lightDescSet[slot];
    }

    void LightingSubsystem::WriteShadowView(ViewResources& vr)
    {
        LH_PROFILE_FUNCTION();
        if (!m_ShadowMap || vr.lightDescSet[0] == VK_NULL_HANDLE) return;

        auto vkShadowTex = std::static_pointer_cast<VKTexture>(m_ShadowMap);
        VkDescriptorImageInfo shadowImgInfo{};
        shadowImgInfo.sampler     = m_ShadowSampler;
        shadowImgInfo.imageView   = vkShadowTex->GetImageView();
        shadowImgInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

        // Binding 4 (RT sun shadow mask): per-view. The mask image view comes from
        // The RT domain's view-local mask. pbr.frag reads it only when
        // rtShadowParams.x > 0.5 (RT mode); CSM-mode pixels take the cascade-PCF branch and
        // don't dynamically access binding 4. Layout is SHADER_READ_ONLY_OPTIMAL; the RG
        // transitions the image to this from the RT pass's GENERAL via the consumer's Read.
        VkDescriptorImageInfo maskImgInfo{};
        if (m_HybridSignalsEnabled && vr.rtShadow && vr.rtShadow->mask)
        {
            auto vkMask = std::static_pointer_cast<VKTexture>(vr.rtShadow->mask);
            maskImgInfo.sampler     = m_SunShadowMaskSampler;
            maskImgInfo.imageView   = vkMask->GetImageView();
            maskImgInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        }

        // Binding 5 (ReSTIR DI): per-view demodulated diffuse irradiance, post-denoise. Bound to
        // vr.diDenoiser->svgfDenoised (the denoiser output), not vr.restirDI: the denoiser owns this slot whenever
        // ReSTIR is on (it passes the raw DI through when denoising is toggled off), so the bind is
        // static and the A/B is denoise-vs-raw with no descriptor swap. Reused mask sampler (linear
        // clamp-to-edge). pbr.frag reads it only when restirParams.x > 0.5; the denoise pass leaves the
        // image in GENERAL, the GeometryPass Read transitions it to SHADER_READ_ONLY_OPTIMAL.
        VkDescriptorImageInfo diImgInfo{};
        if (m_HybridSignalsEnabled && vr.diDenoiser && vr.diDenoiser->svgfDenoised)
        {
            auto vkDI = std::static_pointer_cast<VKTexture>(vr.diDenoiser->svgfDenoised);
            diImgInfo.sampler     = m_SunShadowMaskSampler;
            diImgInfo.imageView   = vkDI->GetImageView();
            diImgInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        }

        // Binding 6 (ReSTIR GI): post-denoise GI irradiance. Bound to vr.giDenoiser->svgfDenoised (the GI
        // denoiser owns this slot, mirroring b5/DI): the bind is static and the A/B is denoise-vs-raw
        // with no descriptor swap (the denoiser passes the raw GI through when disabled). Same reused
        // mask sampler. pbr.frag adds it only when restirParams.y > 0.5; the GeometryPass Read
        // transitions it from the denoiser's GENERAL to SHADER_READ_ONLY_OPTIMAL.
        VkDescriptorImageInfo giImgInfo{};
        if (m_HybridSignalsEnabled && vr.giDenoiser && vr.giDenoiser->svgfDenoised)
        {
            auto vkGI = std::static_pointer_cast<VKTexture>(vr.giDenoiser->svgfDenoised);
            giImgInfo.sampler     = m_SunShadowMaskSampler;
            giImgInfo.imageView   = vkGI->GetImageView();
            giImgInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        }

        // Binding 7 (RT reflections): post-denoise specular radiance. Bound to vr.reflectionDenoiser->svgfDenoised
        // (the specular denoiser owns the slot, mirroring b5/b6). pbr.frag composites it into the split-sum
        // specular IBL when reflParams.x > 0.5; the GeometryPass Read transitions it to SHADER_READ_ONLY.
        VkDescriptorImageInfo reflImgInfo{};
        if (m_HybridSignalsEnabled && vr.reflectionDenoiser && vr.reflectionDenoiser->svgfDenoised)
        {
            auto vkRefl = std::static_pointer_cast<VKTexture>(vr.reflectionDenoiser->svgfDenoised);
            reflImgInfo.sampler     = m_SunShadowMaskSampler;
            reflImgInfo.imageView   = vkRefl->GetImageView();
            reflImgInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        }

        // Binding 8: post-denoise ReSTIR-DI specular. Bound to vr.diSpecDenoiser->svgfDenoised; pbr.frag
        // adds it under restirParams.z. GeometryPass's Read transitions it to SHADER_READ_ONLY.
        VkDescriptorImageInfo diSpecImgInfo{};
        if (m_HybridSignalsEnabled && vr.diSpecDenoiser && vr.diSpecDenoiser->svgfDenoised)
        {
            auto vkDiSpec = std::static_pointer_cast<VKTexture>(vr.diSpecDenoiser->svgfDenoised);
            diSpecImgInfo.sampler     = m_SunShadowMaskSampler;
            diSpecImgInfo.imageView   = vkDiSpec->GetImageView();
            diSpecImgInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        }

        VkDevice device = VulkanContext::Get().GetDevice();
        VkWriteDescriptorSet writes[MAX_FRAMES_IN_FLIGHT * 6] = {};
        u32 writeCount = 0;
        for (u32 s = 0; s < MAX_FRAMES_IN_FLIGHT; ++s)
        {
            writes[writeCount] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
            writes[writeCount].dstSet          = vr.lightDescSet[s];
            writes[writeCount].dstBinding      = 3;
            writes[writeCount].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            writes[writeCount].descriptorCount = 1;
            writes[writeCount].pImageInfo      = &shadowImgInfo;
            ++writeCount;

            if (maskImgInfo.imageView != VK_NULL_HANDLE)
            {
                writes[writeCount] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
                writes[writeCount].dstSet          = vr.lightDescSet[s];
                writes[writeCount].dstBinding      = 4;
                writes[writeCount].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                writes[writeCount].descriptorCount = 1;
                writes[writeCount].pImageInfo      = &maskImgInfo;
                ++writeCount;
            }

            if (diImgInfo.imageView != VK_NULL_HANDLE)
            {
                writes[writeCount] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
                writes[writeCount].dstSet          = vr.lightDescSet[s];
                writes[writeCount].dstBinding      = 5;
                writes[writeCount].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                writes[writeCount].descriptorCount = 1;
                writes[writeCount].pImageInfo      = &diImgInfo;
                ++writeCount;
            }

            if (giImgInfo.imageView != VK_NULL_HANDLE)
            {
                writes[writeCount] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
                writes[writeCount].dstSet          = vr.lightDescSet[s];
                writes[writeCount].dstBinding      = 6;
                writes[writeCount].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                writes[writeCount].descriptorCount = 1;
                writes[writeCount].pImageInfo      = &giImgInfo;
                ++writeCount;
            }

            if (reflImgInfo.imageView != VK_NULL_HANDLE)
            {
                writes[writeCount] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
                writes[writeCount].dstSet          = vr.lightDescSet[s];
                writes[writeCount].dstBinding      = 7;
                writes[writeCount].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                writes[writeCount].descriptorCount = 1;
                writes[writeCount].pImageInfo      = &reflImgInfo;
                ++writeCount;
            }

            if (diSpecImgInfo.imageView != VK_NULL_HANDLE)
            {
                writes[writeCount] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
                writes[writeCount].dstSet          = vr.lightDescSet[s];
                writes[writeCount].dstBinding      = 8;
                writes[writeCount].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                writes[writeCount].descriptorCount = 1;
                writes[writeCount].pImageInfo      = &diSpecImgInfo;
                ++writeCount;
            }
        }
        vkUpdateDescriptorSets(device, writeCount, writes, 0, nullptr);
    }

    // Allocates LightSSBO from the tagged heap, copies the header + point / spot / emissive-triangle
    // arrays + the power-weighted alias table (emissive sections appended after spots[]).
    // Returns the region; the BuildGraph caller threads it through WriteSet3PerView, and
    // Assignment receives this physical slice explicitly through native preparation.
    Memory::GPUSubRegion LightingSubsystem::UploadLightSSBO(const GatheredLights& lights)
    {
        LH_PROFILE_FUNCTION();
        Memory::GPUSubRegion region{};
        auto* jobCtx = JobSystem::GetCurrentJobContext();
        if (!jobCtx) return region;
        const u32 frameAbs = static_cast<u32>(Renderer::GetFrameData()->GetRenderFrameIndex());
        jobCtx->GpuCache.CurrentTag = frameAbs;

        auto& heap = Memory::GPUTaggedPageAllocator::Get();
        const u64 pointBytes = lights.points.size() * sizeof(PointLightData);
        const u64 spotBytes  = lights.spots.size()  * sizeof(SpotLightData);
        const u64 triBytes   = lights.tris.size()   * sizeof(TriangleLightData);
        const u64 aliasBytes = lights.alias.size()  * sizeof(LightAliasEntry);
        const u64 ssboSize   = sizeof(LightSSBOHeader) + pointBytes + spotBytes + triBytes + aliasBytes;
        region = heap.Allocate(jobCtx->GpuCache, ssboSize, 16);
        if (!region.buffer) return {};

        // Header at 0; points[]@48; spots[] after points; tris[] after spots; alias[] after tris. The
        // emissive sections are appended so g_Lights readers that stop at spots[] see an unchanged prefix.
        auto* header = static_cast<LightSSBOHeader*>(region.mappedPtr);
        header->dirLight        = lights.dirLight;
        header->pointLightCount = static_cast<u32>(lights.points.size());
        header->spotLightCount  = static_cast<u32>(lights.spots.size());
        // Publish triangles only with their alias table present; a count without the table would let DI
        // draw an out-of-range light.
        header->triLightCount   = lights.alias.empty() ? 0u : static_cast<u32>(lights.tris.size());
        header->_pad            = 0;
        auto* base = static_cast<u8*>(region.mappedPtr) + sizeof(LightSSBOHeader);
        if (!lights.points.empty()) std::memcpy(base, lights.points.data(), pointBytes);
        if (!lights.spots.empty())  std::memcpy(base + pointBytes, lights.spots.data(), spotBytes);
        if (!lights.tris.empty())   std::memcpy(base + pointBytes + spotBytes, lights.tris.data(), triBytes);
        if (!lights.alias.empty())  std::memcpy(base + pointBytes + spotBytes + triBytes, lights.alias.data(), aliasBytes);
        heap.FlushRegion(region);

        return region;
    }

    void LightingSubsystem::WriteSet3PerView(const Memory::GPUSubRegion& lightSSBORegion,
                                             const Memory::GPUSubRegion& clusterGridRegion,
                                             const Memory::GPUSubRegion& lightIndexRegion)
    {
        LH_PROFILE_FUNCTION();
        const u32 frameAbs = static_cast<u32>(Renderer::GetFrameData()->GetRenderFrameIndex());
        const u32 slot     = frameAbs % MAX_FRAMES_IN_FLIGHT;

        ViewResources* vr = m_Pipeline->GetCurrentViewResources();
        if (!vr || vr->lightDescSet[slot] == VK_NULL_HANDLE) return;
        if (!lightSSBORegion.buffer || !clusterGridRegion.buffer || !lightIndexRegion.buffer) return;

        VkDescriptorBufferInfo lightBi{ lightSSBORegion.buffer,   lightSSBORegion.offset,   lightSSBORegion.size   };
        VkDescriptorBufferInfo gridBi { clusterGridRegion.buffer, clusterGridRegion.offset, clusterGridRegion.size };
        VkDescriptorBufferInfo indexBi{ lightIndexRegion.buffer,  lightIndexRegion.offset,  lightIndexRegion.size  };

        VkWriteDescriptorSet writes[3] = {};
        for (u32 i = 0; i < 3; ++i)
        {
            writes[i] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
            writes[i].dstSet          = vr->lightDescSet[slot];
            writes[i].dstBinding      = i;
            writes[i].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[i].descriptorCount = 1;
        }
        writes[0].pBufferInfo = &lightBi;
        writes[1].pBufferInfo = &gridBi;
        writes[2].pBufferInfo = &indexBi;
        vkUpdateDescriptorSets(VulkanContext::Get().GetDevice(), 3, writes, 0, nullptr);
    }

    // ---- Internal: shadow map + Set 3 layout/pool/set ----
    void LightingSubsystem::CreateShadowResources(VkDevice device)
    {
        LH_PROFILE_FUNCTION();
        // Shadow map: k_ShadowResolution^2, D32_Float, k_ShadowCascadeCount-layer 2D array.
        m_ShadowMap = std::make_shared<VKTexture>(
            k_ShadowResolution, k_ShadowResolution, TextureFormat::D32_Float,
            k_ShadowCascadeCount, /*createFlags*/ 0u, /*mipLevels*/ 1u, /*extraUsage*/ 0u);

        auto shadowTexForViews = std::static_pointer_cast<VKTexture>(m_ShadowMap);
        for (u32 i = 0; i < k_ShadowCascadeCount; ++i)
            m_ShadowLayerViews[i] = shadowTexForViews->CreateLayerView(i);

        // VKTexture's auto-init for depth images transitions to DEPTH_STENCIL_READ_ONLY_OPTIMAL.
        // The Set 3 binding 3 descriptor writes declare SHADER_READ_ONLY_OPTIMAL: matched in CSM
        // mode because ShadowPass writes + GeometryPass Read transitions through. In RT shadow
        // mode, ShadowPass never runs and the cascade map sits in the initial layout forever,
        // mismatching the descriptor. Override to SHADER_READ_ONLY_OPTIMAL at init so both modes
        // agree. ShadowPass first-frame write does SHADER_READ_ONLY -> DSAO normally.
        VulkanContext::Get().ImmediateSubmit([&](VkCommandBuffer cmd) {
            VkImageMemoryBarrier2 b{ VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2 };
            b.srcStageMask        = VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT;
            b.srcAccessMask       = VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT;
            b.dstStageMask        = VulkanBarrierCapabilities::ForEnabledRtPackage(
                VulkanContext::Get().SupportsRayTracing()).SampledImageReadStages();
            b.dstAccessMask       = VK_ACCESS_2_SHADER_READ_BIT;
            b.oldLayout           = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;
            b.newLayout           = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            b.image               = shadowTexForViews->GetImage();
            b.subresourceRange    = { VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, k_ShadowCascadeCount };
            VkDependencyInfo dep{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
            dep.imageMemoryBarrierCount = 1;
            dep.pImageMemoryBarriers    = &b;
            vkCmdPipelineBarrier2(cmd, &dep);
        });

        // Shadow sampler (PCF compare: less).
        VkSamplerCreateInfo samplerInfo{ VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO };
        samplerInfo.magFilter     = VK_FILTER_LINEAR;
        samplerInfo.minFilter     = VK_FILTER_LINEAR;
        samplerInfo.addressModeU  = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
        samplerInfo.addressModeV  = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
        samplerInfo.addressModeW  = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
        samplerInfo.borderColor   = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE;
        samplerInfo.compareEnable = VK_TRUE;
        samplerInfo.compareOp     = VK_COMPARE_OP_LESS;
        samplerInfo.mipmapMode    = VK_SAMPLER_MIPMAP_MODE_NEAREST;
        vkCreateSampler(device, &samplerInfo, nullptr, &m_ShadowSampler);

        // Sun shadow mask sampler (Set 3 binding 4): linear clamp-to-edge, no compare. Matches the
        // pbr.frag::ComputeShadowRT sample: `texture(sunShadowMask, uv).r`. Clamp-to-edge means
        // off-screen UVs (cluster outside frustum) read the edge value (1.0 if the mask was written
        // with no-shadow at the borders, but in practice pbr.frag clamps uv to [0,1] via gl_FragCoord).
        VkSamplerCreateInfo maskSamplerInfo{ VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO };
        maskSamplerInfo.magFilter    = VK_FILTER_LINEAR;
        maskSamplerInfo.minFilter    = VK_FILTER_LINEAR;
        maskSamplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        maskSamplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        maskSamplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        maskSamplerInfo.mipmapMode   = VK_SAMPLER_MIPMAP_MODE_NEAREST;
        if (VulkanContext::Get().SupportsRayTracing())
            vkCreateSampler(device, &maskSamplerInfo, nullptr, &m_SunShadowMaskSampler);

        const bool rt = VulkanContext::Get().SupportsRayTracing();
        const VulkanLightBindings layout({rt, rt});
        m_HybridSignalsEnabled = layout.HasHybridSignals();
        VkDescriptorSetLayoutBindingFlagsCreateInfo bindingFlagsCI{
            VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO };
        bindingFlagsCI.bindingCount  = layout.count;
        bindingFlagsCI.pBindingFlags = layout.flags.data();

        VkDescriptorSetLayoutCreateInfo lightLayoutInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
        lightLayoutInfo.pNext        = &bindingFlagsCI;
        lightLayoutInfo.flags        = VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT;
        lightLayoutInfo.bindingCount = layout.count;
        lightLayoutInfo.pBindings    = layout.bindings.data();
        vkCreateDescriptorSetLayout(device, &lightLayoutInfo, nullptr, &m_LightSetLayout);

        // Descriptor sets themselves move to ViewResources (per-view x MAX_FRAMES_IN_FLIGHT slots).
        // AllocateViewResources allocates from vr.descPool and calls WriteShadowView to populate b3.
    }

    // ---- Internal: IBL precompute (irradiance + prefiltered + BRDF LUT + skybox VB/SPVs) ----
    void LightingSubsystem::LoadIBL(const fs::path& hdrPath)
    {
        LH_PROFILE_FUNCTION();
        IBLResult ibl = IBL::Precompute(hdrPath);
        m_IrradianceMap  = ibl.irradianceMap;
        m_PrefilteredMap = ibl.prefilteredMap;
        m_BRDFLut        = ibl.brdfLut;
        m_IBLSampler     = ibl.iblSampler;
        m_SkyboxVB       = ibl.skyboxVB;
        m_SkyboxVertSpv  = std::move(ibl.skyboxVertSpv);
        m_SkyboxFragSpv  = std::move(ibl.skyboxFragSpv);
        // Per-view set rewrite is the orchestrator's job (RenderPipeline iterates m_ViewResources).
    }

    // ---- Internal: build shadow + skybox pipelines ----
    void LightingSubsystem::BuildShadowPipelines(const std::vector<VkDescriptorSetLayout>& geoLayouts)
    {
        LH_PROFILE_FUNCTION();
        // 4-byte VERTEX push constant carries cascadeIndex.
        VkPushConstantRange shadowCascadePC{};
        shadowCascadePC.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
        shadowCascadePC.offset     = 0;
        shadowCascadePC.size       = sizeof(u32);

        // Position-only attribute with full PBR vertex stride (52 bytes) so the shadow
        // pipeline can reuse the same VB as PBR draws.
        BufferLayout posOnly = { { ShaderDataType::Float3, "a_Position" } };
        auto shadowBindingDescs = posOnly.GetBindingDescriptions();
        auto shadowAttribDescs  = posOnly.GetAttributeDescriptions();
        if (!shadowBindingDescs.empty())
            shadowBindingDescs[0].stride = sizeof(float) * (3 + 3 + 2 + 2 + 3);

        PipelineConfig shadowConfig;
        shadowConfig.colorFormats = {};
        shadowConfig.depthFormat = VK_FORMAT_D32_SFLOAT;
        shadowConfig.depthTest = true; shadowConfig.depthWrite = true;
        shadowConfig.blendEnabled = false;
        shadowConfig.cullMode = VK_CULL_MODE_FRONT_BIT;
        shadowConfig.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
        shadowConfig.bindingDescriptions = shadowBindingDescs;
        shadowConfig.attributeDescriptions = shadowAttribDescs;
        shadowConfig.pushConstantRanges = { shadowCascadePC };

        m_ShadowPipeline = std::make_unique<VKPipeline>(shadowConfig, m_ShadowVertSpv, m_ShadowFragSpv, geoLayouts);

        if (!m_ShadowSkinnedVertSpv.empty())
        {
            // Empty vertex input: deformable VS fetch the deformed buffer by gl_VertexIndex.
            PipelineConfig skinnedConfig = shadowConfig;
            skinnedConfig.bindingDescriptions.clear();
            skinnedConfig.attributeDescriptions.clear();

            m_ShadowSkinnedPipeline = std::make_unique<VKPipeline>(
                skinnedConfig, m_ShadowSkinnedVertSpv, m_ShadowFragSpv, geoLayouts);
        }
    }

    void LightingSubsystem::BuildSkyboxPipeline(const std::vector<VkDescriptorSetLayout>& geoLayouts)
    {
        LH_PROFILE_FUNCTION();
        if (m_SkyboxVertSpv.empty() || m_SkyboxFragSpv.empty()) return;

        BufferLayout skyboxLayout = { { ShaderDataType::Float3, "a_Position" } };

        PipelineConfig skyboxConfig;
        skyboxConfig.colorFormats   = { VK_FORMAT_R16G16B16A16_SFLOAT };
        skyboxConfig.depthFormat    = VK_FORMAT_D32_SFLOAT;
        skyboxConfig.depthTest      = true;
        skyboxConfig.depthWrite     = false;
        skyboxConfig.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;
        skyboxConfig.blendEnabled   = false;
        // Y-flipped projection reverses winding; cull back = show inside faces.
        skyboxConfig.cullMode  = VK_CULL_MODE_BACK_BIT;
        skyboxConfig.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
        skyboxConfig.bindingDescriptions   = skyboxLayout.GetBindingDescriptions();
        skyboxConfig.attributeDescriptions = skyboxLayout.GetAttributeDescriptions();

        m_SkyboxPipeline = std::make_unique<VKPipeline>(skyboxConfig, m_SkyboxVertSpv, m_SkyboxFragSpv, geoLayouts);
    }

    // ---- Render-graph passes ----
    CsmBindings LightingSubsystem::PrepareCsmBindings(const std::array<VkDescriptorSet, 6>& sets, bool captureDraws) const
    {
        CsmBindings result;
        result.rigid = m_ShadowPipeline ? m_ShadowPipeline->GetHandle() : VK_NULL_HANDLE;
        result.deformed = m_ShadowSkinnedPipeline ? m_ShadowSkinnedPipeline->GetHandle() : VK_NULL_HANDLE;
        result.rigidLayout = m_ShadowPipeline ? m_ShadowPipeline->GetLayout() : VK_NULL_HANDLE;
        result.deformedLayout = m_ShadowSkinnedPipeline ? m_ShadowSkinnedPipeline->GetLayout() : VK_NULL_HANDLE;
        result.sets = sets;
        result.texture = m_ShadowMap.get();
        result.image = m_ShadowMap ? static_cast<const VKTexture&>(*m_ShadowMap).GetImage() : VK_NULL_HANDLE;
        std::copy(std::begin(m_ShadowLayerViews), std::end(m_ShadowLayerViews), result.layers.begin());
        result.captureDraws = captureDraws;
        return result;
    }

    GraphTextureRef LightingSubsystem::ImportShadowTarget(RG::RenderGraph& graph, const CsmBindings& bindings, u32 cascadeIndex)
    {
        RG::TextureDesc desc;
        desc.name = "ShadowMap.C" + std::to_string(cascadeIndex);
        desc.width = desc.height = k_ShadowResolution;
        desc.format = RG::TextureFormat::D32_Float;
        return {graph.ImportResource(desc, (void*)bindings.image, (void*)bindings.layers[cascadeIndex],
            RG::ResourceState::Undefined, cascadeIndex, 1), {bindings.texture, 0, 1, cascadeIndex, 1}};
    }
    RG::ResourceHandle LightingSubsystem::AddShadowPass(RG::RenderGraph& rg, RG::ResourceHandle targetDepth,
        const VisibleDrawRange& visible, const CsmBindings& bindings, u32 cascadeIndex,
        const DrawList& draws, const RenderSnapshot& snapshot, FrameDebugger* debugger)
    {
        LH_PROFILE_FUNCTION();
        struct DrawPacket
        {
            std::shared_ptr<Mesh> mesh; // Retain native buffers through recording.
            VkBuffer vertex, index;
            VkDeviceSize indirectOffset;
            u32 entityIndex, indexCount, objectIndex;
            bool deformed, skinned;
            std::string meshName, entityName;
        };
        std::vector<DrawPacket> packets;
        const bool capturing = debugger && bindings.captureDraws;
        if (bindings.rigid)
        {
            if (!bindings.rigidLayout || std::any_of(bindings.sets.begin(), bindings.sets.end(),
                [](VkDescriptorSet set) { return set == VK_NULL_HANDLE; }))
                throw std::invalid_argument("ShadowPass: incomplete native bindings");
            const auto prepareBucket = [&](const auto& bucket) { for (const auto& dc : bucket)
            {
                if (!dc.model) continue;
                auto mesh = dc.model->GetMesh(dc.meshIndex);
                if (!mesh) continue;
                auto vb = std::static_pointer_cast<VKVertexBuffer>(mesh->GetVertexBuffer());
                auto ib = std::static_pointer_cast<VKIndexBuffer>(mesh->GetIndexBuffer());
                if (!vb || !ib || (dc.isDeformed && !bindings.deformed)) continue;
                if (dc.isDeformed && !bindings.deformedLayout)
                    throw std::invalid_argument("ShadowPass: missing deformed pipeline layout");
                if (dc.gpuObjectIndex >= visible.maxDrawCount)
                    throw std::invalid_argument("ShadowPass: draw outside cascade-visible range");
                const VkDeviceSize offset = visible.indirect.binding.offset
                    + (u64(visible.firstDraw) + dc.gpuObjectIndex) * sizeof(VkDrawIndexedIndirectCommand);
                std::string meshName, entityName;
                if (capturing)
                {
                    meshName = dc.model->GetName() + "[" + std::to_string(dc.meshIndex) + "]";
                    entityName = "Entity";
                    const auto entity = entt::to_entity(dc.entity);
                    if (entity < snapshot.tagsByEntity.size() && snapshot.tagsByEntity[entity])
                        entityName = snapshot.tagsByEntity[entity];
                }
                packets.push_back({mesh, vb->GetVulkanBuffer(), ib->GetVulkanBuffer(), offset,
                    dc.entityIndex, ib->GetCount(), dc.gpuObjectIndex, dc.isDeformed, dc.isSkinned,
                    std::move(meshName), std::move(entityName)});
            } };
            prepareBucket(draws.opaque);
            prepareBucket(draws.cutout);
        }
        const VkBuffer indirectBuffer = visible.indirect.binding.slice->buffer;
        struct ShadowPassData { RG::ResourceHandle depthTex; RG::BufferHandle indirectBuf; };
        const std::string passName = "ShadowPass.C" + std::to_string(cascadeIndex);
        const std::string resName = "ShadowMap.C" + std::to_string(cascadeIndex);
        RG::ResourceHandle output;
        auto metadata = RG::RenderPassMetadata::Graphics("shadowDepth", true, true, false, VK_CULL_MODE_FRONT_BIT, 0);
        metadata.indirectDraws = true; metadata.AddDraws(packets);
        rg.AddPass<ShadowPassData>(passName,
            [&, targetDepth](ShadowPassData& data, RG::RenderPassBuilder& builder) {
                builder.SetDebugMetadata(metadata);
                VkClearValue clear{};
                clear.depthStencil = {1.0f, 0};
                data.depthTex = builder.WriteDepth(targetDepth,
                    VK_ATTACHMENT_LOAD_OP_CLEAR, VK_ATTACHMENT_STORE_OP_STORE, clear);
                data.indirectBuf = builder.ReadIndirectBuffer(visible.indirect.handle);
                output = data.depthTex;
            },
            [bindings, packets = std::move(packets), indirectBuffer, cascadeIndex, passName, resName, debugger, capturing]
            (ShadowPassData&, RG::RenderPassContext& ctx) {
                const auto cmd = ctx.commandBuffer;
                if (debugger) debugger->BeginCapturePass(ctx.passIndex, passName, resName, true,
                    {"shadowDepth", 0, VK_CULL_MODE_FRONT_BIT, VK_POLYGON_MODE_FILL, false, true, true, false});
                if (bindings.rigid)
                {
                    const auto bind = [&](bool deformed) {
                        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                            deformed ? bindings.deformed : bindings.rigid);
                        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                            deformed ? bindings.deformedLayout : bindings.rigidLayout,
                            0, 6, bindings.sets.data(), 0, nullptr);
                        vkCmdPushConstants(cmd, deformed ? bindings.deformedLayout : bindings.rigidLayout,
                            VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(u32), &cascadeIndex);
                    };
                    bind(false);
                    VkViewport viewport{};
                    viewport.width = float(k_ShadowResolution); viewport.height = float(k_ShadowResolution); viewport.maxDepth = 1.0f;
                    vkCmdSetViewport(cmd, 0, 1, &viewport);
                    const VkRect2D scissor{{0, 0}, {k_ShadowResolution, k_ShadowResolution}};
                    vkCmdSetScissor(cmd, 0, 1, &scissor);
                    bool currentDeformed = false;
                    // Preserve opaque/cutout ordering and exclude transparent casters.
                    for (const auto& packet : packets)
                    {
                        if (packet.deformed != currentDeformed)
                        {
                            currentDeformed = packet.deformed;
                            bind(currentDeformed);
                        }
                        if (!packet.deformed)
                        {
                            const VkDeviceSize offset = 0;
                            vkCmdBindVertexBuffers(cmd, 0, 1, &packet.vertex, &offset);
                        }
                        vkCmdBindIndexBuffer(cmd, packet.index, 0, VK_INDEX_TYPE_UINT32);
                        vkCmdDrawIndexedIndirect(cmd, indirectBuffer, packet.indirectOffset, 1,
                            sizeof(VkDrawIndexedIndirectCommand));
                        if (capturing)
                            debugger->CaptureIndirectDraw(passName, packet.meshName, packet.entityName,
                                packet.entityIndex, packet.indexCount, packet.objectIndex, packet.indirectOffset,
                                {"shadowDepth", 0, static_cast<u32>(VK_CULL_MODE_FRONT_BIT),
                                    VK_POLYGON_MODE_FILL, packet.skinned, true, true, false});
                    }
                }
                else LH_LOG(Renderer, error, "ShadowPass pipeline is null!");
                if (debugger) debugger->EndCapturePass();
            });
        return output;
    }
    SkyBindings LightingSubsystem::PrepareSkyBindings(const std::array<VkDescriptorSet, 5>& sets) const
    {
        return {m_SkyboxPipeline ? m_SkyboxPipeline->GetHandle() : VK_NULL_HANDLE,
            m_SkyboxPipeline ? m_SkyboxPipeline->GetLayout() : VK_NULL_HANDLE,
            m_SkyboxVB ? m_SkyboxVB->GetVulkanBuffer() : VK_NULL_HANDLE, m_SkyboxVB, sets};
    }

    RG::ResourceHandle LightingSubsystem::AddSkyboxPass(RG::RenderGraph& graph,
        RG::ResourceHandle sceneColor, RG::ResourceHandle sceneDepth, u32 width, u32 height,
        const SkyBindings& bindings, FrameDebugger* debugger)
    {
        LH_PROFILE_FUNCTION();
        struct Data { RG::ResourceHandle color, depth; };
        RG::ResourceHandle output;
        graph.AddPass<Data>("SkyboxPass",
            [&](Data& data, RG::RenderPassBuilder& builder) {
                builder.SetDebugMetadata(RG::RenderPassMetadata::Graphics("skybox", true, false, false, VK_CULL_MODE_BACK_BIT, bindings.pipeline && bindings.vertex ? 1 : 0));
                data.color = builder.Write(sceneColor, VK_ATTACHMENT_LOAD_OP_LOAD, VK_ATTACHMENT_STORE_OP_STORE);
                // Preserve the existing attachment policy. Sky shader depth writes are disabled.
                data.depth = builder.WriteDepth(sceneDepth, VK_ATTACHMENT_LOAD_OP_LOAD, VK_ATTACHMENT_STORE_OP_DONT_CARE);
                output = data.color;
            },
            [bindings, width, height, debugger](Data&, RG::RenderPassContext& ctx) {
                if (debugger) debugger->BeginCapturePass(ctx.passIndex, "SkyboxPass", "SceneColor", false,
                    {"skybox", 0, VK_CULL_MODE_BACK_BIT, VK_POLYGON_MODE_FILL, false, true, false, false});
                if (bindings.pipeline && bindings.vertex)
                {
                    const auto cmd = ctx.commandBuffer;
                    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, bindings.pipeline);
                    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                        bindings.layout, 0, 5, bindings.sets.data(), 0, nullptr);
                    VkViewport viewport{};
                    viewport.width = float(width); viewport.height = float(height); viewport.maxDepth = 1.0f;
                    vkCmdSetViewport(cmd, 0, 1, &viewport);
                    const VkRect2D scissor{{0, 0}, {width, height}};
                    vkCmdSetScissor(cmd, 0, 1, &scissor);
                    const VkDeviceSize offset = 0;
                    vkCmdBindVertexBuffers(cmd, 0, 1, &bindings.vertex, &offset);
                    vkCmdDraw(cmd, 36, 1, 0, 0);
                    if (debugger)
                    {
                        ObjectPushConstants dummy{};
                        debugger->CaptureDrawCall("SkyboxPass", "SkyboxCube", "Skybox", 0, 0, dummy,
                            {"skybox", 0, VK_CULL_MODE_BACK_BIT, VK_POLYGON_MODE_FILL, false, true, false, false});
                    }
                }
                if (debugger) debugger->EndCapturePass();
            });
        return output;
    }
    // Forward+ cluster build. Async-compute; per-view tagged-heap regions for AABB + grid.
    // Returns BufferHandles so downstream LightAssignPass / GeometryPass read the same VkBuffer
    // without re-importing (see arch/rendering-pipeline.md re-import hazard).
    GraphBufferRef LightingSubsystem::ImportLightingBuffer(RG::RenderGraph& graph, const char* name,
        const Memory::GPUSubRegion& slice)
    {
        return {graph.ImportBuffer({name, slice.size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT},
            (void*)slice.buffer, RG::ResourceState::Undefined), {&slice, slice.offset, slice.size}};
    }

    ClusterBindings LightingSubsystem::PrepareClusterBindings(u64 renderFrameIndex, VkDescriptorSet buildSet,
        VkDescriptorSet assignSet, const CameraParams& camera, u32 width, u32 height,
        const Memory::GPUSubRegion& lights, u32 pointCount, u32 spotCount) const
    {
        ClusterBindings out;
        out.buildSet = buildSet; out.assignSet = assignSet; out.lights = lights;
        Mat4 projection = camera.projection;
        projection[1][1] *= -1.0f; // Match GlobalSubsystem's Vulkan Y flip.
        out.buildConstants.invProjection = Math::Inverse(projection);
        out.buildConstants.viewportSize = Vec2(float(width), float(height));
        out.buildConstants.nearZ = camera.nearZ; out.buildConstants.farZ = camera.farZ;
        out.assignConstants.view = camera.view;
        out.assignConstants.pointLightCount = pointCount;
        out.assignConstants.spotLightCount = spotCount;
        if (!m_ClusterBuildPipeline || !m_LightAssignPipeline || !buildSet || !assignSet || !lights.buffer)
            return out;
        auto* jobCtx = JobSystem::GetCurrentJobContext();
        if (!jobCtx) return out;
        jobCtx->GpuCache.CurrentTag = static_cast<u32>(renderFrameIndex);
        auto& heap = Memory::GPUTaggedPageAllocator::Get();
        out.aabb = heap.Allocate(jobCtx->GpuCache, u64(k_ClusterCount) * 32, 16);
        out.grid = heap.Allocate(jobCtx->GpuCache, u64(k_ClusterCount) * sizeof(GPUCluster), 16);
        out.indices = heap.Allocate(jobCtx->GpuCache, u64(k_ClusterCount) * k_MaxLightsPerCluster * sizeof(u32), 16);
        out.counter = heap.Allocate(jobCtx->GpuCache, 16, 16);
        if (!out.aabb.buffer || !out.grid.buffer || !out.indices.buffer || !out.counter.buffer || !out.counter.mappedPtr)
            return out;
        std::memset(out.counter.mappedPtr, 0, 16);
        heap.FlushRegion(out.counter);
        const auto info = [](const Memory::GPUSubRegion& region) {
            return VkDescriptorBufferInfo{region.buffer, region.offset, region.size};
        };
        const std::array buildInfos{info(out.aabb), info(out.grid)};
        const std::array assignInfos{info(lights), info(out.aabb), info(out.grid), info(out.indices), info(out.counter)};
        std::array<VkWriteDescriptorSet, 7> writes{};
        for (u32 i = 0; i < writes.size(); ++i)
        {
            writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[i].dstSet = i < 2 ? buildSet : assignSet;
            writes[i].dstBinding = i < 2 ? i : i - 2;
            writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[i].descriptorCount = 1;
            writes[i].pBufferInfo = i < 2 ? &buildInfos[i] : &assignInfos[i - 2];
        }
        vkUpdateDescriptorSets(VulkanContext::Get().GetDevice(), static_cast<u32>(writes.size()), writes.data(), 0, nullptr);
        out.build = m_ClusterBuildPipeline->GetHandle(); out.buildLayout = m_ClusterBuildPipeline->GetLayout();
        out.assign = m_LightAssignPipeline->GetHandle(); out.assignLayout = m_LightAssignPipeline->GetLayout();
        out.ready = true;
        return out;
    }

    std::array<RG::BufferHandle, 2> LightingSubsystem::AddClusterBuildPass(RG::RenderGraph& graph,
        RG::BufferHandle aabb, RG::BufferHandle grid, const ClusterBindings& bindings, FrameDebugger* debugger)
    {
        struct Data { RG::BufferHandle aabb, grid; };
        std::array<RG::BufferHandle, 2> output;
        graph.AddComputePass<Data>("ClusterBuild", RG::QueueFamily::AsyncCompute,
            [&](Data& data, RG::RenderPassBuilder& builder) {
                data.aabb = builder.WriteBuffer(aabb);
                data.grid = builder.WriteBuffer(grid);
                output = {data.aabb, data.grid};
            },
            [bindings, debugger](Data&, RG::RenderPassContext& ctx) {
                const auto cmd = ctx.commandBuffer;
                if (debugger) debugger->BeginCapturePass(ctx.passIndex, "ClusterBuild", "", false,
                    {"cluster_build", 0, 0, VK_POLYGON_MODE_FILL, false, false, false, false});
                vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, bindings.build);
                vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, bindings.buildLayout,
                    0, 1, &bindings.buildSet, 0, nullptr);
                vkCmdPushConstants(cmd, bindings.buildLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                    sizeof(ClusterBuildConstants), &bindings.buildConstants);
                const u32 x = (k_ClusterTilesX + 3) / 4, y = (k_ClusterTilesY + 3) / 4, z = (k_ClusterSlicesZ + 3) / 4;
                vkCmdDispatch(cmd, x, y, z);
                if (debugger)
                {
                    debugger->CaptureComputeDispatch("ClusterBuild", "cluster_build", x, y, z);
                    debugger->EndCapturePass();
                }
            });
        return output;
    }

    std::array<RG::BufferHandle, 2> LightingSubsystem::AddLightAssignPass(RG::RenderGraph& graph,
        RG::BufferHandle lights, RG::BufferHandle aabb, RG::BufferHandle grid,
        RG::BufferHandle indices, RG::BufferHandle counter, const ClusterBindings& bindings, FrameDebugger* debugger)
    {
        struct Data { RG::BufferHandle lights, aabb, grid, indices, counter; };
        std::array<RG::BufferHandle, 2> output;
        graph.AddComputePass<Data>("LightAssign", RG::QueueFamily::AsyncCompute,
            [&](Data& data, RG::RenderPassBuilder& builder) {
                data.lights = builder.ReadBuffer(lights);
                data.aabb = builder.ReadBuffer(aabb);
                data.grid = builder.WriteBuffer(grid);
                data.indices = builder.WriteBuffer(indices);
                data.counter = builder.WriteBuffer(counter);
                output = {data.grid, data.indices};
            },
            [bindings, debugger](Data&, RG::RenderPassContext& ctx) {
                const auto cmd = ctx.commandBuffer;
                if (debugger) debugger->BeginCapturePass(ctx.passIndex, "LightAssign", "", false,
                    {"light_assign", 0, 0, VK_POLYGON_MODE_FILL, false, false, false, false});
                vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, bindings.assign);
                vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, bindings.assignLayout,
                    0, 1, &bindings.assignSet, 0, nullptr);
                vkCmdPushConstants(cmd, bindings.assignLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                    sizeof(LightAssignConstants), &bindings.assignConstants);
                const u32 x = (k_ClusterCount + 63) / 64;
                vkCmdDispatch(cmd, x, 1, 1);
                if (debugger)
                {
                    debugger->CaptureComputeDispatch("LightAssign", "light_assign", x, 1, 1);
                    debugger->EndCapturePass();
                }
            });
        return output;
    }
}

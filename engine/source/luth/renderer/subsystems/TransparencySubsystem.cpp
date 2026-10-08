#include "luthpch.h"
#include "luth/renderer/subsystems/TransparencySubsystem.h"
#include "luth/renderer/features/FogViewState.h"

#include "luth/renderer/backend/vulkan/VulkanContext.h"
#include "luth/renderer/backend/vulkan/VulkanTexture.h"
#include "luth/renderer/shader/ShaderLibrary.h"
#include "luth/renderer/Renderer.h"
#include "luth/core/diagnostics/Log.h"

#include <algorithm>

namespace Luth
{
    namespace {
        BufferLayout MakePBRVertexLayout() {
            return BufferLayout{
                { ShaderDataType::Float3, "a_Position"  },
                { ShaderDataType::Float3, "a_Normal"    },
                { ShaderDataType::Float2, "a_TexCoord0" },
                { ShaderDataType::Float2, "a_TexCoord1" },
                { ShaderDataType::Float4, "a_Tangent"   },
                { ShaderDataType::Float4, "a_Color"     }
            };
        }
    }

    void TransparencySubsystem::Init()
    {
        LH_PROFILE_FUNCTION();
        VkDevice device = VulkanContext::Get().GetDevice();

        // Set 6: b0 fog atlas (sampler3D, parity-rewritten per frame -> UAB), b1 OIT heads storage
        // image + b2 OIT nodes SSBO (UAB + PARTIALLY_BOUND: unwritten until the PPLL passes land;
        // the sorted pipeline never statically uses them, which partially-bound makes legal).
        VkDescriptorSetLayoutBinding bindings[4]{};
        bindings[0].binding         = 0;
        bindings[0].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        bindings[0].descriptorCount = 1;
        bindings[0].stageFlags      = VK_SHADER_STAGE_FRAGMENT_BIT;
        bindings[1].binding         = 1;
        bindings[1].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        bindings[1].descriptorCount = 1;
        bindings[1].stageFlags      = VK_SHADER_STAGE_FRAGMENT_BIT;
        bindings[2].binding         = 2;
        bindings[2].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[2].descriptorCount = 1;
        bindings[2].stageFlags      = VK_SHADER_STAGE_FRAGMENT_BIT;
        // b3: screen-space refraction backdrop (pre-transparent scene copy). Partially-bound so a frame
        // before the backdrop is first written stays legal (glass-free draws never sample it).
        bindings[3].binding         = 3;
        bindings[3].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        bindings[3].descriptorCount = 1;
        bindings[3].stageFlags      = VK_SHADER_STAGE_FRAGMENT_BIT;

        VkDescriptorBindingFlags bindingFlags[4] = {
            VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT,
            VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT | VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT,
            VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT | VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT,
            VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT | VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT,
        };
        VkDescriptorSetLayoutBindingFlagsCreateInfo bindingFlagsCI{
            VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO };
        bindingFlagsCI.bindingCount  = 4;
        bindingFlagsCI.pBindingFlags = bindingFlags;

        VkDescriptorSetLayoutCreateInfo layoutCI{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
        layoutCI.pNext        = &bindingFlagsCI;
        layoutCI.flags        = VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT;
        layoutCI.bindingCount = 4;
        layoutCI.pBindings    = bindings;
        vkCreateDescriptorSetLayout(device, &layoutCI, nullptr, &m_TransparentSetLayout);

        // Linear clamp-to-edge sampler for the refraction backdrop tap (Set 6 b3).
        VkSamplerCreateInfo sci{ VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO };
        sci.magFilter    = VK_FILTER_LINEAR;
        sci.minFilter    = VK_FILTER_LINEAR;
        sci.mipmapMode   = VK_SAMPLER_MIPMAP_MODE_NEAREST;
        sci.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        sci.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        sci.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        sci.maxLod       = 0.0f;
        vkCreateSampler(device, &sci, nullptr, &m_BackdropSampler);

        // OIT resolve layout: b0 heads (storage image), b1 nodes (SSBO). Stable per-view; rewritten
        // only by WriteOitView on alloc/resize, so no UAB flags needed.
        VkDescriptorSetLayoutBinding resolveBindings[2]{};
        resolveBindings[0].binding         = 0;
        resolveBindings[0].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        resolveBindings[0].descriptorCount = 1;
        resolveBindings[0].stageFlags      = VK_SHADER_STAGE_FRAGMENT_BIT;
        resolveBindings[1].binding         = 1;
        resolveBindings[1].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        resolveBindings[1].descriptorCount = 1;
        resolveBindings[1].stageFlags      = VK_SHADER_STAGE_FRAGMENT_BIT;

        VkDescriptorSetLayoutCreateInfo resolveCI{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
        resolveCI.bindingCount = 2;
        resolveCI.pBindings    = resolveBindings;
        vkCreateDescriptorSetLayout(device, &resolveCI, nullptr, &m_ResolveSetLayout);
    }

    void TransparencySubsystem::BuildPipelines(const std::vector<VkDescriptorSetLayout>& geoLayouts)
    {
        LH_PROFILE_FUNCTION();
        if (auto sh = ShaderLibrary::LoadEngine("shaders/pbr_transparent.slang"))
            m_TransparentFragSpv = sh->GetSpirV();
        if (auto sh = ShaderLibrary::LoadEngine("shaders/pbr_oit_store.slang"))
            m_OitStoreFragSpv = sh->GetSpirV();
        if (auto sh = ShaderLibrary::LoadEngine("shaders/fullscreen.slang"))
            m_FullscreenVertSpv = sh->GetSpirV();
        if (auto sh = ShaderLibrary::LoadEngine("shaders/oit_resolve.slang"))
            m_ResolveFragSpv = sh->GetSpirV();
        if (m_TransparentFragSpv.empty() || m_OitStoreFragSpv.empty() ||
            m_FullscreenVertSpv.empty() || m_ResolveFragSpv.empty())
        {
            LH_LOG(Renderer, error, "TransparencySubsystem: failed to load transparent-tier shaders!");
            return;
        }

        std::vector<VkDescriptorSetLayout> layouts = geoLayouts;
        layouts.push_back(m_TransparentSetLayout);

        const VkPushConstantRange pcRange{ VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(TransparentPC) };

        // Sorted variants mirror GeometrySubsystem's Transparent/Fade arm: GeometryPass attachments
        // (sceneColor + entityID + depth), depth-test-no-write LESS_OR_EQUAL, standard alpha blend.
        // OIT store variants drop ALL color attachments (PPLL writes via Set 6 storage) and blending.
        // emptyInput: the skinned variants fetch the deformed buffer by gl_VertexIndex (no bound VB).
        auto makeFactory = [pcRange](BufferLayout layout, bool oitStore, bool emptyInput = false) {
            auto bindingDescs = emptyInput ? std::vector<VkVertexInputBindingDescription>{}
                                           : layout.GetBindingDescriptions();
            auto attribDescs  = emptyInput ? std::vector<VkVertexInputAttributeDescription>{}
                                           : layout.GetAttributeDescriptions();
            return [bindingDescs, attribDescs, pcRange, oitStore](Material::RenderMode, Material::CullMode cullMode,
                                                                  VkPolygonMode polygonMode) -> PipelineConfig
            {
                PipelineConfig config;
                if (!oitStore)
                    config.colorFormats      = { VK_FORMAT_R16G16B16A16_SFLOAT, VK_FORMAT_R32_UINT };
                config.depthFormat           = VK_FORMAT_D32_SFLOAT;
                config.frontFace             = VK_FRONT_FACE_COUNTER_CLOCKWISE;
                config.bindingDescriptions   = bindingDescs;
                config.attributeDescriptions = attribDescs;
                config.polygonMode           = polygonMode;
                config.depthCompareOp        = VK_COMPARE_OP_LESS_OR_EQUAL;
                config.depthTest             = true;
                config.depthWrite            = false;
                config.blendEnabled          = !oitStore;
                config.pushConstantRanges    = { pcRange };
                switch (cullMode)
                {
                    case Material::CullMode::Back:  config.cullMode = VK_CULL_MODE_BACK_BIT;  break;
                    case Material::CullMode::Front: config.cullMode = VK_CULL_MODE_FRONT_BIT; break;
                    case Material::CullMode::None:  config.cullMode = VK_CULL_MODE_NONE;      break;
                }
                return config;
            };
        };

        m_SortedPm.Init(layouts, makeFactory(MakePBRVertexLayout(), false));
        m_SortedSkinnedPm.Init(layouts, makeFactory(BufferLayout{}, false, true));
        m_OitPm.Init(layouts, makeFactory(MakePBRVertexLayout(), true));
        m_OitSkinnedPm.Init(layouts, makeFactory(BufferLayout{}, true, true));

        BuildResolvePipeline();
    }

    void TransparencySubsystem::BuildResolvePipeline()
    {
        LH_PROFILE_FUNCTION();
        // Under-composite blend: shader outputs (C, T); final = C + background * T.
        PipelineConfig cfg{};
        cfg.colorFormats        = { VK_FORMAT_R16G16B16A16_SFLOAT, VK_FORMAT_R32_UINT };
        cfg.depthFormat         = VK_FORMAT_UNDEFINED;
        cfg.depthTest           = false;
        cfg.depthWrite          = false;
        cfg.blendEnabled        = true;
        cfg.srcColorBlendFactor = VK_BLEND_FACTOR_ONE;
        cfg.dstColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
        cfg.cullMode            = VK_CULL_MODE_NONE;
        cfg.pushConstantRanges  = { { VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(u32) } };
        std::vector<VkDescriptorSetLayout> setLayouts = { m_ResolveSetLayout };
        m_ResolvePipeline = std::make_unique<VKPipeline>(cfg, m_FullscreenVertSpv, m_ResolveFragSpv, setLayouts);
    }

    void TransparencySubsystem::Shutdown()
    {
        LH_PROFILE_FUNCTION();
        m_ViewStates.ReleaseAll([] { Renderer::WaitForGPU(); });
        m_SortedPm.Shutdown();
        m_SortedSkinnedPm.Shutdown();
        m_OitPm.Shutdown();
        m_OitSkinnedPm.Shutdown();
        m_ResolvePipeline.reset();
        VkDevice device = VulkanContext::Get().GetDevice();
        if (m_TransparentSetLayout != VK_NULL_HANDLE)
        {
            vkDestroyDescriptorSetLayout(device, m_TransparentSetLayout, nullptr);
            m_TransparentSetLayout = VK_NULL_HANDLE;
        }
        if (m_ResolveSetLayout != VK_NULL_HANDLE)
        {
            vkDestroyDescriptorSetLayout(device, m_ResolveSetLayout, nullptr);
            m_ResolveSetLayout = VK_NULL_HANDLE;
        }
        if (m_BackdropSampler != VK_NULL_HANDLE)
        {
            vkDestroySampler(device, m_BackdropSampler, nullptr);
            m_BackdropSampler = VK_NULL_HANDLE;
        }
    }

    bool TransparencySubsystem::OnShaderReloaded(const std::string& name, const std::vector<u32>& spv)
    {
        LH_PROFILE_FUNCTION();
        auto invalidateSorted = [this]() {
            if (auto sh = ShaderLibrary::Get("pbr_transparent.slang"))
            {
                m_SortedPm.DeferredInvalidateShader(sh->Handle);
                m_SortedSkinnedPm.DeferredInvalidateShader(sh->Handle);
            }
        };
        auto invalidateOit = [this]() {
            if (auto sh = ShaderLibrary::Get("pbr_oit_store.slang"))
            {
                m_OitPm.DeferredInvalidateShader(sh->Handle);
                m_OitSkinnedPm.DeferredInvalidateShader(sh->Handle);
            }
        };

        if (name == "pbr_transparent.slang")
        {
            m_TransparentFragSpv = spv;
            invalidateSorted();
            return true;
        }
        if (name == "pbr_oit_store.slang")
        {
            m_OitStoreFragSpv = spv;
            invalidateOit();
            return true;
        }
        if (name == "oit_resolve.slang")
        {
            m_ResolveFragSpv = spv;
            // Defer-destroy the live pipeline (in-flight frames may still bind it), then rebuild.
            if (m_ResolvePipeline)
                VulkanContext::Get().PushDeletion([p = m_ResolvePipeline.release()]() { delete p; });
            BuildResolvePipeline();
            return true;
        }
        // Vert reloads are owned by GeometrySubsystem (cached spv there); our variants compiled
        // against the old spv must still drop. Return false so the geometry handler runs too.
        if (name == "pbr_vert.slang" || name == "pbr_skinned.slang")
        {
            invalidateSorted();
            invalidateOit();
        }
        return false;
    }

    void TransparencySubsystem::WritePerFrame(TransparencyViewState& vr, const std::shared_ptr<FogViewState>& fog, VkSampler fogSampler, u32 frameAbs)
    {
        LH_PROFILE_FUNCTION();
        if (m_TransparentSetLayout == VK_NULL_HANDLE) return;
        if (!fog || !fog->volInScatterHistA || !fog->volInScatterHistB) return;

        const u32  slot   = frameAbs % MAX_FRAMES_IN_FLIGHT;
        const bool parity = (frameAbs & 1u) != 0u;
        if (vr.transparentDescSet[slot] == VK_NULL_HANDLE) return;

        // Same parity rule as the volumetric composite's b1: sample this frame's resolved atlas.
        auto vkScat = std::static_pointer_cast<VKTexture>(
            parity ? fog->volInScatterHistA : fog->volInScatterHistB);

        VkDescriptorImageInfo scatInfo{};
        scatInfo.imageView   = vkScat->GetImageView();
        scatInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        scatInfo.sampler     = fogSampler;

        VkDescriptorImageInfo backdropInfo{};
        VkWriteDescriptorSet  writes[2]{};
        u32 count = 0;

        writes[count]                 = VkWriteDescriptorSet{ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
        writes[count].dstSet          = vr.transparentDescSet[slot];
        writes[count].dstBinding      = 0;
        writes[count].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        writes[count].descriptorCount = 1;
        writes[count].pImageInfo      = &scatInfo;
        ++count;

        // b3 refraction backdrop (the copy pass leaves it SHADER_READ_ONLY before the transparent pass).
        if (vr.refractionBackdrop && m_BackdropSampler != VK_NULL_HANDLE)
        {
            auto vkBackdrop = std::static_pointer_cast<VKTexture>(vr.refractionBackdrop);
            backdropInfo.imageView   = vkBackdrop->GetImageView();
            backdropInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            backdropInfo.sampler     = m_BackdropSampler;
            writes[count]                 = VkWriteDescriptorSet{ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
            writes[count].dstSet          = vr.transparentDescSet[slot];
            writes[count].dstBinding      = 3;
            writes[count].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            writes[count].descriptorCount = 1;
            writes[count].pImageInfo      = &backdropInfo;
            ++count;
        }
        vkUpdateDescriptorSets(VulkanContext::Get().GetDevice(), count, writes, 0, nullptr);
        vr.fogBindings[slot] = fog;
    }

    void TransparencySubsystem::WriteOitView(TransparencyViewState& vr)
    {
        LH_PROFILE_FUNCTION();
        if (m_TransparentSetLayout == VK_NULL_HANDLE) return;
        if (!vr.oitHeads || vr.oitNodes.buffer == VK_NULL_HANDLE) return;

        auto vkHeads = std::static_pointer_cast<VKTexture>(vr.oitHeads);

        VkDescriptorImageInfo headsInfo{};
        headsInfo.imageView   = vkHeads->GetImageView();
        headsInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

        // Bind through the producer's GPUSubRegion {buffer, offset, size}; RG BufferHandles are
        // barrier bookkeeping only. see arch/rendering-pipeline.md (hazard 3)
        VkDescriptorBufferInfo nodesInfo{};
        nodesInfo.buffer = vr.oitNodes.buffer;
        nodesInfo.offset = vr.oitNodes.offset;
        nodesInfo.range  = vr.oitNodes.size;

        std::vector<VkWriteDescriptorSet> writes;
        writes.reserve(MAX_FRAMES_IN_FLIGHT * 2 + 2);
        for (u32 slot = 0; slot < MAX_FRAMES_IN_FLIGHT; ++slot)
        {
            if (vr.transparentDescSet[slot] == VK_NULL_HANDLE) continue;
            VkWriteDescriptorSet w{ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
            w.dstSet          = vr.transparentDescSet[slot];
            w.dstBinding      = 1;
            w.descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            w.descriptorCount = 1;
            w.pImageInfo      = &headsInfo;
            writes.push_back(w);
            w.dstBinding      = 2;
            w.descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            w.pImageInfo      = nullptr;
            w.pBufferInfo     = &nodesInfo;
            writes.push_back(w);
        }
        if (vr.oitResolveDescSet != VK_NULL_HANDLE)
        {
            VkWriteDescriptorSet w{ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
            w.dstSet          = vr.oitResolveDescSet;
            w.dstBinding      = 0;
            w.descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            w.descriptorCount = 1;
            w.pImageInfo      = &headsInfo;
            writes.push_back(w);
            w.dstBinding      = 1;
            w.descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            w.pImageInfo      = nullptr;
            w.pBufferInfo     = &nodesInfo;
            writes.push_back(w);
        }
        if (!writes.empty())
            vkUpdateDescriptorSets(VulkanContext::Get().GetDevice(),
                static_cast<u32>(writes.size()), writes.data(), 0, nullptr);
    }

}

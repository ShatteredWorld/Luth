#include "luthpch.h"
#include <atomic>
#include "luth/renderer/subsystems/GeometrySubsystem.h"
#include "luth/renderer/draw/DrawList.h"
#include "luth/renderer/resources/Mesh.h"
#include "luth/renderer/subsystems/LightingSubsystem.h"
#include "luth/renderer/RenderPipeline.h"
#include "luth/renderer/Renderer.h"
#include "luth/renderer/FrameDebugger.h"
#include "luth/scene/systems/RenderingSystem.h"
#include "luth/renderer/material/Material.h"
#include "luth/renderer/material/MaterialGraphCodegen.h"
#include "luth/renderer/material/MaterialSystem.h"
#include "luth/renderer/resources/BoneMatrixBuffer.h"
#include "luth/renderer/resources/Buffer.h"
#include "luth/renderer/resources/Model.h"
#include "luth/renderer/draw/DrawCommand.h"
#include "luth/renderer/shader/ShaderLibrary.h"
#include "luth/renderer/backend/vulkan/VulkanContext.h"
#include "luth/renderer/backend/vulkan/VulkanTexture.h"
#include "luth/renderer/backend/vulkan/VulkanBuffer.h"
#include "luth/renderer/backend/vulkan/VulkanAccelerationStructure.h"
#include "luth/core/FrameData.h"
#include "luth/core/RenderSnapshot.h"
#include "luth/jobs/JobSystem.h"
#include "luth/memory/GPUTaggedPageAllocator.h"
#include "luth/assets/AssetManager.h"

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
        // Position-only attribute with full PBR vertex stride; depth-prepass and shadow
        // pipelines reuse the PBR vertex buffer but only consume a_Position.
        std::pair<std::vector<VkVertexInputBindingDescription>, std::vector<VkVertexInputAttributeDescription>>
        MakePositionOnlyWithFullStride() {
            BufferLayout layout = { { ShaderDataType::Float3, "a_Position" } };
            auto bindings = layout.GetBindingDescriptions();
            auto attribs  = layout.GetAttributeDescriptions();
            // Stride mirrors MakePBRVertexLayout: Position3 + Normal3 + TexCoord0_2 + TexCoord1_2 + Tangent4 + Color4.
            if (!bindings.empty()) bindings[0].stride = sizeof(float) * (3 + 3 + 2 + 2 + 4 + 4);
            return { std::move(bindings), std::move(attribs) };
        }
    }

    void GeometrySubsystem::Init(RenderPipeline& pipeline)
    {
        LH_PROFILE_FUNCTION();
        m_Pipeline = &pipeline;

        auto loadSpv = [](const char* relPath) -> std::vector<u32> {
            auto sh = ShaderLibrary::LoadEngine(relPath);
            return sh ? sh->GetSpirV() : std::vector<u32>{};
        };
        m_PBRVertSpv                 = loadSpv("shaders/pbr_vert.slang");
        m_PBRFragSpv                 = loadSpv("shaders/pbr.slang");
        m_PBRSkinnedVertSpv          = loadSpv("shaders/pbr_skinned.slang");
        m_DepthPrepassVertSpv        = loadSpv("shaders/depthPrepass.slang");
        m_DepthPrepassSkinnedVertSpv = loadSpv("shaders/depthPrepass_skinned.slang");
        m_SlimGBufferVertSpv         = loadSpv("shaders/slim_gbuffer_vert.slang");
        m_SlimGBufferSkinnedVertSpv  = loadSpv("shaders/slim_gbuffer_skinned.slang");
        m_SlimGBufferFragSpv         = loadSpv("shaders/slim_gbuffer.slang");
        m_WireframeOverlayFragSpv    = loadSpv("shaders/wireframe_overlay.slang");

        if (m_PBRVertSpv.empty() || m_PBRFragSpv.empty() || m_PBRSkinnedVertSpv.empty()
         || m_DepthPrepassVertSpv.empty() || m_DepthPrepassSkinnedVertSpv.empty()
         || m_SlimGBufferVertSpv.empty() || m_SlimGBufferSkinnedVertSpv.empty() || m_SlimGBufferFragSpv.empty())
        {
            LH_LOG(Renderer, error, "GeometrySubsystem: shader SPIR-V empty after asset load!");
            return;
        }

        InitObjectSSBO();
        InitCullPipeline();
    }

    const std::vector<u32>& GeometrySubsystem::ResolveFragSpv(const UUID& fragShaderUUID)
    {
        LH_PROFILE_FUNCTION();
        if (!fragShaderUUID.IsValid()) return m_PBRFragSpv;

        auto it = m_GraphFragSpv.find(fragShaderUUID);
        if (it != m_GraphFragSpv.end()) return it->second;

        // Generated graph shaders register in ShaderLibrary (keyed by name); match on the asset Handle.
        for (const auto& [name, sh] : ShaderLibrary::GetAll())
        {
            if (sh && sh->Handle == fragShaderUUID && !sh->GetSpirV().empty())
                return m_GraphFragSpv.emplace(fragShaderUUID, sh->GetSpirV()).first->second;
        }
        return m_PBRFragSpv;  // not yet registered/loaded: stock this frame, retried next
    }

    void GeometrySubsystem::InitObjectSSBO()
    {
        LH_PROFILE_FUNCTION();
        VkDevice device = VulkanContext::Get().GetDevice();

        // Set 5 layout: binding 0 = ObjectSSBO. BuildGPUObjectBuffer rewrites binding 0
        // each render stage against a per-frame slot; UAB no longer required.
        VkDescriptorSetLayoutBinding binding{};
        binding.binding         = 0;
        binding.descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        binding.descriptorCount = 1;
        binding.stageFlags      = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;

        VkDescriptorSetLayoutCreateInfo layoutInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
        layoutInfo.bindingCount = 1;
        layoutInfo.pBindings    = &binding;
        vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &m_ObjectSSBODescLayout);

        VkDescriptorPoolSize poolSize{};
        poolSize.type            = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        poolSize.descriptorCount = MAX_FRAMES_IN_FLIGHT;
        VkDescriptorPoolCreateInfo poolInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
        poolInfo.maxSets       = MAX_FRAMES_IN_FLIGHT;
        poolInfo.poolSizeCount = 1;
        poolInfo.pPoolSizes    = &poolSize;
        vkCreateDescriptorPool(device, &poolInfo, nullptr, &m_ObjectSSBODescPool);

        VkDescriptorSetLayout layouts[MAX_FRAMES_IN_FLIGHT];
        for (u32 i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) layouts[i] = m_ObjectSSBODescLayout;
        VkDescriptorSetAllocateInfo allocInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
        allocInfo.descriptorPool     = m_ObjectSSBODescPool;
        allocInfo.descriptorSetCount = MAX_FRAMES_IN_FLIGHT;
        allocInfo.pSetLayouts        = layouts;
        vkAllocateDescriptorSets(device, &allocInfo, m_ObjectSSBODescSet.data());
        for (u32 i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
        {
            char name[48]; std::snprintf(name, sizeof(name), "Geometry.ObjectSSBO.Slot%u", i);
            VulkanContext::SetDebugName(m_ObjectSSBODescSet[i], name);
        }
    }

    void GeometrySubsystem::InitCullPipeline()
    {
        LH_PROFILE_FUNCTION();
        VkDevice device = VulkanContext::Get().GetDevice();

        // Cull descriptor layout: binding 0 = ObjectSSBO (read), binding 1 = IndirectBuffer (write).
        VkDescriptorSetLayoutBinding bindings[2]{};
        bindings[0].binding         = 0;
        bindings[0].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[0].descriptorCount = 1;
        bindings[0].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;
        bindings[1].binding         = 1;
        bindings[1].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[1].descriptorCount = 1;
        bindings[1].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;

        VkDescriptorSetLayoutCreateInfo layoutInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
        layoutInfo.bindingCount = 2;
        layoutInfo.pBindings    = bindings;
        vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &m_CullDescLayout);

        // Per-frame slot of m_CullDescSet rewritten in BuildGPUObjectBuffer; cycling
        // makes the written slot disjoint from the slot the GPU is consuming.
        VkDescriptorPoolSize poolSize{};
        poolSize.type            = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        poolSize.descriptorCount = 2 * MAX_FRAMES_IN_FLIGHT;
        VkDescriptorPoolCreateInfo poolInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
        poolInfo.maxSets       = MAX_FRAMES_IN_FLIGHT;
        poolInfo.poolSizeCount = 1;
        poolInfo.pPoolSizes    = &poolSize;
        vkCreateDescriptorPool(device, &poolInfo, nullptr, &m_CullDescPool);

        VkDescriptorSetLayout layouts[MAX_FRAMES_IN_FLIGHT];
        for (u32 i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) layouts[i] = m_CullDescLayout;
        VkDescriptorSetAllocateInfo allocInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
        allocInfo.descriptorPool     = m_CullDescPool;
        allocInfo.descriptorSetCount = MAX_FRAMES_IN_FLIGHT;
        allocInfo.pSetLayouts        = layouts;
        vkAllocateDescriptorSets(device, &allocInfo, m_CullDescSet.data());
        for (u32 i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
        {
            char name[48]; std::snprintf(name, sizeof(name), "Geometry.Cull.Slot%u", i);
            VulkanContext::SetDebugName(m_CullDescSet[i], name);
        }

        // PC: 6 frustum planes (96B) + objectCount + destOffset = 104B.
        VkPushConstantRange pcRange{};
        pcRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        pcRange.offset     = 0;
        pcRange.size       = sizeof(Vec4) * 6 + sizeof(u32) * 2;

        auto cullShader = ShaderLibrary::LoadEngine("shaders/gpu_cull.slang");
        auto spv = cullShader ? cullShader->GetSpirV() : std::vector<u32>{};
        if (spv.empty())
        {
            LH_LOG(Renderer, error, "GeometrySubsystem: failed to load gpu_cull.slang!");
            return;
        }
        m_CullPipeline = std::make_unique<VKComputePipeline>(
            spv,
            std::vector<VkDescriptorSetLayout>{ m_CullDescLayout },
            std::vector<VkPushConstantRange>{ pcRange });
    }

    void GeometrySubsystem::BuildPipelines(const std::vector<VkDescriptorSetLayout>& geoLayouts)
    {
        LH_PROFILE_FUNCTION();
        BuildPBRPipelines(geoLayouts);
        BuildDepthPrepassPipelines(geoLayouts);
        BuildSlimGBufferPipelines(geoLayouts);
    }

    void GeometrySubsystem::BuildPBRPipelines(const std::vector<VkDescriptorSetLayout>& geoLayouts)
    {
        LH_PROFILE_FUNCTION();
        auto pbrLayout = MakePBRVertexLayout();
        auto bindingDescs = pbrLayout.GetBindingDescriptions();
        auto attribDescs  = pbrLayout.GetAttributeDescriptions();

        m_GeoPipelineManager.Init(geoLayouts,
            [bindingDescs, attribDescs](Material::RenderMode mode, Material::CullMode cullMode, VkPolygonMode polygonMode) -> PipelineConfig
            {
                PipelineConfig config;
                config.colorFormats = { VK_FORMAT_R16G16B16A16_SFLOAT, VK_FORMAT_R32_UINT };
                config.depthFormat  = VK_FORMAT_D32_SFLOAT;
                config.frontFace    = VK_FRONT_FACE_COUNTER_CLOCKWISE;
                config.bindingDescriptions   = bindingDescs;
                config.attributeDescriptions = attribDescs;
                config.polygonMode = polygonMode;
                // LESS_OR_EQUAL: opaques pass DepthPrepass values (LESS) exactly;
                // cutouts/transparents Z-test against the prepass depth.
                config.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;

                switch (mode)
                {
                    case Material::RenderMode::Opaque:
                    case Material::RenderMode::Cutout:
                        config.depthTest = true; config.depthWrite = true;
                        config.blendEnabled = false;
                        break;
                    case Material::RenderMode::Transparent:
                    case Material::RenderMode::Fade:
                        config.depthTest = true; config.depthWrite = false;
                        config.blendEnabled = true;
                        break;
                }
                switch (cullMode)
                {
                    case Material::CullMode::Back:  config.cullMode = VK_CULL_MODE_BACK_BIT;  break;
                    case Material::CullMode::Front: config.cullMode = VK_CULL_MODE_FRONT_BIT; break;
                    case Material::CullMode::None:  config.cullMode = VK_CULL_MODE_NONE;      break;
                }
                return config;
            });

        // Skinned variant: empty vertex input; the deformable VS fetch pos/normal/tangent/uv from the
        // deformed buffer by gl_VertexIndex, so no VB is bound. see arch/rendering-pipeline.md
        m_GeoSkinnedPipelineManager.Init(geoLayouts,
            [](Material::RenderMode mode, Material::CullMode cullMode, VkPolygonMode polygonMode) -> PipelineConfig
            {
                PipelineConfig config;
                config.colorFormats = { VK_FORMAT_R16G16B16A16_SFLOAT, VK_FORMAT_R32_UINT };
                config.depthFormat  = VK_FORMAT_D32_SFLOAT;
                config.frontFace    = VK_FRONT_FACE_COUNTER_CLOCKWISE;
                config.polygonMode    = polygonMode;
                config.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;

                switch (mode)
                {
                    case Material::RenderMode::Opaque:
                    case Material::RenderMode::Cutout:
                        config.depthTest = true; config.depthWrite = true;
                        config.blendEnabled = false;
                        break;
                    case Material::RenderMode::Transparent:
                    case Material::RenderMode::Fade:
                        config.depthTest = true; config.depthWrite = false;
                        config.blendEnabled = true;
                        break;
                }
                switch (cullMode)
                {
                    case Material::CullMode::Back:  config.cullMode = VK_CULL_MODE_BACK_BIT;  break;
                    case Material::CullMode::Front: config.cullMode = VK_CULL_MODE_FRONT_BIT; break;
                    case Material::CullMode::None:  config.cullMode = VK_CULL_MODE_NONE;      break;
                }
                return config;
            });

        // Shaded-wireframe overlay: flat line-polygon pipelines redrawn over the lit fill (LEQUAL depth,
        // no write). Static uses the pbr vertex layout; skinned uses the empty deformable input. Created
        // directly (not via a manager) for the custom line/depth config; a distinct frag keeps the fill's
        // pipeline cache untouched.
        if (!m_PBRVertSpv.empty() && !m_WireframeOverlayFragSpv.empty())
        {
            PipelineConfig cfg;
            cfg.colorFormats   = { VK_FORMAT_R16G16B16A16_SFLOAT, VK_FORMAT_R32_UINT };
            cfg.depthFormat    = VK_FORMAT_D32_SFLOAT;
            cfg.frontFace      = VK_FRONT_FACE_COUNTER_CLOCKWISE;
            cfg.polygonMode    = VK_POLYGON_MODE_LINE;
            cfg.cullMode       = VK_CULL_MODE_NONE;
            cfg.depthTest      = true;
            cfg.depthWrite     = false;
            cfg.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;
            cfg.blendEnabled   = false;
            cfg.bindingDescriptions   = bindingDescs;
            cfg.attributeDescriptions = attribDescs;
            m_WireframeOverlayPipeline = std::make_unique<VKPipeline>(
                cfg, m_PBRVertSpv, m_WireframeOverlayFragSpv, geoLayouts);

            if (!m_PBRSkinnedVertSpv.empty())
            {
                PipelineConfig scfg = cfg;
                scfg.bindingDescriptions.clear();
                scfg.attributeDescriptions.clear();
                m_WireframeOverlaySkinnedPipeline = std::make_unique<VKPipeline>(
                    scfg, m_PBRSkinnedVertSpv, m_WireframeOverlayFragSpv, geoLayouts);
            }
        }
    }

    void GeometrySubsystem::BuildDepthPrepassPipelines(const std::vector<VkDescriptorSetLayout>& geoLayouts)
    {
        LH_PROFILE_FUNCTION();
        auto [posOnlyBindings, posOnlyAttribs] = MakePositionOnlyWithFullStride();
        const auto& shadowFragSpv = m_Pipeline->GetLighting().GetShadowFragSpv();

        if (!m_DepthPrepassVertSpv.empty() && !shadowFragSpv.empty())
        {
            PipelineConfig cfg;
            cfg.colorFormats = {};
            cfg.depthFormat  = VK_FORMAT_D32_SFLOAT;
            cfg.depthTest    = true;
            cfg.depthWrite   = true;
            cfg.depthCompareOp = VK_COMPARE_OP_LESS;
            cfg.blendEnabled = false;
            cfg.cullMode     = VK_CULL_MODE_BACK_BIT;
            cfg.frontFace    = VK_FRONT_FACE_COUNTER_CLOCKWISE;
            cfg.bindingDescriptions   = posOnlyBindings;
            cfg.attributeDescriptions = posOnlyAttribs;

            m_DepthPrepassPipeline = std::make_unique<VKPipeline>(
                cfg, m_DepthPrepassVertSpv, shadowFragSpv, geoLayouts);
        }

        if (!m_DepthPrepassSkinnedVertSpv.empty() && !shadowFragSpv.empty())
        {
            // Empty vertex input; deformable VS fetch the deformed buffer by gl_VertexIndex.
            PipelineConfig cfg;
            cfg.colorFormats = {};
            cfg.depthFormat  = VK_FORMAT_D32_SFLOAT;
            cfg.depthTest    = true;
            cfg.depthWrite   = true;
            cfg.depthCompareOp = VK_COMPARE_OP_LESS;
            cfg.blendEnabled = false;
            cfg.cullMode     = VK_CULL_MODE_BACK_BIT;
            cfg.frontFace    = VK_FRONT_FACE_COUNTER_CLOCKWISE;

            m_DepthPrepassSkinnedPipeline = std::make_unique<VKPipeline>(
                cfg, m_DepthPrepassSkinnedVertSpv, shadowFragSpv, geoLayouts);
        }
    }

    void GeometrySubsystem::BuildSlimGBufferPipelines(const std::vector<VkDescriptorSetLayout>& geoLayouts)
    {
        LH_PROFILE_FUNCTION();
        // 4 color attachments mirror SlimGBufferOutput. depthFormat lets the pipeline test against
        // prepass depth via EQUAL; no depth writes (DepthPrepass owns the buffer for this frame).
        auto makeConfig = [&](auto bindings, auto attribs) {
            PipelineConfig cfg;
            cfg.colorFormats = {
                VK_FORMAT_R16G16_SFLOAT,  // SlimNormal     (octahedral)
                VK_FORMAT_R8_UNORM,       // SlimRoughness
                VK_FORMAT_R16G16_SFLOAT,  // SlimMotion     (NDC delta)
                VK_FORMAT_R16_UINT,       // SlimMaterialID
            };
            cfg.depthFormat   = VK_FORMAT_D32_SFLOAT;
            cfg.depthTest     = true;
            cfg.depthWrite    = false;                       // DepthPrepass already wrote this depth
            cfg.depthCompareOp= VK_COMPARE_OP_EQUAL;         // exact prepass match; no overdraw
            cfg.blendEnabled  = false;
            cfg.cullMode      = VK_CULL_MODE_BACK_BIT;       // CullMode::None / Wireframe deferred
            cfg.frontFace     = VK_FRONT_FACE_COUNTER_CLOCKWISE;
            cfg.polygonMode   = VK_POLYGON_MODE_FILL;
            cfg.bindingDescriptions   = bindings;
            cfg.attributeDescriptions = attribs;
            return cfg;
        };

        // Cutout variant: same shaders, but writes its own depth (LESS_OR_EQUAL) because the opaque-only
        // prepass omits cutout; the EQUAL opaque config would reject every cutout fragment (prepass cleared
        // those pixels to 1.0 or holds the surface behind). slim_gbuffer.slang alpha-tests the holes away.
        auto makeCutoutConfig = [&](auto bindings, auto attribs) {
            auto cfg = makeConfig(bindings, attribs);
            cfg.depthWrite     = true;
            cfg.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;
            cfg.cullMode       = VK_CULL_MODE_NONE;   // cutout foliage is two-sided; back faces must reach
            return cfg;                               // the slim G-buffer (slim_gbuffer.slang flips the normal),
        };                                            // else RT shadows/reflections read the geometry behind it

        if (!m_SlimGBufferVertSpv.empty() && !m_SlimGBufferFragSpv.empty())
        {
            auto pbrLayout = MakePBRVertexLayout();
            auto pbrBindings = pbrLayout.GetBindingDescriptions();
            auto pbrAttribs  = pbrLayout.GetAttributeDescriptions();
            m_SlimGBufferPipeline = std::make_unique<VKPipeline>(
                makeConfig(pbrBindings, pbrAttribs),
                m_SlimGBufferVertSpv, m_SlimGBufferFragSpv, geoLayouts);
            m_SlimGBufferCutoutPipeline = std::make_unique<VKPipeline>(
                makeCutoutConfig(pbrBindings, pbrAttribs),
                m_SlimGBufferVertSpv, m_SlimGBufferFragSpv, geoLayouts);
        }
        if (!m_SlimGBufferSkinnedVertSpv.empty() && !m_SlimGBufferFragSpv.empty())
        {
            // Empty vertex input; deformable VS fetch the deformed buffer by gl_VertexIndex.
            std::vector<VkVertexInputBindingDescription>   noBindings;
            std::vector<VkVertexInputAttributeDescription> noAttribs;
            m_SlimGBufferSkinnedPipeline = std::make_unique<VKPipeline>(
                makeConfig(noBindings, noAttribs),
                m_SlimGBufferSkinnedVertSpv, m_SlimGBufferFragSpv, geoLayouts);
            m_SlimGBufferCutoutSkinnedPipeline = std::make_unique<VKPipeline>(
                makeCutoutConfig(noBindings, noAttribs),
                m_SlimGBufferSkinnedVertSpv, m_SlimGBufferFragSpv, geoLayouts);
        }
    }

    void GeometrySubsystem::Shutdown()
    {
        LH_PROFILE_FUNCTION();
        VkDevice device = VulkanContext::Get().GetDevice();

        m_SlimGBufferCutoutSkinnedPipeline.reset();
        m_SlimGBufferCutoutPipeline.reset();
        m_SlimGBufferSkinnedPipeline.reset();
        m_SlimGBufferPipeline.reset();
        m_DepthPrepassSkinnedPipeline.reset();
        m_DepthPrepassPipeline.reset();
        // PipelineManagers tear down internally on destruction.

        m_CullPipeline.reset();
        if (m_CullDescPool)   { vkDestroyDescriptorPool(device, m_CullDescPool, nullptr); m_CullDescPool = VK_NULL_HANDLE; }
        if (m_CullDescLayout) { vkDestroyDescriptorSetLayout(device, m_CullDescLayout, nullptr); m_CullDescLayout = VK_NULL_HANDLE; }
        m_CullDescSet.fill(VK_NULL_HANDLE);

        if (m_ObjectSSBODescPool)   { vkDestroyDescriptorPool(device, m_ObjectSSBODescPool, nullptr); m_ObjectSSBODescPool = VK_NULL_HANDLE; }
        if (m_ObjectSSBODescLayout) { vkDestroyDescriptorSetLayout(device, m_ObjectSSBODescLayout, nullptr); m_ObjectSSBODescLayout = VK_NULL_HANDLE; }
        m_ObjectSSBODescSet.fill(VK_NULL_HANDLE);
    }

    bool GeometrySubsystem::OnShaderReloaded(const std::string& name, const std::vector<u32>& spv,
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

        if      (name == "pbr_vert.slang")                   m_PBRVertSpv                 = spv;
        else if (name == "pbr.slang")                  m_PBRFragSpv                 = spv;
        else if (name == "pbr_skinned.slang")           m_PBRSkinnedVertSpv          = spv;
        else if (name == "depthPrepass.slang")          m_DepthPrepassVertSpv        = spv;
        else if (name == "depthPrepass_skinned.slang")  m_DepthPrepassSkinnedVertSpv = spv;
        else if (name == "slim_gbuffer_vert.slang")          m_SlimGBufferVertSpv         = spv;
        else if (name == "slim_gbuffer.slang")          m_SlimGBufferFragSpv         = spv;
        else if (name == "slim_gbuffer_skinned.slang")  m_SlimGBufferSkinnedVertSpv  = spv;
        else if (name != "gpu_cull.slang") return false;

        if (name == "gpu_cull.slang" && m_CullDescLayout)
        {
            deferComp(m_CullPipeline);
            VkPushConstantRange pc{ VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(Vec4) * 6 + sizeof(u32) * 2 };
            m_CullPipeline = std::make_unique<VKComputePipeline>(spv,
                std::vector<VkDescriptorSetLayout>{ m_CullDescLayout },
                std::vector<VkPushConstantRange>{ pc });
        }
        else if (name == "depthPrepass.slang" || name == "depthPrepass_skinned.slang")
        {
            deferGfx(m_DepthPrepassPipeline);
            deferGfx(m_DepthPrepassSkinnedPipeline);
            BuildDepthPrepassPipelines(geoLayouts);
        }
        else if (name == "slim_gbuffer_vert.slang" || name == "slim_gbuffer.slang" || name == "slim_gbuffer_skinned.slang")
        {
            deferGfx(m_SlimGBufferPipeline);
            deferGfx(m_SlimGBufferSkinnedPipeline);
            deferGfx(m_SlimGBufferCutoutPipeline);
            deferGfx(m_SlimGBufferCutoutSkinnedPipeline);
            BuildSlimGBufferPipelines(geoLayouts);
        }
        else
        {
            // pbr.*: invalidate the pipeline manager cache, rebuild the manager.
            const bool isPBR = (name == "pbr_vert.slang" || name == "pbr.slang");
            if (isPBR) {
                UUID pbrKey = ShaderLibrary::Get("pbr_vert.slang")->Handle;
                m_GeoPipelineManager.DeferredInvalidateShader(pbrKey);
                m_GeoSkinnedPipelineManager.DeferredInvalidateShader(pbrKey);
            } else {
                m_GeoPipelineManager.DeferredClear();
                m_GeoSkinnedPipelineManager.DeferredClear();
            }
            BuildPBRPipelines(geoLayouts);
        }
        return true;
    }

    u32 GeometrySubsystem::EnsureMaterialRegistered(std::shared_ptr<Material> material)
    {
        LH_PROFILE_FUNCTION();
        // Lazily lower a graph material to its generated fragment shader on first encounter. This runs in
        // the game-stage snapshot capture (off the render-recording path); the once-guard makes the Slang
        // compile a one-time cost. An editor edit clears the guard entry to force a re-emit.
        if (material->HasGraph() && m_GraphCompiled.insert(material->Handle).second)
            MaterialGraphCodegen::GenerateAndCompile(*material);

        auto it = m_MaterialSlotMap.find(material->Handle);
        if (it != m_MaterialSlotMap.end()) return it->second;

        u32 slot = MaterialSystem::RegisterMaterial(material);
        m_MaterialSlotMap[material->Handle] = slot;
        return slot;
    }

    // Perf observability: meshes dropped last build because the GPU object cap was hit.
    static std::atomic<u32> s_DroppedObjects{ 0 };
    u32 GeometrySubsystem::GetDroppedObjectCount() { return s_DroppedObjects.load(std::memory_order_relaxed); }

    void GeometrySubsystem::BuildGPUObjectBuffer(const RenderSnapshot& snapshot)
    {
        LH_PROFILE_FUNCTION();
        // Allocate fresh regions from the GPU tagged heap. Tag = absolute render-frame index;
        // descriptor slot = same index modulo MAX_FRAMES_IN_FLIGHT (per-frame storage rotation).
        // FreeTag(N-2) reclaims regions once the GPU retires the consuming submission.
        auto* jobCtx = JobSystem::GetCurrentJobContext();
        if (!jobCtx) return;
        const u32 frameAbs = static_cast<u32>(Renderer::GetFrameData()->GetRenderFrameIndex());
        const u32 slot     = frameAbs % MAX_FRAMES_IN_FLIGHT;
        jobCtx->GpuCache.CurrentTag = frameAbs;

        auto& heap = Memory::GPUTaggedPageAllocator::Get();
        const u64 objBytes = static_cast<u64>(RenderPipeline::k_MaxGPUObjects) * sizeof(GPUObjectData);
        const u64 indBytes = static_cast<u64>(RenderPipeline::k_IndirectRegionCount)
                           * RenderPipeline::k_IndirectRegionStride
                           * sizeof(VkDrawIndexedIndirectCommand);

        m_ObjectRegion   = heap.Allocate(jobCtx->GpuCache, objBytes, 16);
        m_IndirectRegion = heap.Allocate(jobCtx->GpuCache, indBytes, 16);
        if (!m_ObjectRegion.buffer || !m_IndirectRegion.buffer) { m_GPUObjectCount = 0; return; }

        auto* objectData   = static_cast<GPUObjectData*>(m_ObjectRegion.mappedPtr);
        auto* indirectCmds = static_cast<VkDrawIndexedIndirectCommand*>(m_IndirectRegion.mappedPtr);
        u32   count        = 0;
        u32   dropped      = 0;   // meshes skipped due to the k_MaxGPUObjects cap (observability)

        // Rebuild entity lookup. Index 0 = null sentinel; valid entities start at 1
        // (the geometry pass writes (entityID + 1) so 0 means "background").
        m_EntityLookup.clear();
        m_EntityLookup.push_back(entt::null);
        m_EntityToSSBOIndex.clear();

        for (const MeshDrawSnapshot& meshSnap : snapshot.meshes)
        {
            if (count >= RenderPipeline::k_MaxGPUObjects) { ++dropped; continue; }

            auto model = AssetManager::GetAsset<Model>(meshSnap.modelUUID);
            if (!model) continue;
            const auto& meshesData = model->GetMeshesData();
            auto mesh = model->GetMesh(meshSnap.meshIndex);
            if (!mesh) continue;

            // Skinned raster fetches its deformed buffer by BDA; skip the draw until the skinned
            // BLAS (+ deformed buffer) is ready, else the deformable VS derefs a null address and
            // faults the device. Transient (one frame) for a still-loading model.
            auto blas = mesh->GetBlas();
            if ((meshSnap.isSkinned || meshSnap.isDeformable) && (!blas || !blas->IsDeformable() || blas->GetDeformedBdaCurr(frameAbs) == 0))
                continue;

            GPUObjectData& obj = objectData[count];
            obj.model = meshSnap.worldMatrix;

            // Resolve previous-frame model from the render-side cache. Newly-spawned entities
            // (cache miss) fall back to current model -> zero motion for one frame.
            entt::entity entity = static_cast<entt::entity>(meshSnap.entity);
            auto pmIt = m_PrevModelByEntity.find(entity);
            const bool firstFrame = (pmIt == m_PrevModelByEntity.end());
            obj.prevModel = firstFrame ? obj.model : pmIt->second;

            const auto& aabb   = meshesData[meshSnap.meshIndex].BindPoseAABB;
            obj.boundingSphere = Vec4(aabb.Center(), Math::Length(aabb.Extents()));

            u32 matSlot = 0;
            if (meshSnap.materialUUID.IsValid()) {
                auto it = m_MaterialSlotMap.find(meshSnap.materialUUID);
                if (it != m_MaterialSlotMap.end()) matSlot = it->second;
            }
            obj.materialIndex = matSlot;
            obj.shadeMode     = static_cast<u32>(m_Pipeline->GetSystem().GetShadeMode());
            obj.entityID      = (u32)m_EntityLookup.size();
            obj.boneOffset    = meshSnap.boneOffset;

            m_EntityLookup.push_back(entity);
            m_EntityToSSBOIndex[entity] = count;

            auto* ib = std::static_pointer_cast<VKIndexBuffer>(mesh->GetIndexBuffer()).get();
            obj.indexCount     = ib ? ib->GetCount() : 0;
            obj.firstIndex     = 0;
            obj.vertexOffset   = 0;
            // Address the dual-buffer SSBO's previous-bones region for skinned motion vectors.
            // Don't-care for non-skinned draws (their shaders never read bones[]).
            obj.prevBoneOffset = meshSnap.boneOffset + BoneMatrixBuffer::PREV_BLOCK_OFFSET;

            // Deformed-vertex buffer BDAs for the deformable raster path (skinned only). CURR holds
            // this frame's skin; PREV is last frame's, for motion vectors. Rigid meshes get 0; their
            // VS reads the bound vertex buffer instead. (Skinned-but-unready meshes were skipped above.)
            if (blas && blas->IsDeformable())
            {
                obj.deformedBdaCurr = blas->GetDeformedBdaCurr(frameAbs);
                // Seed prev=curr on the entity's first frame (prev region still zero-filled) so motion
                // is zero; matches the prevModel cache-miss fallback above.
                obj.deformedBdaPrev = firstFrame ? obj.deformedBdaCurr : blas->GetDeformedBdaPrev(frameAbs);
            }
            else
            {
                obj.deformedBdaCurr = 0;
                obj.deformedBdaPrev = 0;
            }

            VkDrawIndexedIndirectCommand baseCmd{};
            baseCmd.indexCount    = obj.indexCount;
            baseCmd.instanceCount = 1;
            baseCmd.firstIndex    = 0;
            baseCmd.vertexOffset  = 0;
            baseCmd.firstInstance = count;
            for (u32 r = 0; r < RenderPipeline::k_IndirectRegionCount; ++r)
                indirectCmds[r * RenderPipeline::k_IndirectRegionStride + count] = baseCmd;
            count++;
        }

        m_GPUObjectCount = count;

        s_DroppedObjects.store(dropped, std::memory_order_relaxed);
        if (dropped > 0)
        {
            static std::atomic<bool> warned{ false };
            bool expected = false;
            if (warned.compare_exchange_strong(expected, true))
                LH_LOG(Renderer, warn, "GPU object cap ({}) exceeded - up to {} draws dropped. Raise k_MaxGPUObjects or add culling/LOD.",
                             (u32)RenderPipeline::k_MaxGPUObjects, dropped);
        }

        // Atomic-replace the prev-model cache from this frame's snapshot. Reads the ungated
        // snapshot list so entities with transient asset-load issues keep their prev-frame
        // matrix instead of dropping out of the cache for one frame.
        std::unordered_map<entt::entity, Mat4> nextPrev;
        nextPrev.reserve(snapshot.meshes.size());
        for (const auto& m : snapshot.meshes)
            nextPrev.emplace(static_cast<entt::entity>(m.entity), m.worldMatrix);
        m_PrevModelByEntity = std::move(nextPrev);

        heap.FlushRegion(m_ObjectRegion);
        heap.FlushRegion(m_IndirectRegion);

        VkDevice device = VulkanContext::Get().GetDevice();
        {
            VkDescriptorBufferInfo bi{};
            bi.buffer = m_ObjectRegion.buffer;
            bi.offset = m_ObjectRegion.offset;
            bi.range  = m_ObjectRegion.size;

            VkWriteDescriptorSet write{ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
            write.dstSet          = m_ObjectSSBODescSet[slot];
            write.dstBinding      = 0;
            write.descriptorCount = 1;
            write.descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            write.pBufferInfo     = &bi;
            vkUpdateDescriptorSets(device, 1, &write, 0, nullptr);
        }
        {
            VkDescriptorBufferInfo objInfo{};
            objInfo.buffer = m_ObjectRegion.buffer;
            objInfo.offset = m_ObjectRegion.offset;
            objInfo.range  = m_ObjectRegion.size;

            VkDescriptorBufferInfo indInfo{};
            indInfo.buffer = m_IndirectRegion.buffer;
            indInfo.offset = m_IndirectRegion.offset;
            indInfo.range  = m_IndirectRegion.size;

            VkWriteDescriptorSet writes[2]{};
            writes[0].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[0].dstSet          = m_CullDescSet[slot];
            writes[0].dstBinding      = 0;
            writes[0].descriptorCount = 1;
            writes[0].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[0].pBufferInfo     = &objInfo;
            writes[1]                 = writes[0];
            writes[1].dstBinding      = 1;
            writes[1].pBufferInfo     = &indInfo;
            vkUpdateDescriptorSets(device, 2, writes, 0, nullptr);
        }
    }

    CullBindings GeometrySubsystem::PrepareCullBindings(u64 renderFrameIndex, u32 objectCount) const
    {
        const u32 slot = static_cast<u32>(renderFrameIndex) % MAX_FRAMES_IN_FLIGHT;
        return {m_CullPipeline ? m_CullPipeline->GetHandle() : VK_NULL_HANDLE,
            m_CullPipeline ? m_CullPipeline->GetLayout() : VK_NULL_HANDLE, m_CullDescSet[slot], objectCount};
    }

    RG::BufferHandle GeometrySubsystem::AddCullPass(RG::RenderGraph& rg,
                                         RG::BufferHandle objectBuffer, RG::BufferHandle indirectBuffer,
                                         const std::array<Vec4, 6>& frustumPlanes, u32 destOffset,
                                         const char* passName, const CullBindings& bindings, FrameDebugger* debugger)
    {
        LH_PROFILE_FUNCTION();
        // Initialized indirect commands remain a valid unculled fallback when native
        // shader initialization is incomplete; never advertise a no-op cull producer.
        if (!bindings.pipeline || bindings.objectCount == 0) return indirectBuffer;
        if (!bindings.layout || !bindings.descriptorSet)
            throw std::invalid_argument("Visibility: incomplete native cull bindings");

        struct CullPassData {
            RG::BufferHandle objectBuffer;
            RG::BufferHandle indirectBuffer;
        };
        struct CullPushConstants {
            Vec4 frustumPlanes[6]; // 96B
            u32  objectCount;      // 4B
            u32  destOffset;       // 4B; index offset into commands[] (per-view-cascade region)
        };

        std::string name = passName ? passName : "FrustumCull";
        RG::BufferHandle output = indirectBuffer;

        rg.AddComputePass<CullPassData>(name,
            [=, &output](CullPassData& data, RG::RenderPassBuilder& builder)
            {
                data.objectBuffer   = builder.ReadBuffer(objectBuffer);
                data.indirectBuffer = builder.WriteBuffer(indirectBuffer);
                output = data.indirectBuffer;
            },
            [bindings, frustumPlanes, destOffset, name, debugger](CullPassData&, RG::RenderPassContext& ctx)
            {
                VkCommandBuffer cmd = ctx.commandBuffer;
                if (debugger)
                    debugger->BeginCapturePass(ctx.passIndex, name, "", false,
                        { "gpu_cull", 0, 0, VK_POLYGON_MODE_FILL, false, false, false, false });

                vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, bindings.pipeline);
                vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                    bindings.layout, 0, 1, &bindings.descriptorSet, 0, nullptr);

                CullPushConstants pc{};
                for (int i = 0; i < 6; ++i) pc.frustumPlanes[i] = frustumPlanes[i];
                pc.objectCount = bindings.objectCount;
                pc.destOffset  = destOffset;
                vkCmdPushConstants(cmd, bindings.layout,
                    VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(CullPushConstants), &pc);

                u32 groupCountX = (bindings.objectCount + 255) / 256;
                vkCmdDispatch(cmd, groupCountX, 1, 1);

                if (debugger)
                {
                    debugger->CaptureComputeDispatch(name, "gpu_cull", groupCountX, 1, 1);
                    debugger->EndCapturePass();
                }
            });
        return output;
    }

    DepthPrepassBindings GeometrySubsystem::PrepareDepthPrepassBindings(const std::array<VkDescriptorSet, 6>& sets, bool captureDraws) const
    {
        return {m_DepthPrepassPipeline ? m_DepthPrepassPipeline->GetHandle() : VK_NULL_HANDLE,
            m_DepthPrepassSkinnedPipeline ? m_DepthPrepassSkinnedPipeline->GetHandle() : VK_NULL_HANDLE,
            m_DepthPrepassPipeline ? m_DepthPrepassPipeline->GetLayout() : VK_NULL_HANDLE,
            m_DepthPrepassSkinnedPipeline ? m_DepthPrepassSkinnedPipeline->GetLayout() : VK_NULL_HANDLE, sets, captureDraws};
    }

    GraphTextureRef GeometrySubsystem::ImportDepthTarget(RG::RenderGraph& graph, const Texture& texture)
    {
        const auto& native = static_cast<const VKTexture&>(texture);
        RG::TextureDesc desc;
        desc.name = "SceneDepth"; desc.width = texture.GetWidth(); desc.height = texture.GetHeight();
        desc.format = RG::TextureFormat::D32_Float;
        return {graph.ImportResource(desc, (void*)native.GetImage(), (void*)native.GetImageView(),
            RG::ResourceState::Undefined), {&texture}};
    }

    RG::ResourceHandle GeometrySubsystem::AddDepthPrepass(RG::RenderGraph& rg, RG::ResourceHandle targetDepth,
        const VisibleDrawRange& visible, u32 width, u32 height, const DepthPrepassBindings& bindings,
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
                throw std::invalid_argument("DepthPrepass: incomplete native bindings");
            for (const auto& dc : draws.opaque)
            {
                if (!dc.model) continue;
                auto mesh = dc.model->GetMesh(dc.meshIndex);
                if (!mesh) continue;
                auto vb = std::static_pointer_cast<VKVertexBuffer>(mesh->GetVertexBuffer());
                auto ib = std::static_pointer_cast<VKIndexBuffer>(mesh->GetIndexBuffer());
                if (!vb || !ib || (dc.isDeformed && !bindings.deformed)) continue;
                if (dc.isDeformed && !bindings.deformedLayout)
                    throw std::invalid_argument("DepthPrepass: missing deformed pipeline layout");
                if (dc.gpuObjectIndex >= visible.maxDrawCount)
                    throw std::invalid_argument("DepthPrepass: draw outside camera-visible range");
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
            }
        }
        const VkBuffer indirectBuffer = visible.indirect.binding.slice->buffer;
        struct DepthPrepassData { RG::ResourceHandle depthTex; RG::BufferHandle indirectBuf; };
        RG::ResourceHandle output;
        auto metadata = RG::RenderPassMetadata::Graphics("depthPrepass", true, true, false, VK_CULL_MODE_BACK_BIT, 0);
        metadata.indirectDraws = true; metadata.AddDraws(packets);
        rg.AddPass<DepthPrepassData>("DepthPrepass",
            [&, targetDepth](DepthPrepassData& data, RG::RenderPassBuilder& builder) {
                builder.SetDebugMetadata(metadata);
                VkClearValue clear{};
                clear.depthStencil = {1.0f, 0};
                data.depthTex = builder.WriteDepth(targetDepth,
                    VK_ATTACHMENT_LOAD_OP_CLEAR, VK_ATTACHMENT_STORE_OP_STORE, clear);
                data.indirectBuf = builder.ReadIndirectBuffer(visible.indirect.handle);
                output = data.depthTex;
            },
            [bindings, packets = std::move(packets), indirectBuffer, width, height, debugger, capturing]
            (DepthPrepassData&, RG::RenderPassContext& ctx) {
                const auto cmd = ctx.commandBuffer;
                if (debugger) debugger->BeginCapturePass(ctx.passIndex, "DepthPrepass", "SceneDepth", true,
                    {"depthPrepass", 0, VK_CULL_MODE_BACK_BIT, VK_POLYGON_MODE_FILL, false, true, true, false});
                if (bindings.rigid)
                {
                    const auto bind = [&](bool deformed) {
                        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                            deformed ? bindings.deformed : bindings.rigid);
                        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                            deformed ? bindings.deformedLayout : bindings.rigidLayout,
                            0, 6, bindings.sets.data(), 0, nullptr);
                    };
                    bind(false);
                    VkViewport viewport{};
                    viewport.width = float(width); viewport.height = float(height); viewport.maxDepth = 1.0f;
                    vkCmdSetViewport(cmd, 0, 1, &viewport);
                    const VkRect2D scissor{{0, 0}, {width, height}};
                    vkCmdSetScissor(cmd, 0, 1, &scissor);
                    bool currentDeformed = false;
                    // Opaque only; slim G-buffer writes cutout depth in the next contribution.
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
                            debugger->CaptureIndirectDraw("DepthPrepass", packet.meshName, packet.entityName,
                                packet.entityIndex, packet.indexCount, packet.objectIndex, packet.indirectOffset,
                                {"depthPrepass", 0, static_cast<u32>(VK_CULL_MODE_BACK_BIT),
                                    VK_POLYGON_MODE_FILL, packet.skinned, true, true, false});
                    }
                }
                else LH_LOG(Renderer, error, "DepthPrepass pipeline is null!");
                if (debugger) debugger->EndCapturePass();
            });
        return output;
    }
    SlimGBufferBindings GeometrySubsystem::PrepareSlimGBufferBindings(const std::array<VkDescriptorSet, 6>& sets, bool captureDraws) const
    {
        const auto prepare = [&](const auto& rigid, const auto& deformed) {
            return DepthPrepassBindings{rigid ? rigid->GetHandle() : VK_NULL_HANDLE,
                deformed ? deformed->GetHandle() : VK_NULL_HANDLE,
                rigid ? rigid->GetLayout() : VK_NULL_HANDLE,
                deformed ? deformed->GetLayout() : VK_NULL_HANDLE, sets, captureDraws};
        };
        return {prepare(m_SlimGBufferPipeline, m_SlimGBufferSkinnedPipeline),
            prepare(m_SlimGBufferCutoutPipeline, m_SlimGBufferCutoutSkinnedPipeline)};
    }

    GraphTextureRef GeometrySubsystem::ImportSlimTarget(RG::RenderGraph& graph, const Texture& texture,
        const char* name, RG::TextureFormat format)
    {
        const auto& native = static_cast<const VKTexture&>(texture);
        RG::TextureDesc desc;
        desc.name = name; desc.width = texture.GetWidth(); desc.height = texture.GetHeight(); desc.format = format;
        return {graph.ImportResource(desc, (void*)native.GetImage(), (void*)native.GetImageView(),
            RG::ResourceState::Undefined), {&texture}};
    }

    std::array<RG::ResourceHandle, 5> GeometrySubsystem::AddSlimGBufferPass(RG::RenderGraph& rg,
        const std::array<GraphTextureRef, 4>& targets, RG::ResourceHandle prepassDepth,
        const VisibleDrawRange& visible, u32 width, u32 height, const SlimGBufferBindings& bindings,
        const DrawList& draws, const RenderSnapshot& snapshot, FrameDebugger* debugger)
    {
        LH_PROFILE_FUNCTION();
        struct DrawPacket
        {
            std::shared_ptr<Mesh> mesh;
            VkBuffer vertex, index;
            VkDeviceSize indirectOffset;
            u32 entityIndex, indexCount, objectIndex;
            bool deformed, skinned;
            std::string meshName, entityName;
        };
        const auto prepareDraws = [&](const auto& bucket, const DepthPrepassBindings& variant) {
            std::vector<DrawPacket> packets;
            if (!variant.rigid) return packets;
            if (!variant.rigidLayout || std::any_of(variant.sets.begin(), variant.sets.end(),
                [](VkDescriptorSet set) { return set == VK_NULL_HANDLE; }))
                throw std::invalid_argument("SlimGBuffer: incomplete native bindings");
            for (const auto& dc : bucket)
            {
                if (!dc.model) continue;
                auto mesh = dc.model->GetMesh(dc.meshIndex);
                if (!mesh) continue;
                auto vb = std::static_pointer_cast<VKVertexBuffer>(mesh->GetVertexBuffer());
                auto ib = std::static_pointer_cast<VKIndexBuffer>(mesh->GetIndexBuffer());
                if (!vb || !ib || (dc.isDeformed && !variant.deformed)) continue;
                if (dc.isDeformed && !variant.deformedLayout)
                    throw std::invalid_argument("SlimGBuffer: missing deformed pipeline layout");
                if (dc.gpuObjectIndex >= visible.maxDrawCount)
                    throw std::invalid_argument("SlimGBuffer: draw outside camera-visible range");
                const VkDeviceSize offset = visible.indirect.binding.offset
                    + (u64(visible.firstDraw) + dc.gpuObjectIndex) * sizeof(VkDrawIndexedIndirectCommand);
                std::string meshName, entityName;
                if (debugger && variant.captureDraws)
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
            }
            return packets;
        };
        auto opaque = prepareDraws(draws.opaque, bindings.opaque);
        // Preserve the existing whole-pass rigid-PSO guard before attempting cutouts.
        auto cutout = bindings.opaque.rigid ? prepareDraws(draws.cutout, bindings.cutout) : std::vector<DrawPacket>{};
        const VkBuffer indirectBuffer = visible.indirect.binding.slice->buffer;
        struct Data { std::array<RG::ResourceHandle, 5> images; RG::BufferHandle indirect; };
        std::array<RG::ResourceHandle, 5> output;
        auto metadata = RG::RenderPassMetadata::Graphics("slim_gbuffer", true, false, false, VK_CULL_MODE_BACK_BIT, 0);
        metadata.indirectDraws = true; metadata.pipelineStateMixed = !cutout.empty();
        metadata.AddDraws(opaque); metadata.AddDraws(cutout);
        rg.AddPass<Data>("SlimGBufferPass",
            [&](Data& data, RG::RenderPassBuilder& builder) {
                builder.SetDebugMetadata(metadata);
                std::array<VkClearValue, 4> clears{};
                clears[0].color.float32[0] = 0.5f; clears[0].color.float32[1] = 0.5f;
                clears[1].color.float32[0] = 1.0f;
                for (u32 i = 0; i < 4; ++i)
                    data.images[i] = builder.Write(targets[i].handle,
                        VK_ATTACHMENT_LOAD_OP_CLEAR, VK_ATTACHMENT_STORE_OP_STORE, clears[i]);
                // Opaque EQUAL preserves depth; alpha-tested cutouts write their surface.
                data.images[4] = builder.WriteDepth(prepassDepth, VK_ATTACHMENT_LOAD_OP_LOAD, VK_ATTACHMENT_STORE_OP_STORE, {});
                data.indirect = builder.ReadIndirectBuffer(visible.indirect.handle);
                output = data.images;
            },
            [bindings, opaque = std::move(opaque), cutout = std::move(cutout), indirectBuffer, width, height, debugger]
            (Data&, RG::RenderPassContext& ctx) {
                const auto cmd = ctx.commandBuffer;
                if (debugger) debugger->BeginCapturePass(ctx.passIndex, "SlimGBufferPass", "SlimNormal", false,
                    {"slim_gbuffer", 0, VK_CULL_MODE_BACK_BIT, VK_POLYGON_MODE_FILL, false, true, false, false});
                if (bindings.opaque.rigid)
                {
                    VkViewport viewport{};
                    viewport.width = float(width); viewport.height = float(height); viewport.maxDepth = 1.0f;
                    vkCmdSetViewport(cmd, 0, 1, &viewport);
                    const VkRect2D scissor{{0, 0}, {width, height}};
                    vkCmdSetScissor(cmd, 0, 1, &scissor);
                    const auto record = [&](const auto& packets, const DepthPrepassBindings& variant, bool depthWrite) {
                        const auto bind = [&](bool deformed) {
                            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                deformed ? variant.deformed : variant.rigid);
                            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                deformed ? variant.deformedLayout : variant.rigidLayout, 0, 6, variant.sets.data(), 0, nullptr);
                        };
                        bind(false);
                        bool currentDeformed = false;
                        for (const auto& packet : packets)
                        {
                            if (packet.deformed != currentDeformed) { currentDeformed = packet.deformed; bind(currentDeformed); }
                            if (!packet.deformed)
                            {
                                const VkDeviceSize offset = 0;
                                vkCmdBindVertexBuffers(cmd, 0, 1, &packet.vertex, &offset);
                            }
                            vkCmdBindIndexBuffer(cmd, packet.index, 0, VK_INDEX_TYPE_UINT32);
                            vkCmdDrawIndexedIndirect(cmd, indirectBuffer, packet.indirectOffset, 1, sizeof(VkDrawIndexedIndirectCommand));
                            if (debugger && variant.captureDraws)
                                debugger->CaptureIndirectDraw("SlimGBufferPass", packet.meshName, packet.entityName,
                                    packet.entityIndex, packet.indexCount, packet.objectIndex, packet.indirectOffset,
                                    {"slim_gbuffer", 0, static_cast<u32>(VK_CULL_MODE_BACK_BIT),
                                        VK_POLYGON_MODE_FILL, packet.skinned, true, depthWrite, false});
                        }
                    };
                    record(opaque, bindings.opaque, false);
                    if (!cutout.empty()) record(cutout, bindings.cutout, true);
                }
                else LH_LOG(Renderer, error, "SlimGBuffer pipeline is null!");
                if (debugger) debugger->EndCapturePass();
            });
        return output;
    }
    VkDeviceSize GeometrySubsystem::ForwardDrawOffset(const VisibleDrawRange& visible, u32 objectIndex)
    {
        if (!visible.indirect.handle.IsValid() || !visible.indirect.binding.slice ||
            !visible.indirect.binding.slice->buffer ||
            visible.indirect.binding.offset != visible.indirect.binding.slice->offset ||
            visible.indirect.binding.size > visible.indirect.binding.slice->size ||
            objectIndex >= visible.maxDrawCount ||
            (u64(visible.firstDraw) + visible.maxDrawCount) * sizeof(VkDrawIndexedIndirectCommand) > visible.indirect.binding.size)
            throw std::invalid_argument("ForwardOpaque: draw outside camera-visible slice");
        return visible.indirect.binding.offset + (u64(visible.firstDraw) + objectIndex) * sizeof(VkDrawIndexedIndirectCommand);
    }

    GraphTextureRef GeometrySubsystem::ImportForwardTarget(RG::RenderGraph& graph, const Texture& texture,
        const char* name, RG::TextureFormat format, RG::ResourceState initialState)
    {
        const auto& native = static_cast<const VKTexture&>(texture);
        RG::TextureDesc desc;
        desc.name = name; desc.width = texture.GetWidth(); desc.height = texture.GetHeight(); desc.format = format;
        return {graph.ImportResource(desc, (void*)native.GetImage(), (void*)native.GetImageView(), initialState), {&texture}};
    }

    ForwardOpaqueBindings GeometrySubsystem::PrepareForwardOpaqueBindings(const std::array<VkDescriptorSet, 6>& sets,
        bool wireframe, bool shadedWireframe, bool captureDraws, const VisibleDrawRange& visible,
        const DrawList& draws, const RenderSnapshot& snapshot)
    {
        LH_PROFILE_FUNCTION();
        ForwardOpaqueBindings out;
        out.sets = sets; out.captureDraws = captureDraws;
        out.polygon = wireframe ? VK_POLYGON_MODE_LINE : VK_POLYGON_MODE_FILL;
        if (m_PBRVertSpv.empty() || m_PBRFragSpv.empty()) return out;
        const auto shader = ShaderLibrary::Get("pbr_vert.slang");
        if (!shader) return out;
        const UUID pbrUUID = shader->Handle;
        auto* initial = m_GeoPipelineManager.GetOrCreate(pbrUUID, Material::RenderMode::Opaque,
            Material::CullMode::Back, out.polygon, m_PBRVertSpv, m_PBRFragSpv);
        if (!initial) return out;
        out.initialLayout = initial->GetLayout();
        const auto prepareBucket = [&](const auto& bucket, Material::RenderMode mode, bool overlay) {
            VKPipeline* selected = nullptr;
            Material::CullMode currentCull = static_cast<Material::CullMode>(0xFF);
            bool currentDeformed = false;
            UUID currentFrag = UUID::Invalid();
            for (const auto& dc : bucket)
            {
                if (!dc.model) continue;
                VKPipeline* pipeline = nullptr;
                if (overlay)
                    pipeline = dc.isDeformed ? m_WireframeOverlaySkinnedPipeline.get() : m_WireframeOverlayPipeline.get();
                else if (dc.cullMode != currentCull || dc.isDeformed != currentDeformed || dc.fragShaderUUID != currentFrag)
                {
                    currentCull = dc.cullMode; currentDeformed = dc.isDeformed; currentFrag = dc.fragShaderUUID;
                    const UUID key = dc.fragShaderUUID.IsValid() ? dc.fragShaderUUID : pbrUUID;
                    const auto& frag = ResolveFragSpv(dc.fragShaderUUID);
                    pipeline = dc.isDeformed
                        ? m_GeoSkinnedPipelineManager.GetOrCreate(key, mode, dc.cullMode, out.polygon, m_PBRSkinnedVertSpv, frag)
                        : m_GeoPipelineManager.GetOrCreate(key, mode, dc.cullMode, out.polygon, m_PBRVertSpv, frag);
                    selected = pipeline;
                }
                else pipeline = selected;
                if (!pipeline) continue;
                auto mesh = dc.model->GetMesh(dc.meshIndex);
                if (!mesh) continue;
                auto vb = std::static_pointer_cast<VKVertexBuffer>(mesh->GetVertexBuffer());
                auto ib = std::static_pointer_cast<VKIndexBuffer>(mesh->GetIndexBuffer());
                if (!ib || (!vb && (!overlay || !dc.isDeformed))) continue;
                ForwardDrawPacket packet;
                packet.mesh = mesh; packet.pipeline = pipeline->GetHandle(); packet.layout = pipeline->GetLayout();
                packet.vertex = vb ? vb->GetVulkanBuffer() : VK_NULL_HANDLE; packet.index = ib->GetVulkanBuffer();
                packet.indexCount = ib->GetCount(); packet.objectIndex = dc.gpuObjectIndex; packet.entityIndex = dc.entityIndex;
                packet.deformed = dc.isDeformed; packet.mode = static_cast<u32>(mode);
                packet.cull = dc.cullMode == Material::CullMode::Back ? VK_CULL_MODE_BACK_BIT
                    : dc.cullMode == Material::CullMode::Front ? VK_CULL_MODE_FRONT_BIT : VK_CULL_MODE_NONE;
                if (!overlay) packet.indirectOffset = ForwardDrawOffset(visible, dc.gpuObjectIndex);
                if (!overlay && captureDraws)
                {
                    packet.meshName = dc.model->GetName() + "[" + std::to_string(dc.meshIndex) + "]";
                    packet.entityName = "Entity";
                    const auto entity = entt::to_entity(dc.entity);
                    if (entity < snapshot.tagsByEntity.size() && snapshot.tagsByEntity[entity])
                        packet.entityName = snapshot.tagsByEntity[entity];
                }
                (overlay ? out.overlays : out.draws).push_back(std::move(packet));
            }
        };
        prepareBucket(draws.opaque, Material::RenderMode::Opaque, false);
        prepareBucket(draws.cutout, Material::RenderMode::Cutout, false);
        if (shadedWireframe && m_WireframeOverlayPipeline)
        {
            prepareBucket(draws.opaque, Material::RenderMode::Opaque, true);
            prepareBucket(draws.cutout, Material::RenderMode::Cutout, true);
            prepareBucket(draws.transparent, Material::RenderMode::Transparent, true);
        }
        return out;
    }

    std::array<RG::ResourceHandle, 3> GeometrySubsystem::AddForwardOpaquePass(RG::RenderGraph& graph,
        RG::ResourceHandle color, RG::ResourceHandle depth, RG::ResourceHandle picking,
        const VisibleDrawRange& visible, u32 width, u32 height, const ForwardOpaqueBindings& bindings,
        std::span<const RG::ResourceHandle> sampledImages, std::span<const RG::BufferHandle> lightBuffers, FrameDebugger* debugger)
    {
        LH_PROFILE_FUNCTION();
        struct Data { RG::ResourceHandle color, depth, picking; };
        std::array<RG::ResourceHandle, 3> output;
        const auto indirectBuffer = visible.indirect.binding.slice->buffer;
        auto metadata = RG::RenderPassMetadata::Graphics("pbr", true, true, false, VK_CULL_MODE_BACK_BIT, 0);
        metadata.indirectDraws = true; metadata.pipelineStateMixed = true; // Material and overlay PSO variants.
        if (bindings.initialLayout) { metadata.AddDraws(bindings.draws); metadata.AddDraws(bindings.overlays); }
        graph.AddPass<Data>("GeometryPass",
            [&](Data& data, RG::RenderPassBuilder& builder) {
                builder.SetDebugMetadata(metadata);
                data.depth = builder.WriteDepth(depth, VK_ATTACHMENT_LOAD_OP_LOAD, VK_ATTACHMENT_STORE_OP_STORE);
                data.color = builder.Write(color);
                VkClearValue clear{};
                data.picking = builder.Write(picking, VK_ATTACHMENT_LOAD_OP_CLEAR, VK_ATTACHMENT_STORE_OP_STORE, clear);
                for (auto image : sampledImages) builder.Read(image);
                for (auto buffer : lightBuffers) builder.ReadBufferFragment(buffer);
                builder.ReadIndirectBuffer(visible.indirect.handle);
                output = {data.color, data.depth, data.picking};
            },
            [bindings, indirectBuffer, width, height, debugger](Data&, RG::RenderPassContext& ctx) {
                const auto cmd = ctx.commandBuffer;
                if (debugger) debugger->BeginCapturePass(ctx.passIndex, "GeometryPass", "SceneColor", false,
                    {"pbr", 0, VK_CULL_MODE_BACK_BIT, bindings.polygon, false, true, true, false});
                if (bindings.initialLayout)
                {
                    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, bindings.initialLayout,
                        0, 6, bindings.sets.data(), 0, nullptr);
                    VkViewport viewport{};
                    viewport.width = float(width); viewport.height = float(height); viewport.maxDepth = 1.0f;
                    vkCmdSetViewport(cmd, 0, 1, &viewport);
                    const VkRect2D scissor{{0, 0}, {width, height}};
                    vkCmdSetScissor(cmd, 0, 1, &scissor);
                    const auto recordBucket = [&](const auto& packets, bool overlay) {
                        VkPipeline current = VK_NULL_HANDLE;
                        for (const auto& packet : packets)
                        {
                            if (packet.pipeline != current)
                            {
                                current = packet.pipeline;
                                vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, current);
                                if (overlay) vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                    packet.layout, 0, 6, bindings.sets.data(), 0, nullptr);
                            }
                            if (!packet.deformed)
                            {
                                const VkDeviceSize offset = 0;
                                vkCmdBindVertexBuffers(cmd, 0, 1, &packet.vertex, &offset);
                            }
                            vkCmdBindIndexBuffer(cmd, packet.index, 0, VK_INDEX_TYPE_UINT32);
                            if (overlay) vkCmdDrawIndexed(cmd, packet.indexCount, 1, 0, 0, packet.objectIndex);
                            else
                            {
                                vkCmdDrawIndexedIndirect(cmd, indirectBuffer, packet.indirectOffset, 1, sizeof(VkDrawIndexedIndirectCommand));
                                if (debugger && bindings.captureDraws)
                                    debugger->CaptureIndirectDraw("GeometryPass", packet.meshName, packet.entityName,
                                        packet.entityIndex, packet.indexCount, packet.objectIndex, packet.indirectOffset,
                                        {"pbr", packet.mode, packet.cull, bindings.polygon, packet.deformed, true, true, false});
                            }
                        }
                    };
                    recordBucket(bindings.draws, false);
                    recordBucket(bindings.overlays, true);
                }
                if (debugger) debugger->EndCapturePass();
            });
        return output;
    }
}

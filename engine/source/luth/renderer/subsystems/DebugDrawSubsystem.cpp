#include "luthpch.h"

#include "luth/renderer/subsystems/DebugDrawSubsystem.h"

#include "luth/core/DebugDraw.h"
#include "luth/renderer/shader/ShaderLibrary.h"
#include "luth/renderer/backend/vulkan/VulkanContext.h"

namespace Luth
{
    void DebugDrawSubsystem::Init()
    {
        LH_PROFILE_FUNCTION();

        auto loadSpv = [](const char* relPath) -> std::vector<u32> {
            auto sh = ShaderLibrary::LoadEngine(relPath);
            return sh ? sh->GetSpirV() : std::vector<u32>{};
        };
        m_VertSpv = loadSpv("shaders/debugDraw_vert.slang");
        m_FragSpv = loadSpv("shaders/debugDraw.slang");

        if (m_VertSpv.empty() || m_FragSpv.empty())
        {
            LH_LOG(Renderer, error, "DebugDrawSubsystem: shader SPIR-V empty after asset load!");
        }
    }

    void DebugDrawSubsystem::BuildPipelines()
    {
        LH_PROFILE_FUNCTION();
        BuildLinePipeline();
    }

    void DebugDrawSubsystem::BuildLinePipeline()
    {
        LH_PROFILE_FUNCTION();
        if (m_VertSpv.empty() || m_FragSpv.empty()) return;

        // No descriptor set: viewProj rides on the push constant; vertex data binds as VBO.
        std::vector<VkDescriptorSetLayout> layouts;

        VkPushConstantRange pcRange{};
        pcRange.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
        pcRange.offset     = 0;
        pcRange.size       = sizeof(Mat4);

        // DebugVertex: position (vec3) + colorRGBA (RGBA8 unorm). 16 bytes total.
        VkVertexInputBindingDescription binding{};
        binding.binding   = 0;
        binding.stride    = sizeof(DebugVertex);
        binding.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

        VkVertexInputAttributeDescription attribs[2] = {};
        attribs[0].location = 0;
        attribs[0].binding  = 0;
        attribs[0].format   = VK_FORMAT_R32G32B32_SFLOAT;
        attribs[0].offset   = static_cast<u32>(offsetof(DebugVertex, position));
        attribs[1].location = 1;
        attribs[1].binding  = 0;
        attribs[1].format   = VK_FORMAT_R8G8B8A8_UNORM;
        attribs[1].offset   = static_cast<u32>(offsetof(DebugVertex, colorRGBA));

        PipelineConfig cfg;
        cfg.colorFormats          = { VK_FORMAT_R8G8B8A8_UNORM };  // matches LDR target
        cfg.depthFormat           = VK_FORMAT_UNDEFINED;          // depth disabled; always-visible
        cfg.depthTest             = false;
        cfg.depthWrite            = false;
        cfg.blendEnabled          = true;                         // alpha-blend over scene
        cfg.cullMode              = VK_CULL_MODE_NONE;
        cfg.frontFace             = VK_FRONT_FACE_COUNTER_CLOCKWISE;
        cfg.topology              = VK_PRIMITIVE_TOPOLOGY_LINE_LIST;
        cfg.polygonMode           = VK_POLYGON_MODE_FILL;
        cfg.bindingDescriptions   = { binding };
        cfg.attributeDescriptions = { attribs[0], attribs[1] };
        cfg.pushConstantRanges    = { pcRange };

        m_LinePipeline = std::make_unique<VKPipeline>(cfg, m_VertSpv, m_FragSpv, layouts);
    }

    void DebugDrawSubsystem::Shutdown()
    {
        LH_PROFILE_FUNCTION();
        m_LinePipeline.reset();
    }

    bool DebugDrawSubsystem::OnShaderReloaded(const std::string& name, const std::vector<u32>& spv)
    {
        LH_PROFILE_FUNCTION();
        if      (name == "debugDraw_vert.slang") m_VertSpv = spv;
        else if (name == "debugDraw.slang") m_FragSpv = spv;
        else return false;

        if (auto* raw = m_LinePipeline.release(); raw)
            VulkanContext::Get().PushDeletion([raw]() { delete raw; });
        BuildLinePipeline();
        return true;
    }

}

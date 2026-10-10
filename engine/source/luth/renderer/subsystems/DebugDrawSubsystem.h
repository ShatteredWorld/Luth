#pragma once

#include "luth/core/types/LuthTypes.h"
#include "luth/renderer/features/DebugDrawBindings.h"
#include "luth/renderer/rendergraph/RenderGraph.h"
#include "luth/renderer/backend/vulkan/VulkanPipeline.h"

#include <memory>
#include <string>
#include <vector>

namespace Luth
{
    struct FrameDebugger;

    // Render-side companion to luth/core/DebugDraw. Owns a single line-list graphics pipeline and
    // drains the current frame's queued line endpoints into a transient vertex buffer carved from
    // the GPUTaggedPageAllocator. No descriptor set: viewProj rides on a push constant. The pass
    // slots after the outline pass and before ImGui so colliders render on top of the scene but
    // under UI; gated by RenderView::drawDebugShapes (scene view only by default).
    class DebugDrawSubsystem
    {
    public:
        void Init();
        void BuildPipelines();
        void Shutdown();

        bool OnShaderReloaded(const std::string& name, const std::vector<u32>& spv);

        DebugDrawBindings PrepareBindings(std::span<const DebugVertex>, const Mat4& viewProj, u64 renderFrameIndex, bool enabled);
        RG::ResourceHandle AddDebugDrawPass(RG::RenderGraph&, RG::ResourceHandle ldrOutput,
            const DebugDrawBindings&, FrameDebugger*);

    private:
        void BuildLinePipeline();

        std::unique_ptr<VKPipeline> m_LinePipeline;
        std::vector<u32>            m_VertSpv;
        std::vector<u32>            m_FragSpv;
    };
}

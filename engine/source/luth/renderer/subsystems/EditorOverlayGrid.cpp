#include "luthpch.h"
#include "luth/renderer/subsystems/EditorOverlaysSubsystem.h"
#include "luth/renderer/CameraParams.h"
#include "luth/renderer/FrameDebugger.h"
namespace Luth
{
    GridBindings EditorOverlaysSubsystem::PrepareGridBindings(std::shared_ptr<EditorOverlayViewState> state,
        const CameraParams& camera, Vec2 jitter, u64 frame, bool enabled) const
    {
        GridBindings packet; packet.enabled = enabled;
        if (!enabled || !m_GridPipeline) return packet;
        if (!state || !state->sources[2]) throw std::invalid_argument("Grid: missing view state");
        packet.pipeline = m_GridPipeline->GetHandle(); packet.layout = m_GridPipeline->GetLayout();
        packet.set = state->gridSets[frame % MAX_FRAMES_IN_FLIGHT];
        packet.depth = {state->sources[2].get()}; packet.state = std::move(state);
        auto& pc = packet.parameters;
        for (u32 i = 0; i < 4; ++i)
        {
            pc.axisXColor[i] = camera.gridAxisXColor[i]; pc.axisZColor[i] = camera.gridAxisZColor[i];
            pc.gridColor[i] = camera.gridColor[i];
        }
        pc.majorScale = camera.gridMajorScale; pc.fadeStart = camera.gridFadeStart;
        pc.fadeEnd = camera.gridFadeEnd; pc.lineThickness = camera.gridLineThickness;
        pc.jitter[0] = jitter.x; pc.jitter[1] = jitter.y;
        return packet;
    }
    RG::ResourceHandle EditorOverlaysSubsystem::AddGridPass(RG::RenderGraph& graph,
        RG::ResourceHandle color, RG::ResourceHandle depth, const GridBindings& packet, FrameDebugger* debugger)
    {
        struct Data { RG::ResourceHandle color, depth; };
        RG::ResourceHandle output;
        graph.AddPass<Data>("GridPass",
            [&](Data& data, RG::RenderPassBuilder& builder) {
                output = data.color = builder.Write(color, VK_ATTACHMENT_LOAD_OP_LOAD, VK_ATTACHMENT_STORE_OP_STORE);
                data.depth = builder.Read(depth);
            },
            [packet, debugger](Data& data, RG::RenderPassContext& ctx) {
                if (debugger) debugger->BeginCapturePass(ctx.passIndex, "GridPass", "SceneColor", false,
                    {"grid", 0, VK_CULL_MODE_NONE, VK_POLYGON_MODE_FILL, false, false, false, true});
                const auto* res = static_cast<const RG::RenderGraph::ResourceNode*>(ctx.GetResource(data.color));
                VkCommandBuffer cmd = ctx.commandBuffer;
                vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, packet.pipeline);
                vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, packet.layout, 0, 1, &packet.set, 0, nullptr);
                VkViewport vp{}; vp.width = (float)res->desc.width; vp.height = (float)res->desc.height; vp.maxDepth = 1;
                vkCmdSetViewport(cmd, 0, 1, &vp);
                VkRect2D sc{}; sc.extent = {res->desc.width, res->desc.height}; vkCmdSetScissor(cmd, 0, 1, &sc);
                vkCmdPushConstants(cmd, packet.layout, VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(packet.parameters), &packet.parameters);
                vkCmdDraw(cmd, 3, 1, 0, 0);
                if (debugger)
                {
                    ObjectPushConstants dummy{};
                    debugger->CaptureDrawCall("GridPass", "FullscreenTriangle", "GridPass", 0, 0, dummy,
                        {"grid", 0, VK_CULL_MODE_NONE, VK_POLYGON_MODE_FILL, false, false, false, true});
                    debugger->EndCapturePass();
                }
            });
        return output;
    }
}

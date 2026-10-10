#include "luthpch.h"
#include "luth/renderer/subsystems/EditorOverlaysSubsystem.h"
#include "luth/renderer/CameraParams.h"
#include "luth/renderer/FrameDebugger.h"
namespace Luth
{
    OutlinePushConstants MakeOutlineParameters(const CameraParams& camera, u32 width, u32 height)
    {
        if (!width || !height) throw std::invalid_argument("Outline: zero view extent");
        return {camera.outlineWidth, 1.0f / width, 1.0f / height,
            camera.outlineColor.r, camera.outlineColor.g, camera.outlineColor.b, camera.outlineColor.a,
            camera.outlineOccludedAlpha};
    }
    OutlineBindings EditorOverlaysSubsystem::PrepareOutlineBindings(std::shared_ptr<EditorOverlayViewState> state,
        const CameraParams& camera, u32 width, u32 height, bool enabled) const
    {
        OutlineBindings packet; packet.enabled = enabled;
        if (!enabled || !m_OutlinePipeline) return packet;
        if (!state) throw std::invalid_argument("Outline: missing view state");
        packet.pipeline = m_OutlinePipeline->GetHandle(); packet.layout = m_OutlinePipeline->GetLayout();
        packet.set = state->outlineSet; packet.state = std::move(state);
        packet.parameters = MakeOutlineParameters(camera, width, height);
        return packet;
    }
    RG::ResourceHandle EditorOverlaysSubsystem::AddOutlinePass(RG::RenderGraph& graph, RG::ResourceHandle color,
        RG::ResourceHandle mask, RG::ResourceHandle selectedDepth, RG::ResourceHandle depth,
        const OutlineBindings& packet, FrameDebugger* debugger)
    {
        struct Data { RG::ResourceHandle output, mask, selectedDepth, depth; };
        RG::ResourceHandle output;
        graph.AddPass<Data>("OutlinePass",
            [&](Data& data, RG::RenderPassBuilder& builder) {
                builder.SetDebugMetadata(RG::RenderPassMetadata::Graphics("outline", false, false, true, VK_CULL_MODE_NONE));
                output = data.output = builder.Write(color, VK_ATTACHMENT_LOAD_OP_LOAD, VK_ATTACHMENT_STORE_OP_STORE);
                data.mask = builder.Read(mask); data.selectedDepth = builder.Read(selectedDepth); data.depth = builder.Read(depth);
            },
            [packet, debugger](Data& data, RG::RenderPassContext& ctx) {
                if (debugger) debugger->BeginCapturePass(ctx.passIndex, "OutlinePass", "LDROutput", false,
                    {"outline", 0, VK_CULL_MODE_NONE, VK_POLYGON_MODE_FILL, false, false, false, true});
                const auto cmd = ctx.commandBuffer;
                vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, packet.pipeline);
                vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, packet.layout, 0, 1, &packet.set, 0, nullptr);
                const auto* res = static_cast<const RG::RenderGraph::ResourceNode*>(ctx.GetResource(data.output));
                VkViewport vp{}; vp.width = (float)res->desc.width; vp.height = (float)res->desc.height; vp.maxDepth = 1;
                vkCmdSetViewport(cmd, 0, 1, &vp); VkRect2D sc{}; sc.extent = {res->desc.width, res->desc.height};
                vkCmdSetScissor(cmd, 0, 1, &sc);
                vkCmdPushConstants(cmd, packet.layout, VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(packet.parameters), &packet.parameters);
                vkCmdDraw(cmd, 3, 1, 0, 0);
                if (debugger)
                {
                    ObjectPushConstants dummy{};
                    debugger->CaptureDrawCall("OutlinePass", "FullscreenTriangle", "OutlinePass", 0, 0, dummy,
                        {"outline", 0, VK_CULL_MODE_NONE, VK_POLYGON_MODE_FILL, false, false, false, true});
                    debugger->EndCapturePass();
                }
            });
        return output;
    }
}

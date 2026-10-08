#include "luthpch.h"
#include "luth/renderer/subsystems/PostProcessSubsystem.h"
#include "luth/renderer/FrameDebugger.h"
#include <cmath>
namespace Luth
{
    SlimVizBindings PostProcessSubsystem::PrepareSlimVizBindings(VkDescriptorSet set,
        const std::array<std::shared_ptr<Texture>, 4>& sources, u32 mode, float scale, bool enabled) const
    {
        SlimVizBindings packet; packet.enabled = enabled;
        if (!enabled || !m_SlimVizPipeline) return packet;
        if (!set || mode > 3 || !std::isfinite(scale) ||
            std::any_of(sources.begin(), sources.end(), [](const auto& source) { return !source; }))
            throw std::invalid_argument("SlimViz: incomplete native sources or parameters");
        packet.pipeline = m_SlimVizPipeline->GetHandle(); packet.layout = m_SlimVizPipeline->GetLayout();
        packet.set = set; packet.sources = sources; packet.parameters = {mode, scale};
        return packet;
    }
    RG::ResourceHandle PostProcessSubsystem::AddSlimVizPass(RG::RenderGraph& graph, RG::ResourceHandle input,
        const std::array<RG::ResourceHandle, 4>& sources, const SlimVizBindings& packet, FrameDebugger* debugger)
    {
        struct Data { RG::ResourceHandle output; std::array<RG::ResourceHandle, 4> sources; };
        RG::ResourceHandle output;
        graph.AddPass<Data>("SlimVizPass",
            [&](Data& data, RG::RenderPassBuilder& builder) {
                VkClearValue clear{}; clear.color = {{0, 0, 0, 1}};
                output = data.output = builder.Write(input, VK_ATTACHMENT_LOAD_OP_CLEAR, VK_ATTACHMENT_STORE_OP_STORE, clear);
                for (size_t i = 0; i < sources.size(); ++i) data.sources[i] = builder.Read(sources[i]);
            },
            [packet, debugger](Data& data, RG::RenderPassContext& ctx) {
                if (debugger) debugger->BeginCapturePass(ctx.passIndex, "SlimVizPass", "LDROutput", false,
                    {"slim_viz", 0, VK_CULL_MODE_NONE, VK_POLYGON_MODE_FILL, false, false, false, false});
                const auto cmd = ctx.commandBuffer;
                vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, packet.pipeline);
                vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, packet.layout, 0, 1, &packet.set, 0, nullptr);
                vkCmdPushConstants(cmd, packet.layout, VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(packet.parameters), &packet.parameters);
                const auto* res = static_cast<const RG::RenderGraph::ResourceNode*>(ctx.GetResource(data.output));
                VkViewport vp{}; vp.width = (float)res->desc.width; vp.height = (float)res->desc.height; vp.maxDepth = 1;
                vkCmdSetViewport(cmd, 0, 1, &vp); VkRect2D sc{}; sc.extent = {res->desc.width, res->desc.height};
                vkCmdSetScissor(cmd, 0, 1, &sc); vkCmdDraw(cmd, 3, 1, 0, 0);
                if (debugger)
                {
                    ObjectPushConstants dummy{};
                    debugger->CaptureDrawCall("SlimVizPass", "FullscreenTriangle", "SlimViz", 0, 0, dummy,
                        {"slim_viz", 0, VK_CULL_MODE_NONE, VK_POLYGON_MODE_FILL, false, false, false, false});
                    debugger->EndCapturePass();
                }
            });
        return output;
    }
}

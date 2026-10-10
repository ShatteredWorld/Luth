#include "luthpch.h"
#include "luth/renderer/subsystems/LightingSubsystem.h"
#include "luth/renderer/features/ClusterVizBindings.h"
#include "luth/renderer/FrameDebugger.h"
namespace Luth
{
    ClusterVizBindings LightingSubsystem::PrepareClusterVizBindings(std::shared_ptr<ClusterVizViewState> state, VkDescriptorSet lightingSet,
        const Memory::GPUSubRegion& grid,
        u32 width, u32 height, float nearZ, float farZ, bool enabled) const
    {
        ClusterVizBindings packet; packet.enabled = enabled;
        if (!enabled || !m_ClusterVizPipeline) return packet;
        if (!state || !state->set || !state->depth || !lightingSet)
            throw std::invalid_argument("ClusterViz: incomplete native view bindings");
        packet.pipeline = m_ClusterVizPipeline->GetHandle(); packet.layout = m_ClusterVizPipeline->GetLayout();
        packet.sets = {state->set, lightingSet}; packet.state = std::move(state); packet.grid = {grid.buffer, grid.offset, grid.size};
        packet.parameters = {Vec2(float(width), float(height)), nearZ, farZ}; return packet;
    }
    RG::ResourceHandle LightingSubsystem::AddClusterVizPass(RG::RenderGraph& graph, RG::ResourceHandle input,
        RG::ResourceHandle depth, RG::BufferHandle grid, const ClusterVizBindings& packet, FrameDebugger* debugger)
    {
        struct Data { RG::ResourceHandle output, depth; RG::BufferHandle grid; };
        RG::ResourceHandle output;
        graph.AddPass<Data>("ClusterVizPass", [&](Data& data, RG::RenderPassBuilder& builder) {
            builder.SetDebugMetadata(RG::RenderPassMetadata::Graphics("cluster_viz", false, false, false, VK_CULL_MODE_NONE, 1));
            VkClearValue clear{}; clear.color = {{0, 0, 0, 1}};
            output = data.output = builder.Write(input, VK_ATTACHMENT_LOAD_OP_LOAD, VK_ATTACHMENT_STORE_OP_STORE, clear);
            data.depth = builder.Read(depth); data.grid = builder.ReadBufferFragment(grid);
        }, [packet, debugger](Data& data, RG::RenderPassContext& ctx) {
            if (debugger) debugger->BeginCapturePass(ctx.passIndex, "ClusterVizPass", "LDROutput", false,
                {"cluster_viz", 0, VK_CULL_MODE_NONE, VK_POLYGON_MODE_FILL, false, false, false, false});
            const auto cmd = ctx.commandBuffer;
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, packet.pipeline);
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, packet.layout, 0, 2, packet.sets.data(), 0, nullptr);
            vkCmdPushConstants(cmd, packet.layout, VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(packet.parameters), &packet.parameters);
            const auto* res = static_cast<const RG::RenderGraph::ResourceNode*>(ctx.GetResource(data.output));
            VkViewport vp{}; vp.width = (float)res->desc.width; vp.height = (float)res->desc.height; vp.maxDepth = 1;
            vkCmdSetViewport(cmd, 0, 1, &vp); VkRect2D sc{}; sc.extent = {res->desc.width, res->desc.height};
            vkCmdSetScissor(cmd, 0, 1, &sc); vkCmdDraw(cmd, 3, 1, 0, 0);
            if (debugger)
            {
                ObjectPushConstants dummy{};
                debugger->CaptureDrawCall("ClusterVizPass", "FullscreenTriangle", "ClusterViz", 0, 0, dummy,
                    {"cluster_viz", 0, VK_CULL_MODE_NONE, VK_POLYGON_MODE_FILL, false, false, false, false});
                debugger->EndCapturePass();
            }
        });
        return output;
    }
}

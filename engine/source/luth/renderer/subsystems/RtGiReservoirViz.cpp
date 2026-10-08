#include "luthpch.h"
#include "luth/renderer/subsystems/RtRestirGiSubsystem.h"
namespace Luth
{
    GiReservoirVizBindings RtRestirGiSubsystem::PrepareReservoirVizBindings(VkDescriptorSet set,
        std::shared_ptr<Texture> depth, const Memory::GPUSubRegion& reservoir, u32 width, u32 height,
        u32 resWidth, u32 resHeight, u32 temporalMCap, u32 neighbours, u32 maxAge, bool enabled) const
    {
        GiReservoirVizBindings packet; packet.enabled = enabled;
        if (!enabled || !m_ReservoirVizPipeline) return packet;
        packet.pipeline = m_ReservoirVizPipeline->GetHandle(); packet.layout = m_ReservoirVizPipeline->GetLayout();
        packet.set = set; packet.depth = std::move(depth); packet.reservoir = reservoir;
        packet.parameters = {float(width), float(height), float(resWidth), float(resHeight),
            static_cast<float>(temporalMCap * (neighbours + 1u) + 1u), float(maxAge)};
        return packet;
    }
    RG::ResourceHandle RtRestirGiSubsystem::AddReservoirVizPass(RG::RenderGraph& graph, RG::ResourceHandle input,
        RG::ResourceHandle depth, RG::BufferHandle reservoir, const GiReservoirVizBindings& packet)
    {
        struct Data { RG::ResourceHandle output, depth; RG::BufferHandle reservoir; };
        RG::ResourceHandle output;
        graph.AddPass<Data>("GiReservoirVizPass", [&](Data& data, RG::RenderPassBuilder& builder) {
            VkClearValue clear{}; clear.color = {{0, 0, 0, 1}};
            output = data.output = builder.Write(input, VK_ATTACHMENT_LOAD_OP_LOAD, VK_ATTACHMENT_STORE_OP_STORE, clear);
            data.depth = builder.Read(depth); data.reservoir = builder.ReadBufferFragment(reservoir);
        }, [packet](Data&, RG::RenderPassContext& ctx) {
            const auto cmd = ctx.commandBuffer;
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, packet.pipeline);
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, packet.layout, 0, 1, &packet.set, 0, nullptr);
            vkCmdPushConstants(cmd, packet.layout, VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(packet.parameters), &packet.parameters);
            VkViewport vp{}; vp.width = packet.parameters.vx; vp.height = packet.parameters.vy; vp.maxDepth = 1;
            vkCmdSetViewport(cmd, 0, 1, &vp); VkRect2D sc{}; sc.extent = {u32(packet.parameters.vx), u32(packet.parameters.vy)};
            vkCmdSetScissor(cmd, 0, 1, &sc); vkCmdDraw(cmd, 3, 1, 0, 0);
        });
        return output;
    }
}

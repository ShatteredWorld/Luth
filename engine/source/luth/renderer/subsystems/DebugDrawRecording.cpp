#include "luthpch.h"
#include "luth/renderer/subsystems/DebugDrawSubsystem.h"
#include "luth/renderer/FrameDebugger.h"
#include "luth/jobs/JobSystem.h"
#include <cmath>
#include <limits>
namespace Luth
{
    DebugDrawBindings DebugDrawSubsystem::PrepareBindings(std::span<const DebugVertex> lines,
        const Mat4& viewProj, u64 frame, bool enabled)
    {
        DebugDrawBindings packet; packet.enabled = enabled;
        if (!enabled || !m_LinePipeline || lines.empty()) return packet;
        if (lines.size() % 2 || lines.size() > std::numeric_limits<u32>::max())
            throw std::invalid_argument("DebugDraw: invalid line endpoint count");
        for (u32 col = 0; col < 4; ++col) for (u32 row = 0; row < 4; ++row)
            if (!std::isfinite(viewProj[col][row])) throw std::invalid_argument("DebugDraw: non-finite view matrix");
        auto* job = JobSystem::GetCurrentJobContext();
        if (!job) throw std::runtime_error("DebugDraw: upload preparation requires a render job context");
        auto& cache = job->GpuCache; cache.CurrentTag = static_cast<u32>(frame);
        auto& allocator = Memory::GPUTaggedPageAllocator::Get();
        packet.vertices = allocator.Allocate(cache, lines.size_bytes(), 16);
        if (!packet.vertices.buffer || !packet.vertices.mappedPtr || packet.vertices.size < lines.size_bytes())
            throw std::runtime_error("DebugDraw: vertex upload allocation failed");
        std::memcpy(packet.vertices.mappedPtr, lines.data(), lines.size_bytes()); allocator.FlushRegion(packet.vertices);
        packet.pipeline = m_LinePipeline->GetHandle(); packet.layout = m_LinePipeline->GetLayout();
        packet.viewProj = viewProj; packet.renderFrameIndex = frame; packet.vertexCount = static_cast<u32>(lines.size());
        return packet;
    }
    RG::ResourceHandle DebugDrawSubsystem::AddDebugDrawPass(RG::RenderGraph& graph, RG::ResourceHandle input,
        const DebugDrawBindings& packet, FrameDebugger* debugger)
    {
        struct Data { RG::ResourceHandle output; };
        RG::ResourceHandle output;
        graph.AddPass<Data>("DebugDrawPass", [&](Data& data, RG::RenderPassBuilder& builder) {
            builder.SetDebugMetadata(RG::RenderPassMetadata::Graphics("debugDraw", false, false, true, VK_CULL_MODE_NONE));
            output = data.output = builder.Write(input, VK_ATTACHMENT_LOAD_OP_LOAD, VK_ATTACHMENT_STORE_OP_STORE);
        }, [packet, debugger](Data& data, RG::RenderPassContext& ctx) {
            if (debugger) debugger->BeginCapturePass(ctx.passIndex, "DebugDrawPass", "LDROutput", false,
                {"debugDraw", 0, VK_CULL_MODE_NONE, VK_POLYGON_MODE_FILL, false, false, false, true});
            const auto cmd = ctx.commandBuffer;
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, packet.pipeline);
            const auto* res = static_cast<const RG::RenderGraph::ResourceNode*>(ctx.GetResource(data.output));
            VkViewport vp{}; vp.width = (float)res->desc.width; vp.height = (float)res->desc.height; vp.maxDepth = 1;
            vkCmdSetViewport(cmd, 0, 1, &vp); VkRect2D sc{}; sc.extent = {res->desc.width, res->desc.height};
            vkCmdSetScissor(cmd, 0, 1, &sc);
            vkCmdPushConstants(cmd, packet.layout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(Mat4), &packet.viewProj);
            const VkDeviceSize offset = packet.vertices.offset;
            vkCmdBindVertexBuffers(cmd, 0, 1, &packet.vertices.buffer, &offset); vkCmdDraw(cmd, packet.vertexCount, 1, 0, 0);
            if (debugger)
            {
                ObjectPushConstants dummy{};
                debugger->CaptureDrawCall("DebugDrawPass", "Lines", "DebugDrawPass", 0, packet.vertexCount, dummy,
                    {"debugDraw", 0, VK_CULL_MODE_NONE, VK_POLYGON_MODE_FILL, false, false, false, true});
                debugger->EndCapturePass();
            }
        });
        return output;
    }
}

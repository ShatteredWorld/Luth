#include "luthpch.h"
#include "luth/renderer/subsystems/TransparencySubsystem.h"
#include "luth/renderer/subsystems/RtSubsystem.h"
#include "luth/renderer/features/TransparencyBindings.h"
#include "luth/renderer/draw/DrawList.h"
#include "luth/core/RenderSnapshot.h"
#include "luth/renderer/resources/Model.h"
#include "luth/renderer/backend/vulkan/VulkanBuffer.h"
#include "luth/renderer/FrameDebugger.h"

namespace Luth
{
    std::vector<u32> TransparencySubsystem::SortedOrder(const DrawList& draws, const Mat4& view)
    {
        std::vector<u32> order(draws.transparent.size());
        std::vector<f32> keys(order.size());
        for (u32 i = 0; i < order.size(); ++i)
        {
            order[i] = i; const auto& dc = draws.transparent[i]; Vec3 center(0.0f);
            if (dc.model && dc.meshIndex < dc.model->GetMeshesData().size())
                center = dc.model->GetMeshesData()[dc.meshIndex].BindPoseAABB.Center();
            const auto worldCenter = dc.modelMatrix * Vec4(center, 1.0f);
            keys[i] = -(view * worldCenter).z;
        }
        std::sort(order.begin(), order.end(), [&keys](u32 a, u32 b) { return keys[a] > keys[b]; });
        return order;
    }
    TransparencyBindings TransparencySubsystem::PrepareDrawBindings(GeometrySubsystem& geo,
        const std::array<VkDescriptorSet, 7>& sets, bool wireframe, bool captureDraws, const Mat4& view,
        const VisibleDrawRange& visible, const DrawList& draws, const RenderSnapshot& snapshot,
        TextureBindingRef fog, TextureBindingRef backdrop, const RtSubsystem* rayScene, bool oit)
    {
        TransparencyBindings out;
        out.sets = sets; out.polygon = wireframe ? VK_POLYGON_MODE_LINE : VK_POLYGON_MODE_FILL;
        out.captureDraws = captureDraws; out.fog = fog; out.backdrop = backdrop; out.rayScene = rayScene;
        out.oit = oit;
        const auto& fragment = oit ? m_OitStoreFragSpv : m_TransparentFragSpv;
        if (draws.transparent.empty() || fragment.empty()) return out;
        const auto& shaderId = oit ? m_OitShaderId : m_SortedShaderId;
        // Compile native variants during CPU preparation, before parallel recording uses them.
        auto order = oit ? std::vector<u32>(draws.transparent.size()) : SortedOrder(draws, view);
        if (oit) for (u32 i = 0; i < order.size(); ++i) order[i] = i;
        for (auto index : order)
        {
            const auto& dc = draws.transparent[index];
            if (!dc.model) continue;
            auto mesh = dc.model->GetMesh(dc.meshIndex); if (!mesh) continue;
            auto vb = std::static_pointer_cast<VKVertexBuffer>(mesh->GetVertexBuffer());
            auto ib = std::static_pointer_cast<VKIndexBuffer>(mesh->GetIndexBuffer());
            if (!vb || !ib) continue;
            auto* pipeline = dc.isDeformed
                ? (oit ? m_OitSkinnedPm : m_SortedSkinnedPm).GetOrCreate(shaderId, Material::RenderMode::Transparent,
                    dc.cullMode, out.polygon, geo.GetPBRSkinnedVertSpv(), fragment)
                : (oit ? m_OitPm : m_SortedPm).GetOrCreate(shaderId, Material::RenderMode::Transparent,
                    dc.cullMode, out.polygon, geo.GetPBRVertSpv(), fragment);
            if (!pipeline) continue;
            ForwardDrawPacket packet;
            packet.mesh = mesh; packet.pipeline = pipeline->GetHandle(); packet.layout = pipeline->GetLayout();
            packet.vertex = vb->GetVulkanBuffer(); packet.index = ib->GetVulkanBuffer();
            packet.indirectOffset = GeometrySubsystem::ForwardDrawOffset(visible, dc.gpuObjectIndex);
            packet.objectIndex = dc.gpuObjectIndex; packet.entityIndex = dc.entityIndex; packet.indexCount = ib->GetCount();
            packet.deformed = dc.isDeformed; packet.mode = static_cast<u32>(Material::RenderMode::Transparent);
            packet.cull = dc.cullMode == Material::CullMode::Back ? VK_CULL_MODE_BACK_BIT
                : dc.cullMode == Material::CullMode::Front ? VK_CULL_MODE_FRONT_BIT : VK_CULL_MODE_NONE;
            if (captureDraws)
            {
                packet.meshName = dc.model->GetName() + "[" + std::to_string(dc.meshIndex) + "]";
                packet.entityName = "Entity";
                const auto entity = entt::to_entity(dc.entity);
                if (entity < snapshot.tagsByEntity.size() && snapshot.tagsByEntity[entity])
                    packet.entityName = snapshot.tagsByEntity[entity];
            }
            out.draws.push_back(std::move(packet));
        }
        return out;
    }
    std::array<RG::ResourceHandle, 3> TransparencySubsystem::AddSortedPass(RG::RenderGraph& graph,
        RG::ResourceHandle color, RG::ResourceHandle picking, RG::ResourceHandle depth,
        const VisibleDrawRange& visible, u32 width, u32 height, const TransparencyBindings& packet,
        std::span<const RG::ResourceHandle> images, std::span<const RG::BufferHandle> buffers,
        bool fogValid, FrameDebugger* debugger)
    {
        struct Data { RG::ResourceHandle color, picking, depth; };
        std::array<RG::ResourceHandle, 3> outputs;
        const auto indirect = visible.indirect.binding.slice->buffer;
        graph.AddPass<Data>("TransparentPass",
            [&](Data& data, RG::RenderPassBuilder& builder) {
                auto metadata = RG::RenderPassMetadata::Graphics("pbr_transparent", true, false, true, VK_CULL_MODE_BACK_BIT, 0);
                metadata.AddDraws(packet.draws);
                metadata.indirectDraws = true; metadata.pipelineStateMixed = true;
                builder.SetDebugMetadata(std::move(metadata));
                data.color = builder.Write(color, VK_ATTACHMENT_LOAD_OP_LOAD, VK_ATTACHMENT_STORE_OP_STORE);
                data.picking = builder.Write(picking, VK_ATTACHMENT_LOAD_OP_LOAD, VK_ATTACHMENT_STORE_OP_STORE);
                data.depth = builder.WriteDepth(depth, VK_ATTACHMENT_LOAD_OP_LOAD, VK_ATTACHMENT_STORE_OP_STORE, {});
                for (auto image : images) builder.Read(image);
                for (auto buffer : buffers) builder.ReadBufferFragment(buffer);
                builder.ReadIndirectBuffer(visible.indirect.handle);
                outputs = {data.color, data.picking, data.depth};
            },
            [packet, indirect, width, height, fogValid, debugger](Data&, RG::RenderPassContext& ctx) {
                RecordDraws(packet, indirect, width, height, fogValid, 0, ctx, debugger);
            });
        return outputs;
    }
    void TransparencySubsystem::RecordDraws(const TransparencyBindings& packet, VkBuffer indirect,
        u32 width, u32 height, bool fogValid, u32 capacity, RG::RenderPassContext& ctx, FrameDebugger* debugger)
    {
        const char* name = packet.oit ? "OITStore" : "TransparentPass";
        const char* shader = packet.oit ? "pbr_oit_store" : "pbr_transparent";
        if (debugger) debugger->BeginCapturePass(ctx.passIndex, name, "SceneColor", false,
            {shader, 0, VK_CULL_MODE_BACK_BIT, packet.polygon, false, true, !packet.oit, true});
        TransparentPC pc{};
        pc.geomTable = packet.rayScene ? packet.rayScene->GetGeometryTableBDA() : 0;
        pc.flags = fogValid ? 1u : 0u;
        pc.nodeCapacity = capacity;
        VkViewport viewport{}; viewport.width = float(width); viewport.height = float(height); viewport.maxDepth = 1;
        const VkRect2D scissor{{0, 0}, {width, height}};
        vkCmdSetViewport(ctx.commandBuffer, 0, 1, &viewport); vkCmdSetScissor(ctx.commandBuffer, 0, 1, &scissor);
        VkPipeline bound = VK_NULL_HANDLE;
        for (const auto& draw : packet.draws)
        {
            if (bound != draw.pipeline)
            {
                bound = draw.pipeline;
                vkCmdBindPipeline(ctx.commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, bound);
                vkCmdBindDescriptorSets(ctx.commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, draw.layout,
                    0, 7, packet.sets.data(), 0, nullptr);
                vkCmdPushConstants(ctx.commandBuffer, draw.layout, VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(pc), &pc);
            }
            const VkDeviceSize offset = 0;
            if (!draw.deformed) vkCmdBindVertexBuffers(ctx.commandBuffer, 0, 1, &draw.vertex, &offset);
            vkCmdBindIndexBuffer(ctx.commandBuffer, draw.index, 0, VK_INDEX_TYPE_UINT32);
            vkCmdDrawIndexedIndirect(ctx.commandBuffer, indirect, draw.indirectOffset, 1, sizeof(VkDrawIndexedIndirectCommand));
            if (debugger && packet.captureDraws)
                debugger->CaptureIndirectDraw(name, draw.meshName, draw.entityName, draw.entityIndex,
                    draw.indexCount, draw.objectIndex, draw.indirectOffset,
                    {shader, draw.mode, draw.cull, packet.polygon, draw.deformed, true, !packet.oit, true});
        }
        if (debugger) debugger->EndCapturePass();
    }
}

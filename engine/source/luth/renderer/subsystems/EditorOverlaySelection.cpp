#include "luthpch.h"
#include "luth/renderer/subsystems/EditorOverlaysSubsystem.h"
#include "luth/renderer/CameraParams.h"
#include "luth/core/RenderSnapshot.h"
#include "luth/renderer/draw/DrawList.h"
#include "luth/renderer/FrameDebugger.h"
#include "luth/renderer/backend/vulkan/VulkanTexture.h"
#include "luth/renderer/backend/vulkan/VulkanBuffer.h"
namespace Luth
{
    SelectionMaskBindings EditorOverlaysSubsystem::PrepareSelectionMaskBindings(
        std::shared_ptr<EditorOverlayViewState> state, const std::array<VkDescriptorSet, 5>& sets,
        const CameraParams& camera, Vec2 jitter, const DrawList& draws, const RenderSnapshot& snapshot, bool enabled) const
    {
        SelectionMaskBindings packet; packet.enabled = enabled;
        if (!enabled) return packet;
        if (!state || !state->sources[0] || !state->sources[1])
            throw std::invalid_argument("SelectionMask: missing view state");
        packet.state = std::move(state); packet.sets = sets; packet.jitter = jitter;
        packet.width = packet.state->sources[0]->GetWidth(); packet.height = packet.state->sources[0]->GetHeight();
        for (u32 i = 0; i < 2; ++i)
        {
            const auto texture = std::static_pointer_cast<VKTexture>(packet.state->sources[i]);
            if (texture->GetWidth() != packet.width || texture->GetHeight() != packet.height)
                throw std::invalid_argument("SelectionMask: output extent mismatch");
            packet.images[i] = texture->GetImage(); packet.views[i] = texture->GetImageView();
        }
        if (!m_SelectionMaskPipeline) return packet; // Preserve the cold clear-only pass.
        packet.pipeline = m_SelectionMaskPipeline->GetHandle(); packet.layout = m_SelectionMaskPipeline->GetLayout();
        if (m_SelectionMaskSkinnedPipeline)
        {
            packet.skinnedPipeline = m_SelectionMaskSkinnedPipeline->GetHandle();
            packet.skinnedLayout = m_SelectionMaskSkinnedPipeline->GetLayout();
        }
        std::unordered_set<entt::entity> selected;
        CollectSelectedHandles(camera.selectedEntities, selected);
        const auto append = [&](const auto& bucket) {
            for (const auto& dc : bucket)
            {
                if (!selected.contains(dc.entity) || !dc.model) continue;
                const auto mesh = dc.model->GetMesh(dc.meshIndex); if (!mesh) continue;
                const auto vb = std::static_pointer_cast<VKVertexBuffer>(mesh->GetVertexBuffer());
                const auto ib = std::static_pointer_cast<VKIndexBuffer>(mesh->GetIndexBuffer());
                if (!vb || !ib) continue;
                SelectionMaskDraw draw; draw.vertex = vb->GetVulkanBuffer(); draw.index = ib->GetVulkanBuffer();
                draw.vertexOwner = vb; draw.indexOwner = ib; draw.indexCount = ib->GetCount();
                draw.entityIndex = dc.entityIndex; draw.skinned = dc.isSkinned;
                draw.constants.modelMatrix = dc.modelMatrix; draw.constants.boneOffset = dc.boneOffset;
                draw.meshName = dc.model->GetName() + "[" + std::to_string(dc.meshIndex) + "]";
                const u32 index = entt::to_entity(dc.entity);
                draw.entityName = index < snapshot.tagsByEntity.size() && snapshot.tagsByEntity[index]
                    ? snapshot.tagsByEntity[index] : "Entity";
                packet.draws.push_back(std::move(draw));
            }
        };
        append(draws.opaque); append(draws.cutout); append(draws.transparent);
        return packet;
    }
    SelectionMaskGraphOutput EditorOverlaysSubsystem::AddSelectionMaskPass(RG::RenderGraph& graph,
        const SelectionMaskBindings& packet, FrameDebugger* debugger)
    {
        struct Data { RG::ResourceHandle mask, depth; }; SelectionMaskGraphOutput output;
        graph.AddPass<Data>("SelectionMaskPass",
            [&](Data& data, RG::RenderPassBuilder& builder) {
                auto metadata = RG::RenderPassMetadata::Graphics("selectionMask", true, true, false, VK_CULL_MODE_NONE, 0);
                metadata.AddDraws(packet.draws);
                builder.SetDebugMetadata(std::move(metadata));
                RG::TextureDesc desc; desc.name = "SelectionMask"; desc.width = packet.width; desc.height = packet.height;
                desc.format = RG::TextureFormat::RGBA8_Unorm;
                data.mask = graph.ImportResource(desc, (void*)packet.images[0], (void*)packet.views[0], RG::ResourceState::Undefined);
                VkClearValue color{};
                output.mask = data.mask = builder.Write(data.mask, VK_ATTACHMENT_LOAD_OP_CLEAR, VK_ATTACHMENT_STORE_OP_STORE, color);
                desc.name = "SelectionDepth"; desc.format = RG::TextureFormat::D32_Float;
                data.depth = graph.ImportResource(desc, (void*)packet.images[1], (void*)packet.views[1], RG::ResourceState::Undefined);
                VkClearValue depth{}; depth.depthStencil = {1.0f, 0};
                output.depth = data.depth = builder.WriteDepth(data.depth, VK_ATTACHMENT_LOAD_OP_CLEAR, VK_ATTACHMENT_STORE_OP_STORE, depth);
            },
            [packet, debugger](Data&, RG::RenderPassContext& ctx) {
                if (debugger) debugger->BeginCapturePass(ctx.passIndex, "SelectionMaskPass", "SelectionMask", false,
                    {"selectionMask", 0, VK_CULL_MODE_NONE, VK_POLYGON_MODE_FILL, false, true, true, false});
                if (!packet.draws.empty())
                {
                    const auto cmd = ctx.commandBuffer;
                    VkViewport vp{}; vp.width = (float)packet.width; vp.height = (float)packet.height; vp.maxDepth = 1;
                    vkCmdSetViewport(cmd, 0, 1, &vp); VkRect2D sc{}; sc.extent = {packet.width, packet.height};
                    vkCmdSetScissor(cmd, 0, 1, &sc);
                    bool currentSkinned = false;
                    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, packet.pipeline);
                    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, packet.layout, 0, 5, packet.sets.data(), 0, nullptr);
                    for (const auto& draw : packet.draws)
                    {
                        const auto pipeline = draw.skinned && packet.skinnedPipeline ? packet.skinnedPipeline : packet.pipeline;
                        const auto layout = draw.skinned && packet.skinnedPipeline ? packet.skinnedLayout : packet.layout;
                        if (currentSkinned != draw.skinned)
                        {
                            currentSkinned = draw.skinned;
                            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
                            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, layout, 0, 5, packet.sets.data(), 0, nullptr);
                        }
                        const SelectionMaskPushConstants pc{draw.constants, packet.jitter};
                        vkCmdPushConstants(cmd, layout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(pc), &pc);
                        const VkDeviceSize offset = 0; vkCmdBindVertexBuffers(cmd, 0, 1, &draw.vertex, &offset);
                        vkCmdBindIndexBuffer(cmd, draw.index, 0, VK_INDEX_TYPE_UINT32);
                        vkCmdDrawIndexed(cmd, draw.indexCount, 1, 0, 0, 0);
                        if (debugger && debugger->IsRecordingCapture())
                            debugger->CaptureDrawCall("SelectionMaskPass", draw.meshName, draw.entityName, draw.entityIndex,
                                draw.indexCount, draw.constants, {"selectionMask", 0, VK_CULL_MODE_NONE, VK_POLYGON_MODE_FILL,
                                    draw.skinned, true, true, false});
                    }
                }
                if (debugger) debugger->EndCapturePass();
            });
        return output;
    }
}

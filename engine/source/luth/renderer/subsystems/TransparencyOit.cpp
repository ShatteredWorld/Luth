#include "luthpch.h"
#include "luth/renderer/subsystems/TransparencySubsystem.h"
#include "luth/renderer/backend/vulkan/VulkanTexture.h"
#include "luth/renderer/FrameDebugger.h"

namespace Luth
{
    TransparencyBindings TransparencySubsystem::PrepareSortedBindings(GeometrySubsystem& geo,
        const std::array<VkDescriptorSet, 7>& sets, bool wireframe, bool capture, const Mat4& view,
        const VisibleDrawRange& visible, const DrawList& draws, const RenderSnapshot& snapshot,
        TextureBindingRef fog, TextureBindingRef backdrop, const RtSubsystem* rt)
    {
        return PrepareDrawBindings(geo, sets, wireframe, capture, view, visible, draws, snapshot, fog, backdrop, rt, false);
    }
    TransparencyBindings TransparencySubsystem::PrepareTransparencyBindings(GeometrySubsystem& geo,
        const std::array<VkDescriptorSet, 7>& sets, bool wireframe, bool capture, const Mat4& view,
        const VisibleDrawRange& visible, const DrawList& draws, const RenderSnapshot& snapshot,
        TextureBindingRef fog, TextureBindingRef backdrop, const RtSubsystem* rt, bool requestedOit,
        const TransparencyViewState& state, u32 maxK)
    {
        // Preserve the native Sorted fallback when OIT resources or resolve are cold.
        const bool oit = requestedOit && state.oitHeads && state.oitNodes.buffer && m_ResolvePipeline;
        auto packet = PrepareDrawBindings(geo, sets, wireframe, capture, view, visible, draws, snapshot, fog, backdrop, rt, oit);
        if (!oit) return packet;
        if (state.oitNodes.size <= 16 || (state.oitNodes.size - 16) % 16 ||
            (state.oitNodes.size - 16) / 16 > UINT32_MAX)
            throw std::invalid_argument("Transparency: invalid OIT node allocation");
        const auto heads = std::static_pointer_cast<VKTexture>(state.oitHeads);
        packet.headsImage = heads->GetImage(); packet.headsView = heads->GetImageView();
        packet.heads = {state.oitHeads.get()}; packet.nodes = state.oitNodes;
        packet.width = heads->GetWidth(); packet.height = heads->GetHeight();
        packet.capacity = static_cast<u32>((state.oitNodes.size - 16ull) / 16ull);
        packet.maxResolveK = std::min(maxK, 16u);
        packet.resolvePipeline = m_ResolvePipeline->GetHandle(); packet.resolveLayout = m_ResolvePipeline->GetLayout();
        packet.resolveSet = state.oitResolveDescSet;
        return packet;
    }
    std::array<RG::ResourceHandle, 3> TransparencySubsystem::AddOitPasses(RG::RenderGraph& graph,
        RG::ResourceHandle color, RG::ResourceHandle picking, RG::ResourceHandle depth,
        const VisibleDrawRange& visible, u32 width, u32 height, const TransparencyBindings& packet,
        std::span<const RG::ResourceHandle> images, std::span<const RG::BufferHandle> buffers,
        bool fogValid, FrameDebugger* debugger)
    {
        // Import once in the previous resolve's state: clear must wait for its reads.
        RG::TextureDesc imageDesc; imageDesc.name = "OITHeads"; imageDesc.width = width; imageDesc.height = height;
        imageDesc.format = RG::TextureFormat::R32_Uint;
        auto heads = graph.ImportResource(imageDesc, (void*)packet.headsImage, (void*)packet.headsView, RG::ResourceState::FragmentStorageRead);
        RG::BufferDesc bufferDesc; bufferDesc.name = "OITNodes"; bufferDesc.size = packet.nodes.size;
        auto nodes = graph.ImportBuffer(bufferDesc, (void*)packet.nodes.buffer, RG::ResourceState::FragmentStorageRead);
        struct Clear { RG::ResourceHandle heads; RG::BufferHandle nodes; };
        graph.AddComputePass<Clear>("OITClear",
            [&](Clear& data, RG::RenderPassBuilder& builder) {
                data.heads = builder.WriteTransfer(heads); data.nodes = builder.WriteBufferTransfer(nodes);
                heads = data.heads; nodes = data.nodes;
            },
            [image = packet.headsImage, allocation = packet.nodes](Clear&, RG::RenderPassContext& ctx) {
                VkClearColorValue value{}; value.uint32[0] = 0xFFFFFFFFu;
                const VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
                vkCmdClearColorImage(ctx.commandBuffer, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &value, 1, &range);
                vkCmdFillBuffer(ctx.commandBuffer, allocation.buffer, allocation.offset, 16, 0u);
            });
        struct Store { RG::ResourceHandle depth, heads; RG::BufferHandle nodes; };
        RG::ResourceHandle storedDepth;
        graph.AddPass<Store>("OITStore",
            [&](Store& data, RG::RenderPassBuilder& builder) {
                auto metadata = RG::RenderPassMetadata::Graphics("pbr_oit_store", true, false, false, VK_CULL_MODE_BACK_BIT, 0); metadata.AddDraws(packet.draws);
                metadata.indirectDraws = true; metadata.pipelineStateMixed = true;
                builder.SetDebugMetadata(std::move(metadata));
                data.depth = builder.WriteDepth(depth, VK_ATTACHMENT_LOAD_OP_LOAD, VK_ATTACHMENT_STORE_OP_STORE, {});
                data.heads = builder.WriteStorageImageFragment(heads); data.nodes = builder.WriteBufferFragment(nodes);
                for (auto image : images) builder.Read(image);
                for (auto buffer : buffers) builder.ReadBufferFragment(buffer);
                builder.ReadIndirectBuffer(visible.indirect.handle);
                heads = data.heads; nodes = data.nodes; storedDepth = data.depth;
            },
            [packet, indirect = visible.indirect.binding.slice->buffer, width, height, fogValid, debugger](Store&, RG::RenderPassContext& ctx) {
                RecordDraws(packet, indirect, width, height, fogValid, packet.capacity, ctx, debugger);
            });
        struct Resolve { RG::ResourceHandle color, picking; };
        std::array<RG::ResourceHandle, 3> outputs;
        graph.AddPass<Resolve>("OITResolve",
            [&](Resolve& data, RG::RenderPassBuilder& builder) {
                builder.SetDebugMetadata(RG::RenderPassMetadata::Graphics("oit_resolve", false, false, true, VK_CULL_MODE_NONE));
                data.color = builder.Write(color, VK_ATTACHMENT_LOAD_OP_LOAD, VK_ATTACHMENT_STORE_OP_STORE);
                data.picking = builder.Write(picking, VK_ATTACHMENT_LOAD_OP_LOAD, VK_ATTACHMENT_STORE_OP_STORE);
                builder.ReadStorageImageFragment(heads); builder.ReadBufferFragment(nodes);
                outputs = {data.color, data.picking, storedDepth};
            },
            [pipeline = packet.resolvePipeline, layout = packet.resolveLayout, set = packet.resolveSet,
                maxK = packet.maxResolveK, width, height, debugger](Resolve&, RG::RenderPassContext& ctx) {
                if (debugger) debugger->BeginCapturePass(ctx.passIndex, "OITResolve", "SceneColor", false,
                    {"oit_resolve", 0, VK_CULL_MODE_NONE, VK_POLYGON_MODE_FILL, false, false, false, true});
                vkCmdBindPipeline(ctx.commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
                vkCmdBindDescriptorSets(ctx.commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, layout, 0, 1, &set, 0, nullptr);
                vkCmdPushConstants(ctx.commandBuffer, layout, VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(maxK), &maxK);
                VkViewport viewport{}; viewport.width = float(width); viewport.height = float(height); viewport.maxDepth = 1;
                const VkRect2D scissor{{0, 0}, {width, height}};
                vkCmdSetViewport(ctx.commandBuffer, 0, 1, &viewport); vkCmdSetScissor(ctx.commandBuffer, 0, 1, &scissor);
                vkCmdDraw(ctx.commandBuffer, 3, 1, 0, 0);
                if (debugger) debugger->EndCapturePass();
            });
        return outputs;
    }
}

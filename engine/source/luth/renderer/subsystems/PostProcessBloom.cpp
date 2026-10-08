#include "luthpch.h"
#include "luth/renderer/subsystems/PostProcessSubsystem.h"
#include "luth/renderer/backend/vulkan/VulkanTexture.h"
#include "luth/renderer/backend/vulkan/VulkanContext.h"
#include "luth/renderer/FrameDebugger.h"

namespace Luth
{
    BloomBindings PostProcessSubsystem::PrepareBloomBindings(const std::shared_ptr<BloomViewState>& state,
        TextureBindingRef source, u64 frame, float threshold, float radius, bool enabled)
    {
        BloomBindings out;
        out.enabled = enabled; out.state = state; out.source = source;
        out.threshold = threshold; out.radius = radius;
        if (!enabled || !state || !m_BloomDownPipeline || !m_BloomUpPipeline ||
            !m_BloomDownPipeline->GetHandle() || !m_BloomUpPipeline->GetHandle()) return out;
        out.downPipeline = m_BloomDownPipeline->GetHandle(); out.upPipeline = m_BloomUpPipeline->GetHandle();
        out.downLayout = m_BloomDownPipeline->GetLayout(); out.upLayout = m_BloomUpPipeline->GetLayout();
        out.prefilterSet = state->prefilterSets[frame % MAX_FRAMES_IN_FLIGHT];
        out.downSets = state->downSets; out.upSets = state->upSets;
        out.width = state->width; out.height = state->height;
        for (u32 i = 0; i < BloomViewState::kMipCount; ++i)
        {
            if (!state->mips[i]) throw std::invalid_argument("Bloom: missing prepared mip");
            const auto texture = std::static_pointer_cast<VKTexture>(state->mips[i]);
            out.mips[i] = {texture.get()}; out.images[i] = texture->GetImage(); out.views[i] = texture->GetImageView();
        }
        if (!source.texture) throw std::invalid_argument("Bloom: missing sampled source");
        VkDescriptorImageInfo image{m_Sampler, static_cast<const VKTexture*>(source.texture)->GetImageView(),
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
        VkWriteDescriptorSet write{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        write.dstSet = out.prefilterSet; write.dstBinding = 0; write.descriptorCount = 1;
        write.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; write.pImageInfo = &image;
        vkUpdateDescriptorSets(VulkanContext::Get().GetDevice(), 1, &write, 0, nullptr);
        return out;
    }
    namespace
    {
        void RecordBloom(RG::RenderPassContext& ctx, VkPipeline pipeline, VkPipelineLayout layout,
            VkDescriptorSet set, const void* constants, u32 size, u32 width, u32 height,
            const char* label, const char* shader, FrameDebugger* debugger)
        {
            if (debugger) debugger->BeginCapturePass(ctx.passIndex, label, "BloomMip", false,
                {shader, 0, 0, VK_POLYGON_MODE_FILL, false, false, false, false});
            const auto cmd = ctx.commandBuffer;
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0, 1, &set, 0, nullptr);
            vkCmdPushConstants(cmd, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, size, constants);
            const u32 gx = (width + 7) / 8, gy = (height + 7) / 8;
            vkCmdDispatch(cmd, gx, gy, 1);
            if (debugger) debugger->CaptureComputeDispatch(label, shader, gx, gy, 1);
            if (debugger) debugger->EndCapturePass();
        }
    }
    RG::ResourceHandle PostProcessSubsystem::AddBloomPasses(RG::RenderGraph& graph, RG::ResourceHandle source,
        const BloomBindings& packet, FrameDebugger* debugger)
    {
        constexpr u32 N = BloomViewState::kMipCount;
        std::array<RG::ResourceHandle, N> handles{};
        const auto config = BloomViewState::Config(packet.width, packet.height);
        const auto extent = [&](u32 mip) { return BloomViewState::MipExtent(config, mip); };
        const auto import = [&](u32 mip, RG::RenderPassBuilder& builder) {
            const auto [width, height] = extent(mip);
            RG::TextureDesc desc; desc.name = "BloomMip" + std::to_string(mip); desc.width = width; desc.height = height;
            desc.format = RG::TextureFormat::RGBA16_Float;
            return builder.WriteStorageImage(graph.ImportResource(desc, (void*)packet.images[mip],
                (void*)packet.views[mip], RG::ResourceState::Undefined));
        };
        struct Data {};
        const auto addDown = [&](u32 mip, RG::ResourceHandle input, const char* label, bool prefilter) {
            const auto [srcW, srcH] = prefilter ? std::pair{packet.width, packet.height} : extent(mip - 1);
            const auto [width, height] = extent(mip);
            BloomDownPC pc{}; pc.srcTexel = {1.0f / srcW, 1.0f / srcH}; pc.dstSize = {(i32)width, (i32)height};
            if (prefilter) { pc.threshold = packet.threshold; pc.knee = 0.5f; pc.prefilter = 1; }
            const auto set = prefilter ? packet.prefilterSet : packet.downSets[mip - 1];
            graph.AddComputePass<Data>(prefilter ? "BloomPrefilter" : "BloomDown" + std::to_string(mip - 1),
                [&](Data&, RG::RenderPassBuilder& builder) {
                    builder.ReadStorageImage(input); handles[mip] = import(mip, builder);
                },
                [packet, pc, set, width, height, label, debugger](Data&, RG::RenderPassContext& ctx) {
                    RecordBloom(ctx, packet.downPipeline, packet.downLayout, set, &pc, sizeof(pc), width, height,
                        label, "bloom_downsample", debugger);
                });
        };
        addDown(0, source, "BloomPrefilter", true);
        for (u32 i = 1; i < N; ++i) addDown(i, handles[i - 1], "BloomDown", false);
        for (i32 i = N - 2; i >= 0; --i)
        {
            const auto [srcW, srcH] = extent(i + 1);
            const auto [width, height] = extent(i);
            BloomUpPC pc{}; pc.srcTexel = {1.0f / srcW, 1.0f / srcH}; pc.dstSize = {(i32)width, (i32)height}; pc.radius = packet.radius;
            graph.AddComputePass<Data>("BloomUp" + std::to_string(i),
                [&](Data&, RG::RenderPassBuilder& builder) {
                    builder.ReadStorageImage(handles[i + 1]);
                    // Preserve additive imageLoad visibility and the single imported node per mip.
                    handles[i] = builder.ReadStorageImageGeneral(handles[i]);
                    handles[i] = builder.WriteStorageImage(handles[i]);
                },
                [packet, pc, i, width, height, debugger](Data&, RG::RenderPassContext& ctx) {
                    RecordBloom(ctx, packet.upPipeline, packet.upLayout, packet.upSets[i], &pc, sizeof(pc), width, height,
                        "BloomUp", "bloom_upsample", debugger);
                });
        }
        return handles[0];
    }
}

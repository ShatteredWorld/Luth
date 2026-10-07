#include "luthpch.h"
#include "luth/renderer/subsystems/TransparencySubsystem.h"
#include "luth/renderer/backend/vulkan/VulkanTexture.h"

namespace Luth
{
    RefractionBackdropBindings TransparencySubsystem::PrepareBackdropBindings(
        const std::shared_ptr<Texture>& texture, bool enabled)
    {
        RefractionBackdropBindings out;
        out.enabled = enabled;
        if (!enabled || !texture) return out;
        const auto native = std::static_pointer_cast<VKTexture>(texture);
        out.image = native->GetImage(); out.view = native->GetImageView();
        out.width = native->GetWidth(); out.height = native->GetHeight();
        out.binding = {texture.get()};
        return out;
    }
    GraphTextureRef TransparencySubsystem::AddBackdropCopyPass(RG::RenderGraph& graph,
        RG::ResourceHandle source, const RefractionBackdropBindings& packet)
    {
        RG::TextureDesc desc;
        desc.name = "RefractionBackdrop"; desc.width = packet.width; desc.height = packet.height;
        desc.format = RG::TextureFormat::RGBA16_Float;
        const auto destination = graph.ImportResource(desc, (void*)packet.image, (void*)packet.view,
            RG::ResourceState::ShaderResource);
        struct CopyData { RG::ResourceHandle src, dst; };
        RG::ResourceHandle output;
        graph.AddComputePass<CopyData>("RefractionBackdropCopy",
            [&](CopyData& data, RG::RenderPassBuilder& builder) {
                data.src = builder.ReadTransfer(source); data.dst = builder.WriteTransfer(destination);
                output = data.dst;
            },
            [](CopyData& data, RG::RenderPassContext& ctx) {
                auto* src = (RG::RenderGraph::ResourceNode*)ctx.GetResource(data.src);
                auto* dst = (RG::RenderGraph::ResourceNode*)ctx.GetResource(data.dst);
                VkImageCopy region{};
                region.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
                region.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
                region.extent = {src->desc.width, src->desc.height, 1};
                vkCmdCopyImage(ctx.commandBuffer, src->image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                    dst->image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
            });
        return {output, packet.binding};
    }
}

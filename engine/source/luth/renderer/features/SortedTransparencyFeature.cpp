#include "luthpch.h"
#include "luth/renderer/features/SortedTransparencyFeature.h"
#include "luth/renderer/features/SortedTransparencyBindings.h"
#include "luth/renderer/subsystems/TransparencySubsystem.h"

namespace Luth
{
    FeatureInfo SortedTransparencyFeature::Describe() const
    {
        FeatureInfo info;
        info.name = "SortedTransparency"; info.phase = FeaturePhase::AfterAsync;
        info.resources.reads = {{TransparencyResources::Bindings}, {RenderResources::FoggedHDR},
            {RenderResources::LitDepth}, {RenderResources::OpaquePickingIDs}, {RenderResources::CameraVisibleDraws},
            {RenderResources::ResolvedFog, ResourceReadRequirement::Optional},
            {RenderResources::RefractionBackdrop, ResourceReadRequirement::Optional},
            {RenderResources::LightData, ResourceReadRequirement::Optional},
            {RenderResources::ClusterGrid, ResourceReadRequirement::Optional},
            {RenderResources::LightIndices, ResourceReadRequirement::Optional}};
        info.resources.writes = {{RenderResources::TransparentHDR, ResourceOutputPresence::Required, ResourceKeyRef{RenderResources::FoggedHDR}, true},
            {RenderResources::FinalPickingIDs, ResourceOutputPresence::Required, ResourceKeyRef{RenderResources::OpaquePickingIDs}, true},
            {TransparencyResources::Depth, ResourceOutputPresence::Required, ResourceKeyRef{RenderResources::LitDepth}, true}};
        info.capabilities.consumes = {{&DeformationResources::DeformedGeometry}};
        return info;
    }
    void SortedTransparencyFeature::Build(RG::RenderGraph& graph, RenderFeatureContext& ctx)
    {
        const auto& color = ctx.resources.Get(RenderResources::FoggedHDR);
        const auto& depth = ctx.resources.Get(RenderResources::LitDepth);
        const auto& picking = ctx.resources.Get(RenderResources::OpaquePickingIDs);
        const auto& visible = ctx.resources.Get(RenderResources::CameraVisibleDraws);
        const auto* packet = ctx.resources.Get(TransparencyResources::Bindings).native;
        const auto valid = [&](const GraphTextureRef& value, RG::TextureFormat format) {
            if (!value.handle.IsValid() || value.handle.index > graph.GetResources().size() || !value.binding.texture ||
                value.binding.baseMip || value.binding.mipCount != 1 || value.binding.baseLayer || value.binding.layerCount != 1) return false;
            const auto& desc = graph.GetResources()[value.handle.index - 1].desc;
            return desc.width == ctx.view.width && desc.height == ctx.view.height && desc.format == format;
        };
        if (!packet || !ctx.view.width || !ctx.view.height || !valid(color, RG::TextureFormat::RGBA16_Float) ||
            !valid(depth, RG::TextureFormat::D32_Float) || !valid(picking, RG::TextureFormat::R32_Uint) ||
            !visible.indirect.handle.IsValid() || visible.indirect.handle.index > graph.GetBuffers().size() ||
            !visible.indirect.binding.slice || !visible.indirect.binding.slice->buffer ||
            visible.indirect.binding.offset != visible.indirect.binding.slice->offset ||
            visible.indirect.binding.size > visible.indirect.binding.slice->size ||
            (u64(visible.firstDraw) + visible.maxDrawCount) * sizeof(VkDrawIndexedIndirectCommand) > visible.indirect.binding.size)
            throw std::invalid_argument("SortedTransparency: invalid stage inputs or visible slice");
        for (const auto& draw : packet->draws)
            if (!draw.pipeline || !draw.layout || !draw.index || (!draw.deformed && !draw.vertex) ||
                draw.indirectOffset != GeometrySubsystem::ForwardDrawOffset(visible, draw.objectIndex))
                throw std::invalid_argument("SortedTransparency: invalid prepared draw");
        if (!packet->draws.empty() && std::any_of(packet->sets.begin(), packet->sets.end(), [](auto set) { return !set; }))
            throw std::invalid_argument("SortedTransparency: incomplete descriptor bindings");
        std::vector<RG::ResourceHandle> images;
        RG::ResourceHandle fog;
        for (const auto key : {RenderResources::ResolvedFog, RenderResources::RefractionBackdrop})
            if (const auto* image = ctx.resources.TryGet(key))
            {
                const auto expected = key.identity == RenderResources::ResolvedFog.identity ? packet->fog : packet->backdrop;
                if (!image->handle.IsValid() || image->handle.index > graph.GetResources().size() ||
                    !image->binding.texture || image->binding.texture != expected.texture || image->binding.baseMip ||
                    image->binding.mipCount != 1 || image->binding.baseLayer || image->binding.layerCount != 1 ||
                    graph.GetResources()[image->handle.index - 1].desc.format != RG::TextureFormat::RGBA16_Float)
                    throw std::invalid_argument("SortedTransparency: sampled binding mismatch");
                images.push_back(image->handle);
                if (key.identity == RenderResources::ResolvedFog.identity) fog = image->handle;
            }
        std::vector<RG::BufferHandle> buffers;
        for (auto key : {RenderResources::LightData, RenderResources::ClusterGrid, RenderResources::LightIndices})
            if (const auto* value = ctx.resources.TryGet(key))
            {
                if (!value->handle.IsValid() || value->handle.index > graph.GetBuffers().size() || !value->binding.slice ||
                    !value->binding.slice->buffer || value->binding.offset != value->binding.slice->offset ||
                    value->binding.size > value->binding.slice->size)
                    throw std::invalid_argument("SortedTransparency: invalid lighting slice");
                buffers.push_back(value->handle);
            }
        auto outputs = std::array{color.handle, picking.handle, depth.handle};
        if (!packet->draws.empty())
            outputs = m_Native.AddSortedPass(graph, color.handle, picking.handle, depth.handle, visible,
                ctx.view.width, ctx.view.height, *packet, images, buffers, fog.IsValid(), m_Debugger);
        ctx.resources.Publish(RenderResources::TransparentHDR, GraphTextureRef{outputs[0], color.binding});
        ctx.resources.Publish(RenderResources::FinalPickingIDs, GraphTextureRef{outputs[1], picking.binding});
        ctx.resources.Publish(TransparencyResources::Depth, GraphTextureRef{outputs[2], depth.binding});
    }
}

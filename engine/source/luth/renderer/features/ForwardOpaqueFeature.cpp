#include "luthpch.h"
#include "luth/renderer/features/ForwardOpaqueFeature.h"
#include "luth/renderer/features/ForwardOpaqueCompatibility.h"
#include "luth/renderer/subsystems/GeometrySubsystem.h"

namespace Luth
{
    FeatureInfo ForwardOpaqueFeature::Describe() const
    {
        FeatureInfo info;
        info.name = "ForwardOpaque"; info.phase = FeaturePhase::AfterAsync;
        info.resources.reads = {{RenderResources::CameraVisibleDraws}, {RenderResources::SurfaceDepth},
            {ForwardOpaqueResources::ColorTarget}, {ForwardOpaqueResources::PickingTarget}, {ForwardOpaqueResources::Bindings},
            {RenderResources::ShadowCascades, ResourceReadRequirement::Optional},
            {RenderResources::AmbientOcclusion, ResourceReadRequirement::Optional},
            {RenderResources::LightData, ResourceReadRequirement::Optional},
            {RenderResources::ClusterGrid, ResourceReadRequirement::Optional},
            {RenderResources::LightIndices, ResourceReadRequirement::Optional}};
        info.resources.writes = {{RenderResources::OpaqueHDR, ResourceOutputPresence::Required,
                ResourceKeyRef{ForwardOpaqueResources::ColorTarget}, true},
            {RenderResources::LitDepth, ResourceOutputPresence::Required, ResourceKeyRef{RenderResources::SurfaceDepth}, true},
            {RenderResources::OpaquePickingIDs, ResourceOutputPresence::Required, ResourceKeyRef{ForwardOpaqueResources::PickingTarget}, true}};
        info.capabilities.consumes = {{&DeformationResources::DeformedGeometry}};
        return info;
    }
    void ForwardOpaqueFeature::Build(RG::RenderGraph& graph, RenderFeatureContext& ctx)
    {
        const auto& color = ctx.resources.Get(ForwardOpaqueResources::ColorTarget);
        const auto& picking = ctx.resources.Get(ForwardOpaqueResources::PickingTarget);
        const auto& depth = ctx.resources.Get(RenderResources::SurfaceDepth);
        const auto& visible = ctx.resources.Get(RenderResources::CameraVisibleDraws);
        const auto* native = ctx.resources.Get(ForwardOpaqueResources::Bindings).native;
        const auto validTarget = [&](const GraphTextureRef& target, RG::TextureFormat format) {
            if (!target.handle.IsValid() || target.handle.index > graph.GetResources().size() || !target.binding.texture ||
                target.binding.baseMip || target.binding.mipCount != 1 || target.binding.baseLayer || target.binding.layerCount != 1)
                return false;
            const auto& desc = graph.GetResources()[target.handle.index - 1].desc;
            return desc.width == ctx.view.width && desc.height == ctx.view.height && desc.format == format;
        };
        if (!native || !ctx.view.width || !ctx.view.height ||
            !validTarget(color, RG::TextureFormat::RGBA16_Float) || !validTarget(depth, RG::TextureFormat::D32_Float) ||
            !validTarget(picking, RG::TextureFormat::R32_Uint) ||
            !visible.indirect.handle.IsValid() || visible.indirect.handle.index > graph.GetBuffers().size() ||
            !visible.indirect.binding.slice || !visible.indirect.binding.slice->buffer ||
            visible.indirect.binding.offset != visible.indirect.binding.slice->offset ||
            visible.indirect.binding.size > visible.indirect.binding.slice->size ||
            (u64(visible.firstDraw) + visible.maxDrawCount) * sizeof(VkDrawIndexedIndirectCommand) > visible.indirect.binding.size ||
            (native->initialLayout && std::any_of(native->sets.begin(), native->sets.end(), [](auto set) { return !set; })))
            throw std::invalid_argument("ForwardOpaque: invalid stage targets, native packet or visible draw range");
        for (const auto& draw : native->draws)
            if (!draw.pipeline || !draw.layout || !draw.index || (!draw.deformed && !draw.vertex) ||
                draw.indirectOffset != m_Native.ForwardDrawOffset(visible, draw.objectIndex))
                throw std::invalid_argument("ForwardOpaque: draw packet does not match visible slice");
        for (const auto& draw : native->overlays)
            if (!draw.pipeline || !draw.layout || !draw.index || (!draw.deformed && !draw.vertex) ||
                draw.objectIndex >= visible.maxDrawCount)
                throw std::invalid_argument("ForwardOpaque: invalid direct overlay packet");
        if (!native->initialLayout && (!native->draws.empty() || !native->overlays.empty()))
            throw std::invalid_argument("ForwardOpaque: draw packets require a compatible descriptor layout");
        std::vector<RG::ResourceHandle> images;
        if (const auto* shadows = ctx.resources.TryGet(RenderResources::ShadowCascades))
            for (const auto& cascade : shadows->cascades) images.push_back(cascade.handle);
        if (const auto* ao = ctx.resources.TryGet(RenderResources::AmbientOcclusion)) images.push_back(ao->handle);
        AppendCompatibilityReads(ctx, images);
        for (auto image : images)
            if (!image.IsValid() || image.index > graph.GetResources().size())
                throw std::invalid_argument("ForwardOpaque: sampled image is outside the current graph");
        std::vector<RG::BufferHandle> buffers;
        for (auto key : {RenderResources::LightData, RenderResources::ClusterGrid, RenderResources::LightIndices})
            if (const auto* buffer = ctx.resources.TryGet(key))
            {
                if (!buffer->handle.IsValid() || buffer->handle.index > graph.GetBuffers().size() || !buffer->binding.slice ||
                    !buffer->binding.slice->buffer || buffer->binding.offset != buffer->binding.slice->offset ||
                    buffer->binding.size > buffer->binding.slice->size)
                    throw std::invalid_argument("ForwardOpaque: invalid native lighting slice");
                buffers.push_back(buffer->handle);
            }
        const auto output = m_Native.AddForwardOpaquePass(graph, color.handle, depth.handle, picking.handle,
            visible, ctx.view.width, ctx.view.height, *native, images, buffers, m_Debugger);
        ctx.resources.Publish(RenderResources::OpaqueHDR, GraphTextureRef{output[0], color.binding});
        ctx.resources.Publish(RenderResources::LitDepth, GraphTextureRef{output[1], depth.binding});
        ctx.resources.Publish(RenderResources::OpaquePickingIDs, GraphTextureRef{output[2], picking.binding});
    }
    FeatureInfo HybridForwardOpaqueFeature::Describe() const
    {
        auto info = ForwardOpaqueFeature::Describe();
        info.name = "HybridForwardOpaque";
        for (auto key : {ForwardCompatibilityResources::SunShadowMask, ForwardCompatibilityResources::DenoisedDiffuseDI,
            ForwardCompatibilityResources::DenoisedDiffuseGI, ForwardCompatibilityResources::DenoisedReflectionRadiance,
            ForwardCompatibilityResources::DenoisedSpecularDI})
            info.resources.reads.push_back({key, ResourceReadRequirement::Optional});
        return info;
    }
    void HybridForwardOpaqueFeature::AppendCompatibilityReads(RenderFeatureContext& ctx,
        std::vector<RG::ResourceHandle>& images) const
    {
        for (auto key : {ForwardCompatibilityResources::SunShadowMask, ForwardCompatibilityResources::DenoisedDiffuseDI,
            ForwardCompatibilityResources::DenoisedDiffuseGI, ForwardCompatibilityResources::DenoisedReflectionRadiance,
            ForwardCompatibilityResources::DenoisedSpecularDI})
            if (const auto* image = ctx.resources.TryGet(key)) images.push_back(image->handle);
    }
}

#include "luthpch.h"
#include "luth/renderer/features/FogCompositeFeature.h"
#include "luth/renderer/subsystems/VolumetricSubsystem.h"

namespace Luth
{
    FeatureInfo FogCompositeFeature::Describe() const
    {
        FeatureInfo info;
        info.name = "FogComposite"; info.phase = FeaturePhase::AfterAsync;
        info.resources.reads = {{RenderResources::SkyHDR}, {RenderResources::SurfaceDepth},
            {FogCompositeResources::Bindings}, {RenderResources::ResolvedFog, ResourceReadRequirement::Optional}};
        info.resources.writes = {{RenderResources::FoggedHDR, ResourceOutputPresence::Required,
            ResourceKeyRef{RenderResources::SkyHDR}, true}};
        return info;
    }
    void FogCompositeFeature::Build(RG::RenderGraph& graph, RenderFeatureContext& ctx)
    {
        const auto& color = ctx.resources.Get(RenderResources::SkyHDR);
        const auto& depth = ctx.resources.Get(RenderResources::SurfaceDepth);
        const auto* fog = ctx.resources.TryGet(RenderResources::ResolvedFog);
        const auto* packet = ctx.resources.Get(FogCompositeResources::Bindings).native;
        const auto valid = [&](const GraphTextureRef& texture) {
            return texture.handle.IsValid() && texture.handle.index <= graph.GetResources().size() &&
                texture.binding.texture && !texture.binding.baseMip && texture.binding.mipCount == 1 &&
                !texture.binding.baseLayer && texture.binding.layerCount == 1;
        };
        if (!packet || !ctx.view.width || !ctx.view.height || !valid(color) || !valid(depth))
            throw std::invalid_argument("FogComposite: missing stage inputs or native packet");
        const auto& colorDesc = graph.GetResources()[color.handle.index - 1].desc;
        const auto& depthDesc = graph.GetResources()[depth.handle.index - 1].desc;
        if (colorDesc.format != RG::TextureFormat::RGBA16_Float || depthDesc.format != RG::TextureFormat::D32_Float ||
            colorDesc.width != ctx.view.width || colorDesc.height != ctx.view.height ||
            depthDesc.width != ctx.view.width || depthDesc.height != ctx.view.height)
            throw std::invalid_argument("FogComposite: incompatible stage formats or extent");
        GraphTextureRef output = color;
        if (packet->enabled && fog && packet->pipeline)
        {
            if (!valid(*fog) || !packet->layout || !packet->sets[0] || !packet->sets[1] ||
                depth.binding.texture != packet->depth.texture || fog->binding.texture != packet->resolved.texture ||
                graph.GetResources()[fog->handle.index - 1].desc.format != RG::TextureFormat::RGBA16_Float ||
                fog->handle.index == color.handle.index || fog->handle.index == depth.handle.index)
                throw std::invalid_argument("FogComposite: incomplete or mismatched sampled bindings");
            output.handle = m_Native.AddCompositePass(graph, color.handle, depth.handle, fog->handle,
                ctx.view.width, ctx.view.height, *packet, m_Debugger);
        }
        ctx.resources.Publish(RenderResources::FoggedHDR, output);
    }
}

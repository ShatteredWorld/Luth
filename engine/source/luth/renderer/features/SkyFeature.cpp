#include "luthpch.h"
#include "luth/renderer/features/SkyFeature.h"
#include "luth/renderer/subsystems/LightingSubsystem.h"

namespace Luth
{
    FeatureInfo SkyFeature::Describe() const
    {
        FeatureInfo info;
        info.name = "Sky";
        info.phase = FeaturePhase::AfterAsync;
        info.resources.reads = {{RenderResources::OpaqueHDR}, {RenderResources::LitDepth}, {SkyResources::Bindings}};
        info.resources.writes = {{RenderResources::SkyHDR, ResourceOutputPresence::Required,
            ResourceKeyRef{RenderResources::OpaqueHDR}, true}};
        return info;
    }
    void SkyFeature::Build(RG::RenderGraph& graph, RenderFeatureContext& ctx)
    {
        const auto& color = ctx.resources.Get(RenderResources::OpaqueHDR);
        const auto& depth = ctx.resources.Get(RenderResources::LitDepth);
        const auto* packet = ctx.resources.Get(SkyResources::Bindings).native;
        const auto valid = [&](const GraphTextureRef& texture, RG::TextureFormat format) {
            if (!texture.handle.IsValid() || texture.handle.index > graph.GetResources().size() ||
                !texture.binding.texture || texture.binding.baseMip || texture.binding.mipCount != 1 ||
                texture.binding.baseLayer || texture.binding.layerCount != 1) return false;
            const auto& desc = graph.GetResources()[texture.handle.index - 1].desc;
            return desc.width == ctx.view.width && desc.height == ctx.view.height && desc.format == format;
        };
        if (!packet || !ctx.view.width || !ctx.view.height ||
            !valid(color, RG::TextureFormat::RGBA16_Float) || !valid(depth, RG::TextureFormat::D32_Float) ||
            (packet->pipeline && packet->vertex && (!packet->layout ||
                std::any_of(packet->sets.begin(), packet->sets.end(), [](auto set) { return !set; }))))
            throw std::invalid_argument("Sky: invalid stage inputs, extent or native bindings");
        const auto output = m_Native.AddSkyboxPass(graph, color.handle, depth.handle,
            ctx.view.width, ctx.view.height, *packet, m_Debugger);
        ctx.resources.Publish(RenderResources::SkyHDR, GraphTextureRef{output, color.binding});
    }
}

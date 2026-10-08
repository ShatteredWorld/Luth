#include "luthpch.h"
#include "luth/renderer/features/FogVizFeature.h"
#include "luth/renderer/subsystems/VolumetricSubsystem.h"
#include <cmath>
namespace Luth
{
    FeatureInfo FogVizFeature::Describe() const
    {
        FeatureInfo info; info.name = "FogVisualization"; info.phase = FeaturePhase::AfterAsync;
        info.resources.reads = {{ClusterVizResources::Output}, {FogVizResources::Bindings},
            {RenderResources::SurfaceDepth, ResourceReadRequirement::Optional},
            {RenderResources::FogDensity, ResourceReadRequirement::Optional},
            {RenderResources::ResolvedFog, ResourceReadRequirement::Optional}};
        info.resources.writes = {{RenderResources::VisualizedLDR, ResourceOutputPresence::Required,
            ResourceKeyRef{ClusterVizResources::Output}, true}};
        return info;
    }
    void FogVizFeature::Build(RG::RenderGraph& graph, RenderFeatureContext& ctx)
    {
        const auto& input = ctx.resources.Get(ClusterVizResources::Output);
        const auto* packet = ctx.resources.Get(FogVizResources::Bindings).native;
        const auto valid = [&](const GraphTextureRef& ref, RG::TextureFormat format) {
            return ref.handle.IsValid() && ref.handle.index <= graph.GetResources().size() && ref.binding.texture &&
                !ref.binding.baseMip && ref.binding.mipCount == 1 && !ref.binding.baseLayer && ref.binding.layerCount == 1 &&
                graph.GetResources()[ref.handle.index - 1].desc.format == format;
        };
        if (!packet || !ctx.view.width || !ctx.view.height || !valid(input, RG::TextureFormat::RGBA8_Unorm))
            throw std::invalid_argument("FogViz: missing packet or incompatible LDR input");
        const auto& colorDesc = graph.GetResources()[input.handle.index - 1].desc;
        if (colorDesc.width != ctx.view.width || colorDesc.height != ctx.view.height)
            throw std::invalid_argument("FogViz: incompatible LDR extent");
        GraphTextureRef output = input;
        const auto* resolved = ctx.resources.TryGet(RenderResources::ResolvedFog);
        if (packet->enabled && packet->pipeline && resolved)
        {
            const auto* depth = ctx.resources.TryGet(RenderResources::SurfaceDepth);
            const auto* density = ctx.resources.TryGet(RenderResources::FogDensity);
            const auto& state = packet->state;
            if (!state || !packet->layout || !packet->sets[0] || !packet->sets[1] ||
                packet->renderFrameIndex != ctx.frame.renderFrameIndex ||
                packet->sets[1] != state->volVizDescSet[packet->renderFrameIndex % MAX_FRAMES_IN_FLIGHT] ||
                !depth || !valid(*depth, RG::TextureFormat::D32_Float) ||
                !density || !valid(*density, RG::TextureFormat::RGBA16_Float) || !valid(*resolved, RG::TextureFormat::RGBA16_Float) ||
                depth->binding.texture != state->depthSource.get() || density->binding.texture != state->volDensity.get() ||
                resolved->binding.texture != ((packet->renderFrameIndex & 1u) ? state->volInScatterHistA.get() : state->volInScatterHistB.get()) ||
                packet->parameters.mode > 1 || !std::isfinite(packet->parameters.scale) || !std::isfinite(packet->parameters.overlayAlpha))
                throw std::invalid_argument("FogViz: incomplete or mismatched frozen sampled bindings");
            const auto& depthDesc = graph.GetResources()[depth->handle.index - 1].desc;
            for (const auto* source : {depth, density, resolved})
                if (source->handle.index == input.handle.index || source->binding.texture == input.binding.texture)
                    throw std::invalid_argument("FogViz: color sampling feedback");
            const auto& densityDesc = graph.GetResources()[density->handle.index - 1].desc;
            const auto& resolvedDesc = graph.GetResources()[resolved->handle.index - 1].desc;
            if (depthDesc.width != ctx.view.width || depthDesc.height != ctx.view.height ||
                !state->volDimX || !state->volDimY || !state->volDimZ ||
                densityDesc.width != state->volDimX || densityDesc.height != state->volDimY ||
                resolvedDesc.width != state->volDimX || resolvedDesc.height != state->volDimY)
                throw std::invalid_argument("FogViz: incompatible depth or atlas extent");
            output.handle = m_Native.AddVizPass(graph, input.handle, density->handle, resolved->handle, depth->handle, *packet, m_Debugger);
        }
        ctx.resources.Publish(RenderResources::VisualizedLDR, output);
    }
}

#include "luthpch.h"
#include "luth/renderer/features/rt/GiReservoirVizFeature.h"
#include "luth/renderer/subsystems/RtRestirGiSubsystem.h"
#include <cmath>
namespace Luth
{
    FeatureInfo GiReservoirVizFeature::Describe() const
    {
        FeatureInfo info; info.name = "GiReservoirVisualization"; info.phase = FeaturePhase::AfterAsync;
        info.resources.reads = {{FogVizResources::Output}, {GiReservoirVizResources::Bindings},
            {RenderResources::SurfaceDepth, ResourceReadRequirement::Optional},
            {GiReservoirVizResources::SpatialReservoir, ResourceReadRequirement::Optional}};
        info.resources.writes = {{RenderResources::VisualizedLDR, ResourceOutputPresence::Required,
            ResourceKeyRef{FogVizResources::Output}, true}};
        return info;
    }
    void GiReservoirVizFeature::Build(RG::RenderGraph& graph, RenderFeatureContext& ctx)
    {
        const auto& input = ctx.resources.Get(FogVizResources::Output);
        const auto* packet = ctx.resources.Get(GiReservoirVizResources::Bindings).native;
        const auto valid = [&](const GraphTextureRef& ref, RG::TextureFormat format) {
            if (!ref.handle.IsValid() || ref.handle.index > graph.GetResources().size() || !ref.binding.texture ||
                ref.binding.baseMip || ref.binding.mipCount != 1 || ref.binding.baseLayer || ref.binding.layerCount != 1) return false;
            const auto& desc = graph.GetResources()[ref.handle.index - 1].desc;
            return desc.format == format && desc.width == ctx.view.width && desc.height == ctx.view.height;
        };
        if (!packet || !ctx.view.width || !ctx.view.height || !valid(input, RG::TextureFormat::RGBA8_Unorm))
            throw std::invalid_argument("GiReservoirViz: missing packet or incompatible LDR");
        GraphTextureRef output = input;
        const auto* reservoir = ctx.resources.TryGet(GiReservoirVizResources::SpatialReservoir);
        if (packet->enabled && packet->pipeline && reservoir)
        {
            const auto* depth = ctx.resources.TryGet(RenderResources::SurfaceDepth); const auto& pc = packet->parameters;
            if (!depth || !valid(*depth, RG::TextureFormat::D32_Float) || !packet->layout || !packet->set || !packet->depth ||
                packet->depth.get() != depth->binding.texture || depth->handle.index == input.handle.index ||
                depth->binding.texture == input.binding.texture || !reservoir->handle.IsValid() ||
                reservoir->handle.index > graph.GetBuffers().size() || !reservoir->binding.slice ||
                !packet->reservoir.buffer || packet->reservoir.buffer != reservoir->binding.slice->buffer ||
                graph.GetBuffers()[reservoir->handle.index - 1].buffer != packet->reservoir.buffer ||
                packet->reservoir.offset != reservoir->binding.offset || packet->reservoir.size != reservoir->binding.size ||
                reservoir->binding.offset != reservoir->binding.slice->offset || reservoir->binding.size > reservoir->binding.slice->size ||
                graph.GetBuffers()[reservoir->handle.index - 1].desc.size < reservoir->binding.size ||
                pc.vx != float(ctx.view.width) || pc.vy != float(ctx.view.height) ||
                !std::isfinite(pc.resW) || !std::isfinite(pc.resH) || pc.resW <= 0 || pc.resH <= 0 ||
                pc.resW > pc.vx || pc.resH > pc.vy || std::floor(pc.resW) != pc.resW || std::floor(pc.resH) != pc.resH ||
                !std::isfinite(pc.mCap) || !std::isfinite(pc.ageCap) || pc.mCap <= 0 || pc.ageCap < 0 ||
                double(pc.resW) * double(pc.resH) * 64 > double(reservoir->binding.size))
                throw std::invalid_argument("GiReservoirViz: mismatched depth/reservoir bindings or frozen parameters");
            output.handle = m_Native.AddReservoirVizPass(graph, input.handle, depth->handle, reservoir->handle, *packet);
        }
        ctx.resources.Publish(RenderResources::VisualizedLDR, output);
    }
}

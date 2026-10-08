#include "luthpch.h"
#include "luth/renderer/features/ClusterVizFeature.h"
#include "luth/renderer/subsystems/LightingSubsystem.h"
#include <cmath>
namespace Luth
{
    FeatureInfo ClusterVizFeature::Describe() const
    {
        FeatureInfo info; info.name = "ClusterVisualization"; info.phase = FeaturePhase::AfterAsync;
        info.resources.reads = {{SlimVizResources::Output}, {ClusterVizResources::Bindings},
            {RenderResources::SurfaceDepth, ResourceReadRequirement::Optional},
            {RenderResources::ClusterGrid, ResourceReadRequirement::Optional}};
        info.resources.writes = {{RenderResources::VisualizedLDR, ResourceOutputPresence::Required,
            ResourceKeyRef{SlimVizResources::Output}, true}};
        return info;
    }
    void ClusterVizFeature::Build(RG::RenderGraph& graph, RenderFeatureContext& ctx)
    {
        const auto& input = ctx.resources.Get(SlimVizResources::Output);
        const auto* packet = ctx.resources.Get(ClusterVizResources::Bindings).native;
        const auto valid = [&](const GraphTextureRef& ref, RG::TextureFormat format) {
            if (!ref.handle.IsValid() || ref.handle.index > graph.GetResources().size() || !ref.binding.texture ||
                ref.binding.baseMip || ref.binding.mipCount != 1 || ref.binding.baseLayer || ref.binding.layerCount != 1) return false;
            const auto& desc = graph.GetResources()[ref.handle.index - 1].desc;
            return desc.format == format && desc.width == ctx.view.width && desc.height == ctx.view.height;
        };
        if (!packet || !ctx.view.width || !ctx.view.height || !valid(input, RG::TextureFormat::RGBA8_Unorm))
            throw std::invalid_argument("ClusterViz: missing packet or incompatible LDR stage");
        GraphTextureRef output = input;
        const auto* grid = ctx.resources.TryGet(RenderResources::ClusterGrid);
        if (packet->enabled && packet->pipeline && grid)
        {
            const auto* depth = ctx.resources.TryGet(RenderResources::SurfaceDepth);
            if (!depth || !valid(*depth, RG::TextureFormat::D32_Float) || !packet->layout || !packet->sets[0] || !packet->sets[1] ||
                !packet->state || packet->state->set != packet->sets[0] ||
                !packet->state->depth || packet->state->depth.get() != depth->binding.texture ||
                input.handle.index == depth->handle.index || input.binding.texture == depth->binding.texture ||
                !grid->handle.IsValid() || grid->handle.index > graph.GetBuffers().size() || !grid->binding.slice ||
                !packet->grid.buffer || packet->grid.buffer != grid->binding.slice->buffer ||
                graph.GetBuffers()[grid->handle.index - 1].buffer != packet->grid.buffer ||
                graph.GetBuffers()[grid->handle.index - 1].desc.size < grid->binding.size ||
                packet->grid.offset != grid->binding.offset || packet->grid.range != grid->binding.size ||
                grid->binding.offset != grid->binding.slice->offset || grid->binding.size > grid->binding.slice->size ||
                grid->binding.size < u64(k_ClusterCount) * sizeof(GPUCluster) ||
                packet->parameters.viewport != Vec2(float(ctx.view.width), float(ctx.view.height)) ||
                !std::isfinite(packet->parameters.nearZ) || !std::isfinite(packet->parameters.farZ) ||
                packet->parameters.nearZ <= 0 || packet->parameters.farZ <= packet->parameters.nearZ)
                throw std::invalid_argument("ClusterViz: incomplete or mismatched frozen depth/grid bindings");
            output.handle = m_Native.AddClusterVizPass(graph, input.handle, depth->handle, grid->handle, *packet, m_Debugger);
        }
        ctx.resources.Publish(RenderResources::VisualizedLDR, output);
    }
}

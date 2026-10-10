#include "luthpch.h"
#include "luth/renderer/features/GridFeature.h"
#include "luth/renderer/subsystems/EditorOverlaysSubsystem.h"
namespace Luth
{
    FeatureInfo GridFeature::Describe() const
    {
        FeatureInfo info; info.name = "Grid"; info.phase = FeaturePhase::AfterAsync;
        info.resources.reads = {{RenderResources::ResolvedHDR}, {GridResources::Bindings},
            {RenderResources::LitDepth, ResourceReadRequirement::Optional},
            {RenderResources::BloomOutput, ResourceReadRequirement::Optional}};
        info.resources.writes = {{RenderResources::GridHDR, ResourceOutputPresence::Required,
            ResourceKeyRef{RenderResources::ResolvedHDR}, true}};
        return info;
    }
    void GridFeature::Build(RG::RenderGraph& graph, RenderFeatureContext& ctx)
    {
        const auto& input = ctx.resources.Get(RenderResources::ResolvedHDR);
        const auto* packet = ctx.resources.Get(GridResources::Bindings).native;
        const auto valid = [&](const GraphTextureRef& ref, RG::TextureFormat format) {
            if (!ref.handle.IsValid() || ref.handle.index > graph.GetResources().size() ||
                !ref.binding.texture || ref.binding.baseMip || ref.binding.mipCount != 1 ||
                ref.binding.baseLayer || ref.binding.layerCount != 1) return false;
            const auto& desc = graph.GetResources()[ref.handle.index - 1].desc;
            return desc.format == format && desc.width == ctx.view.width && desc.height == ctx.view.height;
        };
        if (!packet || !ctx.view.width || !ctx.view.height || !valid(input, RG::TextureFormat::RGBA16_Float))
            throw std::invalid_argument("Grid: missing packet or incompatible HDR stage");
        GraphTextureRef output = input;
        if (packet->enabled && packet->pipeline)
        {
            const auto* depth = ctx.resources.TryGet(RenderResources::LitDepth);
            if (!depth || !valid(*depth, RG::TextureFormat::D32_Float) || !packet->layout || !packet->set ||
                !packet->state || packet->depth.texture != depth->binding.texture ||
                packet->depth.baseMip || packet->depth.mipCount != 1 || packet->depth.baseLayer || packet->depth.layerCount != 1 ||
                packet->state->sources[2].get() != depth->binding.texture || depth->handle.index == input.handle.index ||
                std::find(packet->state->gridSets.begin(), packet->state->gridSets.end(), packet->set) == packet->state->gridSets.end())
                throw std::invalid_argument("Grid: incomplete or mismatched frozen bindings");
            output.handle = m_Native.AddGridPass(graph, input.handle, depth->handle, *packet, m_Debugger);
        }
        ctx.resources.Publish(RenderResources::GridHDR, output);
    }
}

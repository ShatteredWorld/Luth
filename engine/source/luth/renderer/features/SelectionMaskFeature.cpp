#include "luthpch.h"
#include "luth/renderer/features/SelectionMaskFeature.h"
#include "luth/renderer/subsystems/EditorOverlaysSubsystem.h"
namespace Luth
{
    FeatureInfo SelectionMaskFeature::Describe() const
    {
        FeatureInfo info; info.name = "SelectionMask"; info.phase = FeaturePhase::AfterAsync;
        info.resources.reads = {{SelectionMaskResources::Bindings}};
        info.resources.writes = {{RenderResources::SelectionMask, ResourceOutputPresence::Optional},
            {RenderResources::SelectionDepth, ResourceOutputPresence::Optional}};
        return info;
    }
    void SelectionMaskFeature::Build(RG::RenderGraph& graph, RenderFeatureContext& ctx)
    {
        const auto* packet = ctx.resources.Get(SelectionMaskResources::Bindings).native;
        if (!packet) throw std::invalid_argument("SelectionMask: missing frozen packet");
        if (!packet->enabled)
        {
            ctx.resources.PublishAbsent(RenderResources::SelectionMask);
            ctx.resources.PublishAbsent(RenderResources::SelectionDepth); return;
        }
        if (!packet->state || !packet->width || !packet->height || packet->width != ctx.view.width ||
            packet->height != ctx.view.height || !packet->state->sources[0] || !packet->state->sources[1] ||
            !packet->images[0] || !packet->images[1] || packet->images[0] == packet->images[1] ||
            !packet->views[0] || !packet->views[1] || packet->state->sources[0] == packet->state->sources[1])
            throw std::invalid_argument("SelectionMask: incompatible output bindings or extent");
        for (const auto& resource : graph.GetResources())
            if (resource.image == packet->images[0] || resource.image == packet->images[1])
                throw std::invalid_argument("SelectionMask: output already imported");
        if (!packet->draws.empty())
        {
            if (!packet->pipeline || !packet->layout ||
                std::any_of(packet->sets.begin(), packet->sets.end(), [](auto set) { return !set; }))
                throw std::invalid_argument("SelectionMask: incomplete draw bindings");
            for (const auto& draw : packet->draws)
                if (!draw.vertex || !draw.index || !draw.vertexOwner || !draw.indexOwner ||
                    (packet->skinnedPipeline && !packet->skinnedLayout))
                    throw std::invalid_argument("SelectionMask: incomplete mesh bindings");
        }
        const auto output = m_Native.AddSelectionMaskPass(graph, *packet, m_Debugger);
        ctx.resources.Publish(RenderResources::SelectionMask, GraphTextureRef{output.mask, {packet->state->sources[0].get()}});
        ctx.resources.Publish(RenderResources::SelectionDepth, GraphTextureRef{output.depth, {packet->state->sources[1].get()}});
    }
}

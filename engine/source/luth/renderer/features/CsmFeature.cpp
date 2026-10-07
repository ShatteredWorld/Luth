#include "luthpch.h"
#include "luth/renderer/features/CsmFeature.h"
#include "luth/renderer/subsystems/LightingSubsystem.h"

namespace Luth
{
    FeatureInfo CsmFeature::Describe() const
    {
        FeatureInfo info;
        info.name = "CSM";
        info.phase = FeaturePhase::BeforeAsync;
        info.activation = FeatureActivation::Conditional;
        info.resources.reads = {{CsmResources::Parameters}, {CsmResources::Bindings},
            {RenderResources::CascadeVisibleDraws, ResourceReadRequirement::Optional}};
        info.resources.writes = {{RenderResources::ShadowCascades, ResourceOutputPresence::Optional}};
        info.capabilities.consumes = {{&DeformationResources::DeformedGeometry}};
        return info;
    }
    FeatureFrameDecision CsmFeature::Evaluate(const FeaturePrepareContext& ctx) const
    {
        for (const auto bindings : {ctx.frame.resources, ctx.view.resources})
            for (const auto& binding : bindings)
                if (const auto* params = binding.TryGet(CsmResources::Parameters)) return {params->enabled};
        throw std::invalid_argument("CSM: missing frozen parameters");
    }
    void CsmFeature::Prepare(FeaturePrepareContext& ctx)
    {
        if (!ctx.frame.draws || !ctx.frame.snapshot)
            throw std::invalid_argument("CSM: missing immutable frame inputs");
    }
    void CsmFeature::Build(RG::RenderGraph& graph, RenderFeatureContext& ctx)
    {
        const auto* ranges = ctx.resources.TryGet(RenderResources::CascadeVisibleDraws);
        const auto* bindings = ctx.resources.Get(CsmResources::Bindings).native;
        if (!ranges || !bindings || !bindings->texture || !bindings->image ||
            std::any_of(bindings->layers.begin(), bindings->layers.end(), [](auto layer) { return !layer; }) ||
            (bindings->rigid && (!bindings->rigidLayout ||
                std::any_of(bindings->sets.begin(), bindings->sets.end(), [](auto set) { return !set; }))) ||
            (bindings->deformed && !bindings->deformedLayout))
            throw std::invalid_argument("CSM: incomplete cascade ranges or native bindings");
        for (const auto& range : ranges->cascades)
            if (!range.indirect.handle.IsValid() || range.indirect.handle.index > graph.GetBuffers().size() ||
                !range.indirect.binding.slice || !range.indirect.binding.slice->buffer ||
                range.indirect.binding.offset != range.indirect.binding.slice->offset ||
                range.indirect.binding.size > range.indirect.binding.slice->size ||
                (u64(range.firstDraw) + range.maxDrawCount) * sizeof(VkDrawIndexedIndirectCommand) > range.indirect.binding.size)
                throw std::invalid_argument("CSM: cascade draw range is outside the current graph or native slice");
        ShadowCascadeRefs output;
        for (u32 i = 0; i < k_ShadowCascadeCount; ++i)
        {
            auto target = m_Native.ImportShadowTarget(graph, *bindings, i);
            target.handle = m_Native.AddShadowPass(graph, target.handle, ranges->cascades[i], *bindings, i,
                *ctx.frame.draws, *ctx.frame.snapshot, m_Debugger);
            output.cascades[i] = target;
        }
        ctx.resources.Publish(RenderResources::ShadowCascades, output);
    }
}

#include "luthpch.h"
#include "luth/renderer/features/DeformationFeature.h"
#include "luth/renderer/subsystems/SkinningSubsystem.h"

namespace Luth
{
    FeatureInfo DeformationFeature::Describe() const
    {
        FeatureInfo info;
        info.name = "Deformation";
        info.phase = FeaturePhase::BeforeAsync;
        info.resources.reads = {{DeformationResources::Parameters}};
        info.capabilities.provides = {&DeformationResources::DeformedGeometry};
        return info;
    }
    void DeformationFeature::Prepare(FeaturePrepareContext& ctx)
    {
        if (!ctx.frame.snapshot)
            throw std::invalid_argument("Deformation: missing immutable frame snapshot");
    }
    void DeformationFeature::Build(RG::RenderGraph& graph, RenderFeatureContext& ctx)
    {
        const auto& parameters = ctx.resources.Get(DeformationResources::Parameters);
        m_Native.AddDeformPass(graph, *ctx.frame.snapshot, parameters.wind,
            parameters.time, ctx.frame.renderFrameIndex);
    }
}

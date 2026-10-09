#include "luthpch.h"
#include "luth/renderer/features/rt/RtFogFeature.h"
#include "luth/renderer/features/FogComputeBindings.h"

namespace Luth
{
    FeatureInfo RtFogFeature::Describe() const
    {
        auto info = m_Compute.Describe();
        info.name = "RtFogCompute";
        info.resources.reads.push_back({RtSceneResources::Parameters});
        info.resources.reads.push_back({RtSceneResources::Scene, ResourceReadRequirement::Optional});
        info.capabilities.consumes = {{&RtSceneResources::RayScene}};
        info.capabilities.deviceRequirements = {&RtSceneResources::AccelerationStructures, &RtSceneResources::RayQueries};
        return info;
    }
    FeatureFrameDecision RtFogFeature::Evaluate(const FeaturePrepareContext& ctx) const
    {
        return {RtFogRequested(ctx) && m_Compute.Evaluate(ctx).active};
    }
    void RtFogFeature::Build(RG::RenderGraph& graph, RenderFeatureContext& ctx)
    {
        const auto& packet = *ctx.resources.Get(FogResources::Bindings).native;
        const auto* scene = ctx.resources.TryGet(RtSceneResources::Scene);
        if (!scene || !scene->native || scene->native->frameIndex != ctx.frame.renderFrameIndex ||
            (scene->native->tlas.result.instanceCount && !scene->native->HasSceneData()) ||
            !packet.rtShadows || !packet.tlas || packet.frameIndex != ctx.frame.renderFrameIndex ||
            packet.view != ctx.view.id || packet.generation != ctx.view.resourceGeneration ||
            packet.tlas != scene->native->GetTlas() || packet.inject.geomTableBDA != scene->native->GetGeometryTableBDA())
            throw std::invalid_argument("RtFogCompute: incomplete or stale paired scene/view bindings");
        m_Compute.BuildPrepared(graph, ctx, packet);
    }
}

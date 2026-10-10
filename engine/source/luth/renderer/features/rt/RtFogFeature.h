#pragma once
#include "luth/renderer/features/FogComputeFeature.h"
#include "luth/renderer/features/rt/RtSceneFeature.h"

namespace Luth
{
    inline bool RtFogRequested(const RtSceneParameters& params)
    {
        return params.active[static_cast<size_t>(RtSceneConsumer::Fog)] &&
            !params.active[static_cast<size_t>(RtSceneConsumer::PathTrace)];
    }
    inline bool RtFogRequested(const FeaturePrepareContext& ctx) { return RtFogRequested(GetRtSceneParameters(ctx)); }
    // Provider-segment demand until M18 joins the compiled segments.
    class RtFogDemandFeature final : public IRenderFeature
    {
    public:
        FeatureInfo Describe() const override
        {
            FeatureInfo info;
            info.name = "RtFogDemand"; info.phase = FeaturePhase::Async;
            info.activation = FeatureActivation::Conditional;
            info.resources.reads = {{RtSceneResources::Parameters}};
            info.capabilities.consumes = {{&RtSceneResources::RayScene}};
            info.capabilities.deviceRequirements = {&RtSceneResources::RayQueries};
            return info;
        }
        FeatureFrameDecision Evaluate(const FeaturePrepareContext& ctx) const override { return {RtFogRequested(ctx)}; }
        void Build(RG::RenderGraph&, RenderFeatureContext&) override {}
    };
    class RtFogFeature final : public IRenderFeature
    {
    public:
        explicit RtFogFeature(VolumetricSubsystem& native, FrameDebugger* debugger = nullptr)
            : m_Compute(native, debugger) {}
        FeatureInfo Describe() const override;
        FeatureFrameDecision Evaluate(const FeaturePrepareContext&) const override;
        void Build(RG::RenderGraph&, RenderFeatureContext&) override;
    private:
        FogComputeFeature m_Compute;
    };
}

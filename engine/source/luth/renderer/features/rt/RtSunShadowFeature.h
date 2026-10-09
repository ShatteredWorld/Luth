#pragma once
#include "luth/renderer/features/rt/RtSceneFeature.h"

namespace Luth
{
    struct RtSunShadowBindings;
    struct RtSunShadowBindingRef { const RtSunShadowBindings* native = nullptr; };
    namespace RtSunShadowResources
    {
        inline constexpr ResourceKeyIdentity BindingsIdentity{"RtSunShadow.Bindings"};
        inline constexpr RenderResourceKey<RtSunShadowBindingRef> Bindings{&BindingsIdentity};
        inline constexpr ResourceKeyIdentity MaskIdentity{"RtSunShadow.Mask"};
        inline constexpr RenderResourceKey<GraphTextureRef> Mask{&MaskIdentity};
    }
    inline bool RtSunShadowRequested(const FeaturePrepareContext& ctx)
    {
        const auto& params = GetRtSceneParameters(ctx);
        return params.active[static_cast<size_t>(RtSceneConsumer::SunShadow)] &&
            !params.active[static_cast<size_t>(RtSceneConsumer::PathTrace)];
    }
    // CPU demand contribution in the provider segment until the full composition joins
    // the current segments. It shares the actual technique's activation policy.
    class RtSunShadowDemandFeature final : public IRenderFeature
    {
    public:
        FeatureInfo Describe() const override
        {
            FeatureInfo info;
            info.name = "RtSunShadowDemand";
            info.phase = FeaturePhase::Async;
            info.activation = FeatureActivation::Conditional;
            info.resources.reads = {{RtSceneResources::Parameters}};
            info.capabilities.consumes = {{&RtSceneResources::RayScene}};
            info.capabilities.deviceRequirements = {&RtSceneResources::RayQueries};
            return info;
        }
        FeatureFrameDecision Evaluate(const FeaturePrepareContext& ctx) const override
        { return {RtSunShadowRequested(ctx)}; }
        void Build(RG::RenderGraph&, RenderFeatureContext&) override {}
    };
    class RtSunShadowFeature final : public IRenderFeature
    {
    public:
        FeatureInfo Describe() const override;
        FeatureFrameDecision Evaluate(const FeaturePrepareContext&) const override;
        void Build(RG::RenderGraph&, RenderFeatureContext&) override;
    };
}

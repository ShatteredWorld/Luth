#pragma once
#include "luth/renderer/features/rt/RtSceneFeature.h"

namespace Luth
{
    struct ReflectionBindings;
    struct ReflectionBindingRef { const ReflectionBindings* native = nullptr; };
    namespace ReflectionResources
    {
        inline constexpr ResourceKeyIdentity BindingsIdentity{"Reflection.Bindings"};
        inline constexpr RenderResourceKey<ReflectionBindingRef> Bindings{&BindingsIdentity};
        inline constexpr ResourceKeyIdentity RadianceIdentity{"Reflection.DemodulatedRadianceHitDistance"};
        inline constexpr RenderResourceKey<GraphTextureRef> Radiance{&RadianceIdentity};
    }
    inline bool ReflectionRequested(const FeaturePrepareContext& ctx)
    {
        const auto& params = GetRtSceneParameters(ctx);
        return params.active[static_cast<size_t>(RtSceneConsumer::Reflections)] &&
            !params.active[static_cast<size_t>(RtSceneConsumer::PathTrace)];
    }
    class ReflectionDemandFeature final : public IRenderFeature
    {
    public:
        FeatureInfo Describe() const override {
            FeatureInfo info; info.name = "ReflectionDemand"; info.phase = FeaturePhase::Async;
            info.activation = FeatureActivation::Conditional;
            info.resources.reads = {{RtSceneResources::Parameters}};
            info.capabilities.consumes = {{&RtSceneResources::RayScene}};
            info.capabilities.deviceRequirements = {&RtSceneResources::RayQueries};
            return info;
        }
        FeatureFrameDecision Evaluate(const FeaturePrepareContext& ctx) const override { return {ReflectionRequested(ctx)}; }
        void Build(RG::RenderGraph&, RenderFeatureContext&) override {}
    };
    class ReflectionFeature final : public IRenderFeature
    {
    public:
        FeatureInfo Describe() const override;
        FeatureFrameDecision Evaluate(const FeaturePrepareContext&) const override;
        void Build(RG::RenderGraph&, RenderFeatureContext&) override;
    };
}

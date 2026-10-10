#pragma once
#include "luth/renderer/features/rt/RtSceneFeature.h"
#include "luth/renderer/features/rt/GiReservoirVizFeature.h"
namespace Luth
{
    struct RestirGiBindings;
    struct RestirGiBindingRef { const RestirGiBindings* native = nullptr; };
    namespace RestirGiResources
    {
        inline constexpr ResourceKeyIdentity BindingsIdentity{"RestirGI.Bindings"};
        inline constexpr RenderResourceKey<RestirGiBindingRef> Bindings{&BindingsIdentity};
        inline constexpr ResourceKeyIdentity DiffuseIdentity{"RestirGI.DemodulatedDiffuse"};
        inline constexpr RenderResourceKey<GraphTextureRef> Diffuse{&DiffuseIdentity};
    }
    inline bool RestirGiRequested(const FeaturePrepareContext& ctx)
    {
        const auto& params = GetRtSceneParameters(ctx);
        return params.active[static_cast<size_t>(RtSceneConsumer::GlobalIllumination)] &&
            !params.active[static_cast<size_t>(RtSceneConsumer::PathTrace)];
    }
    class RestirGiDemandFeature final : public IRenderFeature
    {
    public:
        FeatureInfo Describe() const override {
            FeatureInfo info; info.name = "RestirGiDemand"; info.phase = FeaturePhase::Async;
            info.activation = FeatureActivation::Conditional;
            info.resources.reads = {{RtSceneResources::Parameters}};
            info.capabilities.consumes = {{&RtSceneResources::RayScene}};
            info.capabilities.deviceRequirements = {&RtSceneResources::RayQueries};
            return info;
        }
        FeatureFrameDecision Evaluate(const FeaturePrepareContext& ctx) const override { return {RestirGiRequested(ctx)}; }
        void Build(RG::RenderGraph&, RenderFeatureContext&) override {}
    };
    class RestirGiFeature final : public IRenderFeature
    {
    public:
        FeatureInfo Describe() const override;
        FeatureFrameDecision Evaluate(const FeaturePrepareContext&) const override;
        void Build(RG::RenderGraph&, RenderFeatureContext&) override;
    };
}

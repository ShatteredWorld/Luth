#pragma once
#include "luth/renderer/features/rt/RtSceneFeature.h"
namespace Luth
{
    struct RestirDiBindings;
    struct RestirDiBindingRef { const RestirDiBindings* native = nullptr; };
    namespace RestirDiResources
    {
        inline constexpr ResourceKeyIdentity BindingsIdentity{"RestirDI.Bindings"};
        inline constexpr RenderResourceKey<RestirDiBindingRef> Bindings{&BindingsIdentity};
        inline constexpr ResourceKeyIdentity DiffuseIdentity{"RestirDI.DemodulatedDiffuse"};
        inline constexpr RenderResourceKey<GraphTextureRef> Diffuse{&DiffuseIdentity};
        inline constexpr ResourceKeyIdentity SpecularIdentity{"RestirDI.DemodulatedSpecular"};
        inline constexpr RenderResourceKey<GraphTextureRef> Specular{&SpecularIdentity};
    }
    inline bool RestirDiRequested(const FeaturePrepareContext& ctx)
    {
        const auto& params = GetRtSceneParameters(ctx);
        return params.active[static_cast<size_t>(RtSceneConsumer::DirectLighting)] &&
            !params.active[static_cast<size_t>(RtSceneConsumer::PathTrace)];
    }
    class RestirDiDemandFeature final : public IRenderFeature
    {
    public:
        FeatureInfo Describe() const override {
            FeatureInfo info; info.name = "RestirDiDemand"; info.phase = FeaturePhase::Async;
            info.activation = FeatureActivation::Conditional;
            info.resources.reads = {{RtSceneResources::Parameters}};
            info.capabilities.consumes = {{&RtSceneResources::RayScene}};
            info.capabilities.deviceRequirements = {&RtSceneResources::RayQueries};
            return info;
        }
        FeatureFrameDecision Evaluate(const FeaturePrepareContext& ctx) const override { return {RestirDiRequested(ctx)}; }
        void Build(RG::RenderGraph&, RenderFeatureContext&) override {}
    };
    class RestirDiFeature final : public IRenderFeature
    {
    public:
        FeatureInfo Describe() const override;
        FeatureFrameDecision Evaluate(const FeaturePrepareContext&) const override;
        void Build(RG::RenderGraph&, RenderFeatureContext&) override;
    };
}

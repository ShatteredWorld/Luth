#pragma once
#include "luth/renderer/features/rt/ReflectionDenoiserFeature.h"
#include "luth/renderer/features/rt/ReflectionUpscaleViewState.h"
namespace Luth
{
    struct ReflectionUpscaleBindingRef { const ReflectionUpscaleBindings* native = nullptr; };
    namespace ReflectionUpscaleResources
    {
        inline constexpr ResourceKeyIdentity BindingsIdentity{"ReflectionUpscale.Bindings"};
        inline constexpr RenderResourceKey<ReflectionUpscaleBindingRef> Bindings{&BindingsIdentity};
        inline constexpr ResourceKeyIdentity RadianceIdentity{"ReflectionUpscale.DemodulatedRadiance"};
        inline constexpr RenderResourceKey<GraphTextureRef> Radiance{&RadianceIdentity};
    }
    class ReflectionUpscaleFeature final : public IRenderFeature
    {
    public:
        FeatureInfo Describe() const override;
        FeatureFrameDecision Evaluate(const FeaturePrepareContext&) const override { return {true}; }
        void Build(RG::RenderGraph&, RenderFeatureContext&) override;
    };
}

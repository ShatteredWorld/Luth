#pragma once
#include "luth/renderer/features/rt/ReflectionFeature.h"
namespace Luth
{
    struct ReflectionDenoiserBindings;
    struct ReflectionDenoiserBindingRef { const ReflectionDenoiserBindings* native = nullptr; };
    namespace ReflectionDenoiserResources
    {
        inline constexpr ResourceKeyIdentity BindingsIdentity{"DenoiseReflection.Bindings"};
        inline constexpr RenderResourceKey<ReflectionDenoiserBindingRef> Bindings{&BindingsIdentity};
        inline constexpr ResourceKeyIdentity RadianceIdentity{"DenoiseReflection.DemodulatedRadiance"};
        inline constexpr RenderResourceKey<GraphTextureRef> Radiance{&RadianceIdentity};
    }
    class ReflectionDenoiserFeature final : public IRenderFeature
    {
    public:
        FeatureInfo Describe() const override;
        FeatureFrameDecision Evaluate(const FeaturePrepareContext&) const override { return {true}; }
        void Build(RG::RenderGraph&, RenderFeatureContext&) override;
    };
}

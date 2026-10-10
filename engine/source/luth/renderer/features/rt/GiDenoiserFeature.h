#pragma once
#include "luth/renderer/features/rt/RestirGiFeature.h"
namespace Luth
{
    struct GiDenoiserBindings;
    struct GiDenoiserBindingRef { const GiDenoiserBindings* native = nullptr; };
    namespace GiDenoiserResources
    {
        inline constexpr ResourceKeyIdentity BindingsIdentity{"DenoiseGI.Bindings"};
        inline constexpr RenderResourceKey<GiDenoiserBindingRef> Bindings{&BindingsIdentity};
        inline constexpr ResourceKeyIdentity DiffuseIdentity{"DenoiseGI.DemodulatedDiffuse"};
        inline constexpr RenderResourceKey<GraphTextureRef> Diffuse{&DiffuseIdentity};
    }
    class GiDenoiserFeature final : public IRenderFeature
    {
    public:
        FeatureInfo Describe() const override;
        FeatureFrameDecision Evaluate(const FeaturePrepareContext&) const override { return {true}; }
        void Build(RG::RenderGraph&, RenderFeatureContext&) override;
    };
}

#pragma once
#include "luth/renderer/features/rt/RestirDiFeature.h"
namespace Luth
{
    struct DiDenoiserBindings;
    struct DiDenoiserBindingRef { const DiDenoiserBindings* native = nullptr; };
    namespace DiDenoiserResources
    {
        inline constexpr ResourceKeyIdentity BindingsIdentity{"DenoiseDI.Bindings"};
        inline constexpr RenderResourceKey<DiDenoiserBindingRef> Bindings{&BindingsIdentity};
        inline constexpr ResourceKeyIdentity DiffuseIdentity{"DenoiseDI.DemodulatedDiffuse"};
        inline constexpr RenderResourceKey<GraphTextureRef> Diffuse{&DiffuseIdentity};
    }
    class DiDenoiserFeature final : public IRenderFeature
    {
    public:
        FeatureInfo Describe() const override;
        FeatureFrameDecision Evaluate(const FeaturePrepareContext&) const override { return {true}; }
        void Build(RG::RenderGraph&, RenderFeatureContext&) override;
    };
}

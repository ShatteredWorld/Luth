#pragma once
#include "luth/renderer/features/rt/GiDenoiserFeature.h"
#include "luth/renderer/features/rt/GiUpscaleViewState.h"
namespace Luth
{
    struct GiUpscaleBindingRef { const GiUpscaleBindings* native = nullptr; };
    namespace GiUpscaleResources
    {
        inline constexpr ResourceKeyIdentity BindingsIdentity{"GiUpscale.Bindings"};
        inline constexpr RenderResourceKey<GiUpscaleBindingRef> Bindings{&BindingsIdentity};
        inline constexpr ResourceKeyIdentity DiffuseIdentity{"GiUpscale.DemodulatedDiffuse"};
        inline constexpr RenderResourceKey<GraphTextureRef> Diffuse{&DiffuseIdentity};
    }
    class GiUpscaleFeature final : public IRenderFeature
    {
    public:
        FeatureInfo Describe() const override;
        FeatureFrameDecision Evaluate(const FeaturePrepareContext&) const override { return {true}; }
        void Build(RG::RenderGraph&, RenderFeatureContext&) override;
    };
}

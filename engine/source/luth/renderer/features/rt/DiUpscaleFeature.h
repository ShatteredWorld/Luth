#pragma once
#include "luth/renderer/features/rt/DiDenoiserFeature.h"
#include "luth/renderer/features/rt/DiUpscaleViewState.h"

namespace Luth
{
    struct DiUpscaleBindingRef { const DiUpscaleBindings* native = nullptr; };
    namespace DiUpscaleResources
    {
        inline constexpr ResourceKeyIdentity BindingsIdentity{"DiUpscale.Bindings"};
        inline constexpr RenderResourceKey<DiUpscaleBindingRef> Bindings{&BindingsIdentity};
        inline constexpr ResourceKeyIdentity SpecularBindingsIdentity{"DiUpscale.SpecularBindings"};
        inline constexpr RenderResourceKey<DiUpscaleBindingRef> SpecularBindings{&SpecularBindingsIdentity};
        inline constexpr ResourceKeyIdentity DiffuseIdentity{"DiUpscale.DemodulatedDiffuse"};
        inline constexpr RenderResourceKey<GraphTextureRef> Diffuse{&DiffuseIdentity};
        inline constexpr ResourceKeyIdentity SpecularIdentity{"DiUpscale.Specular"};
        inline constexpr RenderResourceKey<GraphTextureRef> Specular{&SpecularIdentity};
    }
    class DiUpscaleFeature final : public IRenderFeature
    {
    public:
        explicit DiUpscaleFeature(DiDenoiserSignal signal = DiDenoiserSignal::Diffuse) : m_Signal(signal) {
            if (signal != DiDenoiserSignal::Diffuse && signal != DiDenoiserSignal::Specular)
                throw std::invalid_argument("DI upscale feature: invalid signal");
        }
        FeatureInfo Describe() const override;
        FeatureFrameDecision Evaluate(const FeaturePrepareContext&) const override { return {true}; }
        void Build(RG::RenderGraph&, RenderFeatureContext&) override;
    private:
        DiDenoiserSignal m_Signal;
    };
}

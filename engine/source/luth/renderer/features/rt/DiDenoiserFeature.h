#pragma once
#include "luth/renderer/features/rt/RestirDiFeature.h"
#include "luth/renderer/features/rt/DiDenoiserViewState.h"
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
        inline constexpr ResourceKeyIdentity SpecularBindingsIdentity{"DenoiseDI.SpecularBindings"};
        inline constexpr RenderResourceKey<DiDenoiserBindingRef> SpecularBindings{&SpecularBindingsIdentity};
        inline constexpr ResourceKeyIdentity SpecularIdentity{"DenoiseDI.Specular"};
        inline constexpr RenderResourceKey<GraphTextureRef> Specular{&SpecularIdentity};
    }
    class DiDenoiserFeature final : public IRenderFeature
    {
    public:
        explicit DiDenoiserFeature(DiDenoiserSignal signal = DiDenoiserSignal::Diffuse) : m_Signal(signal) {
            if (signal != DiDenoiserSignal::Diffuse && signal != DiDenoiserSignal::Specular)
                throw std::invalid_argument("DI denoiser feature: invalid signal");
        }
        FeatureInfo Describe() const override;
        FeatureFrameDecision Evaluate(const FeaturePrepareContext&) const override { return {true}; }
        void Build(RG::RenderGraph&, RenderFeatureContext&) override;
    private:
        DiDenoiserSignal m_Signal;
    };
}

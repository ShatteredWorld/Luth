#pragma once
#include "luth/renderer/features/RenderFeature.h"
#include "luth/renderer/features/RefractionBackdropBindings.h"

namespace Luth
{
    class TransparencySubsystem;
    struct RefractionBackdropBindingRef { const RefractionBackdropBindings* native = nullptr; };
    namespace RefractionResources
    {
        inline constexpr ResourceKeyIdentity BindingsIdentity{"RefractionBackdrop.Bindings"};
        inline constexpr RenderResourceKey<RefractionBackdropBindingRef> Bindings{&BindingsIdentity};
    }
    class RefractionBackdropFeature final : public IRenderFeature
    {
    public:
        explicit RefractionBackdropFeature(TransparencySubsystem& native) : m_Native(native) {}
        FeatureInfo Describe() const override;
        FeatureFrameDecision Evaluate(const FeaturePrepareContext&) const override;
        void Build(RG::RenderGraph&, RenderFeatureContext&) override;
    private:
        TransparencySubsystem& m_Native;
    };
}

#pragma once
#include "luth/renderer/features/RenderFeature.h"
#include "luth/renderer/features/FogCompositeBindings.h"

namespace Luth
{
    class VolumetricSubsystem;
    struct FrameDebugger;
    struct FogCompositeBindingRef { const FogCompositeBindings* native = nullptr; };
    namespace FogCompositeResources
    {
        inline constexpr ResourceKeyIdentity BindingsIdentity{"FogComposite.Bindings"};
        inline constexpr RenderResourceKey<FogCompositeBindingRef> Bindings{&BindingsIdentity};
    }
    class FogCompositeFeature final : public IRenderFeature
    {
    public:
        explicit FogCompositeFeature(VolumetricSubsystem& native, FrameDebugger* debugger = nullptr)
            : m_Native(native), m_Debugger(debugger) {}
        FeatureInfo Describe() const override;
        FeatureFrameDecision Evaluate(const FeaturePrepareContext&) const override { return {}; }
        void Build(RG::RenderGraph&, RenderFeatureContext&) override;
    private:
        VolumetricSubsystem& m_Native;
        FrameDebugger* m_Debugger;
    };
}

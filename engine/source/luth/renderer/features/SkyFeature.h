#pragma once
#include "luth/renderer/features/RenderFeature.h"

namespace Luth
{
    class LightingSubsystem;
    struct SkyBindings;
    struct FrameDebugger;
    struct SkyBindingRef { const SkyBindings* native = nullptr; };
    namespace SkyResources
    {
        inline constexpr ResourceKeyIdentity BindingsIdentity{"Sky.Bindings"};
        inline constexpr RenderResourceKey<SkyBindingRef> Bindings{&BindingsIdentity};
    }
    class SkyFeature final : public IRenderFeature
    {
    public:
        explicit SkyFeature(LightingSubsystem& native, FrameDebugger* debugger = nullptr)
            : m_Native(native), m_Debugger(debugger) {}
        FeatureInfo Describe() const override;
        FeatureFrameDecision Evaluate(const FeaturePrepareContext&) const override { return {}; }
        void Build(RG::RenderGraph&, RenderFeatureContext&) override;
    private:
        LightingSubsystem& m_Native;
        FrameDebugger* m_Debugger;
    };
}

#pragma once
#include "luth/renderer/features/DeformationFeature.h"

namespace Luth
{
    class LightingSubsystem;
    struct CsmBindings;
    struct FrameDebugger;
    struct CsmParameters { bool enabled = false; };
    struct CsmBindingRef { const CsmBindings* native = nullptr; };
    namespace CsmResources
    {
        inline constexpr ResourceKeyIdentity ParametersIdentity{"CSM.Parameters"};
        inline constexpr RenderResourceKey<CsmParameters> Parameters{&ParametersIdentity};
        inline constexpr ResourceKeyIdentity BindingsIdentity{"CSM.Bindings"};
        inline constexpr RenderResourceKey<CsmBindingRef> Bindings{&BindingsIdentity};
    }
    class CsmFeature final : public IRenderFeature
    {
    public:
        explicit CsmFeature(LightingSubsystem& native, FrameDebugger* debugger = nullptr)
            : m_Native(native), m_Debugger(debugger) {}
        FeatureInfo Describe() const override;
        FeatureFrameDecision Evaluate(const FeaturePrepareContext&) const override;
        void Prepare(FeaturePrepareContext&) override;
        void Build(RG::RenderGraph&, RenderFeatureContext&) override;
    private:
        LightingSubsystem& m_Native;
        FrameDebugger* m_Debugger;
    };
}

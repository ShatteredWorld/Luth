#pragma once
#include "luth/renderer/features/RenderFeature.h"
#include "luth/renderer/features/DebugDrawBindings.h"
namespace Luth
{
    class DebugDrawSubsystem;
    struct FrameDebugger;
    struct DebugDrawBindingRef { const DebugDrawBindings* native = nullptr; };
    namespace DebugDrawResources
    {
        inline constexpr ResourceKeyIdentity BindingsIdentity{"DebugDraw.Bindings"};
        inline constexpr RenderResourceKey<DebugDrawBindingRef> Bindings{&BindingsIdentity};
    }
    class DebugDrawFeature final : public IRenderFeature
    {
    public:
        explicit DebugDrawFeature(DebugDrawSubsystem& native, FrameDebugger* debugger = nullptr)
            : m_Native(native), m_Debugger(debugger) {}
        FeatureInfo Describe() const override;
        FeatureFrameDecision Evaluate(const FeaturePrepareContext&) const override { return {}; }
        void Build(RG::RenderGraph&, RenderFeatureContext&) override;
    private:
        DebugDrawSubsystem& m_Native;
        FrameDebugger* m_Debugger;
    };
}

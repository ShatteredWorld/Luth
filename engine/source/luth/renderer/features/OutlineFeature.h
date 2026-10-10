#pragma once
#include "luth/renderer/features/RenderFeature.h"
#include "luth/renderer/features/OutlineBindings.h"
namespace Luth
{
    class EditorOverlaysSubsystem;
    struct FrameDebugger;
    struct OutlineBindingRef { const OutlineBindings* native = nullptr; };
    namespace OutlineResources
    {
        inline constexpr ResourceKeyIdentity BindingsIdentity{"Outline.Bindings"};
        inline constexpr RenderResourceKey<OutlineBindingRef> Bindings{&BindingsIdentity};
    }
    class OutlineFeature final : public IRenderFeature
    {
    public:
        explicit OutlineFeature(EditorOverlaysSubsystem& native, FrameDebugger* debugger = nullptr)
            : m_Native(native), m_Debugger(debugger) {}
        FeatureInfo Describe() const override;
        FeatureFrameDecision Evaluate(const FeaturePrepareContext&) const override { return {}; }
        void Build(RG::RenderGraph&, RenderFeatureContext&) override;
    private:
        EditorOverlaysSubsystem& m_Native;
        FrameDebugger* m_Debugger;
    };
}

#pragma once
#include "luth/renderer/features/RenderFeature.h"
#include "luth/renderer/features/SelectionMaskBindings.h"
namespace Luth
{
    class EditorOverlaysSubsystem;
    struct FrameDebugger;
    struct SelectionMaskBindingRef { const SelectionMaskBindings* native = nullptr; };
    namespace SelectionMaskResources
    {
        inline constexpr ResourceKeyIdentity BindingsIdentity{"SelectionMask.Bindings"};
        inline constexpr RenderResourceKey<SelectionMaskBindingRef> Bindings{&BindingsIdentity};
    }
    class SelectionMaskFeature final : public IRenderFeature
    {
    public:
        explicit SelectionMaskFeature(EditorOverlaysSubsystem& native, FrameDebugger* debugger = nullptr)
            : m_Native(native), m_Debugger(debugger) {}
        FeatureInfo Describe() const override;
        FeatureFrameDecision Evaluate(const FeaturePrepareContext&) const override { return {}; }
        void Build(RG::RenderGraph&, RenderFeatureContext&) override;
    private:
        EditorOverlaysSubsystem& m_Native;
        FrameDebugger* m_Debugger;
    };
}

#pragma once
#include "luth/renderer/features/RenderFeature.h"
#include "luth/renderer/features/GridBindings.h"
namespace Luth
{
    class EditorOverlaysSubsystem;
    struct FrameDebugger;
    struct GridBindingRef { const GridBindings* native = nullptr; };
    namespace GridResources
    {
        inline constexpr ResourceKeyIdentity BindingsIdentity{"Grid.Bindings"};
        inline constexpr RenderResourceKey<GridBindingRef> Bindings{&BindingsIdentity};
    }
    class GridFeature final : public IRenderFeature
    {
    public:
        explicit GridFeature(EditorOverlaysSubsystem& native, FrameDebugger* debugger = nullptr)
            : m_Native(native), m_Debugger(debugger) {}
        FeatureInfo Describe() const override;
        FeatureFrameDecision Evaluate(const FeaturePrepareContext&) const override { return {}; }
        void Build(RG::RenderGraph&, RenderFeatureContext&) override;
    private:
        EditorOverlaysSubsystem& m_Native;
        FrameDebugger* m_Debugger;
    };
}

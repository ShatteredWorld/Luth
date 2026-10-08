#pragma once
#include "luth/renderer/features/RenderFeature.h"
#include "luth/renderer/features/SlimVizBindings.h"
namespace Luth
{
    class PostProcessSubsystem;
    struct FrameDebugger;
    struct SlimVizBindingRef { const SlimVizBindings* native = nullptr; };
    namespace SlimVizResources
    {
        inline constexpr ResourceKeyIdentity BindingsIdentity{"SlimViz.Bindings"};
        inline constexpr RenderResourceKey<SlimVizBindingRef> Bindings{&BindingsIdentity};
    }
    class SlimVizFeature final : public IRenderFeature
    {
    public:
        explicit SlimVizFeature(PostProcessSubsystem& native, FrameDebugger* debugger = nullptr)
            : m_Native(native), m_Debugger(debugger) {}
        FeatureInfo Describe() const override;
        FeatureFrameDecision Evaluate(const FeaturePrepareContext&) const override { return {}; }
        void Build(RG::RenderGraph&, RenderFeatureContext&) override;
    private:
        PostProcessSubsystem& m_Native;
        FrameDebugger* m_Debugger;
    };
}

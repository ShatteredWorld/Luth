#pragma once
#include "luth/renderer/features/SlimVizFeature.h"
#include "luth/renderer/features/ClusterVizBindings.h"
namespace Luth
{
    class LightingSubsystem;
    struct ClusterVizBindingRef { const ClusterVizBindings* native = nullptr; };
    namespace ClusterVizResources
    {
        inline constexpr ResourceKeyIdentity OutputIdentity{"ClusterViz.LDR"};
        inline constexpr RenderResourceKey<GraphTextureRef> Output{&OutputIdentity};
        inline constexpr ResourceKeyIdentity BindingsIdentity{"ClusterViz.Bindings"};
        inline constexpr RenderResourceKey<ClusterVizBindingRef> Bindings{&BindingsIdentity};
    }
    class ClusterVizFeature final : public IRenderFeature
    {
    public:
        explicit ClusterVizFeature(LightingSubsystem& native, FrameDebugger* debugger = nullptr)
            : m_Native(native), m_Debugger(debugger) {}
        FeatureInfo Describe() const override;
        FeatureFrameDecision Evaluate(const FeaturePrepareContext&) const override { return {}; }
        void Build(RG::RenderGraph&, RenderFeatureContext&) override;
    private:
        LightingSubsystem& m_Native;
        FrameDebugger* m_Debugger;
    };
}

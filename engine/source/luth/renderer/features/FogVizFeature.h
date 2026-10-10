#pragma once
#include "luth/renderer/features/ClusterVizFeature.h"
#include "luth/renderer/features/FogVizBindings.h"
namespace Luth
{
    class VolumetricSubsystem;
    struct FogVizBindingRef { const FogVizBindings* native = nullptr; };
    namespace FogVizResources
    {
        inline constexpr ResourceKeyIdentity OutputIdentity{"FogViz.LDR"};
        inline constexpr RenderResourceKey<GraphTextureRef> Output{&OutputIdentity};
        inline constexpr ResourceKeyIdentity BindingsIdentity{"FogViz.Bindings"};
        inline constexpr RenderResourceKey<FogVizBindingRef> Bindings{&BindingsIdentity};
    }
    class FogVizFeature final : public IRenderFeature
    {
    public:
        explicit FogVizFeature(VolumetricSubsystem& native, FrameDebugger* debugger = nullptr)
            : m_Native(native), m_Debugger(debugger) {}
        FeatureInfo Describe() const override;
        FeatureFrameDecision Evaluate(const FeaturePrepareContext&) const override { return {}; }
        void Build(RG::RenderGraph&, RenderFeatureContext&) override;
    private:
        VolumetricSubsystem& m_Native;
        FrameDebugger* m_Debugger;
    };
}

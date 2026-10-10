#pragma once
#include "luth/renderer/features/RenderFeature.h"

namespace Luth
{
    class LightingSubsystem;
    struct ClusterBindings;
    struct FrameDebugger;
    struct ClusterBindingRef { const ClusterBindings* native = nullptr; };
    namespace ClusterResources
    {
        inline constexpr ResourceKeyIdentity BindingsIdentity{"ClusteredLighting.Bindings"};
        inline constexpr RenderResourceKey<ClusterBindingRef> Bindings{&BindingsIdentity};
        inline constexpr ResourceKeyIdentity UploadedLightsIdentity{"ClusteredLighting.UploadedLights"};
        inline constexpr RenderResourceKey<GraphBufferRef> UploadedLights{&UploadedLightsIdentity};
    }
    class ClusteredLightingFeature final : public IRenderFeature
    {
    public:
        explicit ClusteredLightingFeature(LightingSubsystem& native, FrameDebugger* debugger = nullptr)
            : m_Native(native), m_Debugger(debugger) {}
        FeatureInfo Describe() const override;
        FeatureFrameDecision Evaluate(const FeaturePrepareContext&) const override;
        void Build(RG::RenderGraph&, RenderFeatureContext&) override;
    private:
        LightingSubsystem& m_Native;
        FrameDebugger* m_Debugger;
    };
}

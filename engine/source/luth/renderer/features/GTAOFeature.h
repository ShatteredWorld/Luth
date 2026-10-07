#pragma once

#include "luth/renderer/features/RenderFeature.h"
#include "luth/renderer/features/GtaoViewState.h"

namespace Luth
{
    class GTAOSubsystem;
    struct FrameDebugger;
    struct GtaoFrameParameters
    {
        bool enabled = false; // Frozen effective setting, including native readiness.
        bool realtime = true;
        u32 shaderFrameIndex = 0; // Preserve the presentation-frame noise seed.
    };
    namespace GtaoResources
    {
        inline constexpr ResourceKeyIdentity ParametersIdentity{"GTAO.Parameters"};
        inline constexpr RenderResourceKey<GtaoFrameParameters> Parameters{&ParametersIdentity};
    }
    class GTAOFeature final : public IRenderFeature
    {
    public:
        GTAOFeature(GTAOSubsystem&, GtaoViewStateStore&, FrameDebugger* = nullptr);
        FeatureInfo Describe() const override;
        FeatureFrameDecision Evaluate(const FeaturePrepareContext&) const override;
        void Prepare(FeaturePrepareContext&) override;
        void Build(RG::RenderGraph&, RenderFeatureContext&) override;
        void ReleaseView(RenderViewId) override;
    private:
        GTAOSubsystem& m_Native;
        GtaoViewStateStore& m_States;
        FrameDebugger* m_Debugger;
    };
}

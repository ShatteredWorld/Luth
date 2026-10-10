#pragma once

#include "luth/renderer/features/RenderFeature.h"
#include "luth/renderer/settings/WindSettings.h"

namespace Luth
{
    class SkinningSubsystem;
    struct DeformationParameters
    {
        WindSettings wind;
        f32 time = 0.0f;
    };
    namespace DeformationResources
    {
        inline constexpr ResourceKeyIdentity ParametersIdentity{"Deformation.Parameters"};
        inline constexpr RenderResourceKey<DeformationParameters> Parameters{&ParametersIdentity};
        inline constexpr RenderCapabilityIdentity DeformedGeometry{"DeformedGeometry"};
    }
    // Always runs, including wind-disabled and path-tracing views. Native buffers still
    // belong to BLAS during this migration; this capability declares ordering, not ownership.
    class DeformationFeature final : public IRenderFeature
    {
    public:
        explicit DeformationFeature(SkinningSubsystem& native) : m_Native(native) {}
        FeatureInfo Describe() const override;
        FeatureFrameDecision Evaluate(const FeaturePrepareContext&) const override { return {}; }
        void Prepare(FeaturePrepareContext&) override;
        void Build(RG::RenderGraph&, RenderFeatureContext&) override;
    private:
        SkinningSubsystem& m_Native;
    };
}

#pragma once

#include "luth/renderer/features/DeformationFeature.h"

namespace Luth
{
    class GeometrySubsystem;
    struct FrameDebugger;
    struct VisibilityParameters
    {
        std::array<Vec4, 6> cameraPlanes{};
        std::array<std::array<Vec4, 6>, 4> cascadePlanes{};
        u32 viewIndex = 0; // Submission slot, independent of stable view identity.
        u32 regionStride = 4096;
        u32 maxViews = 2;
        u32 objectCount = 0;
        bool realtime = true;
        bool cullCascades = false;
    };
    namespace VisibilityResources
    {
        inline constexpr ResourceKeyIdentity ParametersIdentity{"Visibility.Parameters"};
        inline constexpr RenderResourceKey<VisibilityParameters> Parameters{&ParametersIdentity};
        inline constexpr ResourceKeyIdentity CulledIndirectIdentity{"Visibility.CulledIndirect"};
        inline constexpr RenderResourceKey<GraphBufferRef> CulledIndirect{&CulledIndirectIdentity};
    }
    class VisibilityFeature final : public IRenderFeature
    {
    public:
        explicit VisibilityFeature(GeometrySubsystem& native, FrameDebugger* debugger = nullptr)
            : m_Native(native), m_Debugger(debugger) {}
        FeatureInfo Describe() const override;
        FeatureFrameDecision Evaluate(const FeaturePrepareContext&) const override;
        void Build(RG::RenderGraph&, RenderFeatureContext&) override;
    private:
        GeometrySubsystem& m_Native;
        FrameDebugger* m_Debugger;
    };
}

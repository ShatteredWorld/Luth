#pragma once
#include "luth/renderer/features/FogVizFeature.h"
#include "luth/renderer/features/rt/GiReservoirVizBindings.h"
namespace Luth
{
    class RtRestirGiSubsystem;
    struct GiReservoirVizBindingRef { const GiReservoirVizBindings* native = nullptr; };
    namespace GiReservoirVizResources
    {
        inline constexpr ResourceKeyIdentity BindingsIdentity{"RT.GiReservoirViz.Bindings"}, ReservoirIdentity{"RT.GI.SpatialReservoir"};
        inline constexpr RenderResourceKey<GiReservoirVizBindingRef> Bindings{&BindingsIdentity};
        inline constexpr RenderResourceKey<GraphBufferRef> SpatialReservoir{&ReservoirIdentity};
    }
    class GiReservoirVizFeature final : public IRenderFeature
    {
    public:
        explicit GiReservoirVizFeature(RtRestirGiSubsystem& native) : m_Native(native) {}
        FeatureInfo Describe() const override;
        FeatureFrameDecision Evaluate(const FeaturePrepareContext&) const override { return {}; }
        void Build(RG::RenderGraph&, RenderFeatureContext&) override;
    private:
        RtRestirGiSubsystem& m_Native;
    };
}

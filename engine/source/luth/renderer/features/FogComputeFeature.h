#pragma once
#include "luth/renderer/features/RenderFeature.h"

namespace Luth
{
    class VolumetricSubsystem;
    struct FogComputeBindings;
    struct FrameDebugger;
    struct FogComputeBindingRef { const FogComputeBindings* native = nullptr; };
    namespace FogResources
    {
        inline constexpr ResourceKeyIdentity BindingsIdentity{"FogCompute.Bindings"};
        inline constexpr RenderResourceKey<FogComputeBindingRef> Bindings{&BindingsIdentity};
        inline constexpr ResourceKeyIdentity VolumesIdentity{"FogCompute.Volumes"};
        inline constexpr RenderResourceKey<GraphBufferRef> Volumes{&VolumesIdentity};
        inline constexpr ResourceKeyIdentity IntegratedIdentity{"FogCompute.IntegratedScatter"};
        inline constexpr RenderResourceKey<GraphTextureRef> IntegratedScatter{&IntegratedIdentity};
    }
    class FogComputeFeature final : public IRenderFeature
    {
    public:
        explicit FogComputeFeature(VolumetricSubsystem& native, FrameDebugger* debugger = nullptr)
            : m_Native(native), m_Debugger(debugger) {}
        FeatureInfo Describe() const override;
        FeatureFrameDecision Evaluate(const FeaturePrepareContext&) const override;
        void Build(RG::RenderGraph&, RenderFeatureContext&) override;
        // Shared native chain after the selected adapter validates its technique contract.
        void BuildPrepared(RG::RenderGraph&, RenderFeatureContext&, const FogComputeBindings&);
    private:
        VolumetricSubsystem& m_Native;
        FrameDebugger* m_Debugger;
    };
}

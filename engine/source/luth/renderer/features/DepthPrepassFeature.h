#pragma once

#include "luth/renderer/features/DeformationFeature.h"

namespace Luth
{
    class GeometrySubsystem;
    struct DepthPrepassBindings;
    struct FrameDebugger;
    // Borrowed native binding packet, prepared by the compatibility host. The feature
    // forwards it to its native adapter without exposing descriptor/layout operations.
    struct DepthPrepassBindingRef { const DepthPrepassBindings* native = nullptr; };
    namespace DepthPrepassResources
    {
        inline constexpr ResourceKeyIdentity TargetIdentity{"DepthPrepass.Target"};
        inline constexpr RenderResourceKey<GraphTextureRef> Target{&TargetIdentity};
        inline constexpr ResourceKeyIdentity BindingsIdentity{"DepthPrepass.Bindings"};
        inline constexpr RenderResourceKey<DepthPrepassBindingRef> Bindings{&BindingsIdentity};
    }
    class DepthPrepassFeature final : public IRenderFeature
    {
    public:
        explicit DepthPrepassFeature(GeometrySubsystem& native, FrameDebugger* debugger = nullptr)
            : m_Native(native), m_Debugger(debugger) {}
        FeatureInfo Describe() const override;
        FeatureFrameDecision Evaluate(const FeaturePrepareContext&) const override { return {}; }
        void Prepare(FeaturePrepareContext&) override;
        void Build(RG::RenderGraph&, RenderFeatureContext&) override;
    private:
        GeometrySubsystem& m_Native;
        FrameDebugger* m_Debugger;
    };
}

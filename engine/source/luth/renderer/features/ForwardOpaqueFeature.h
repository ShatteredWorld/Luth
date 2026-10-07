#pragma once
#include "luth/renderer/features/DeformationFeature.h"

namespace Luth
{
    class GeometrySubsystem;
    struct ForwardOpaqueBindings;
    struct FrameDebugger;
    struct ForwardOpaqueBindingRef { const ForwardOpaqueBindings* native = nullptr; };
    namespace ForwardOpaqueResources
    {
        inline constexpr ResourceKeyIdentity BindingsIdentity{"ForwardOpaque.Bindings"};
        inline constexpr RenderResourceKey<ForwardOpaqueBindingRef> Bindings{&BindingsIdentity};
        inline constexpr ResourceKeyIdentity ColorTargetIdentity{"ForwardOpaque.ColorTarget"};
        inline constexpr RenderResourceKey<GraphTextureRef> ColorTarget{&ColorTargetIdentity};
        inline constexpr ResourceKeyIdentity PickingTargetIdentity{"ForwardOpaque.PickingTarget"};
        inline constexpr RenderResourceKey<GraphTextureRef> PickingTarget{&PickingTargetIdentity};
    }
    class ForwardOpaqueFeature : public IRenderFeature
    {
    public:
        explicit ForwardOpaqueFeature(GeometrySubsystem& native, FrameDebugger* debugger = nullptr)
            : m_Native(native), m_Debugger(debugger) {}
        FeatureInfo Describe() const override;
        FeatureFrameDecision Evaluate(const FeaturePrepareContext&) const override { return {}; }
        void Build(RG::RenderGraph&, RenderFeatureContext&) override;
    protected:
        virtual void AppendCompatibilityReads(RenderFeatureContext&, std::vector<RG::ResourceHandle>&) const {}
    private:
        GeometrySubsystem& m_Native;
        FrameDebugger* m_Debugger;
    };
}

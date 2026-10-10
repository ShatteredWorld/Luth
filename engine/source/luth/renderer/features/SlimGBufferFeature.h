#pragma once

#include "luth/renderer/features/DeformationFeature.h"

namespace Luth
{
    class GeometrySubsystem;
    struct SlimGBufferBindings;
    struct FrameDebugger;
    struct SlimGBufferBindingRef { const SlimGBufferBindings* native = nullptr; };
    namespace SlimGBufferResources
    {
        inline constexpr ResourceKeyIdentity NormalTargetIdentity{"SlimGBuffer.NormalTarget"};
        inline constexpr RenderResourceKey<GraphTextureRef> NormalTarget{&NormalTargetIdentity};
        inline constexpr ResourceKeyIdentity RoughnessTargetIdentity{"SlimGBuffer.RoughnessTarget"};
        inline constexpr RenderResourceKey<GraphTextureRef> RoughnessTarget{&RoughnessTargetIdentity};
        inline constexpr ResourceKeyIdentity MotionTargetIdentity{"SlimGBuffer.MotionTarget"};
        inline constexpr RenderResourceKey<GraphTextureRef> MotionTarget{&MotionTargetIdentity};
        inline constexpr ResourceKeyIdentity MaterialTargetIdentity{"SlimGBuffer.MaterialTarget"};
        inline constexpr RenderResourceKey<GraphTextureRef> MaterialTarget{&MaterialTargetIdentity};
        inline constexpr ResourceKeyIdentity BindingsIdentity{"SlimGBuffer.Bindings"};
        inline constexpr RenderResourceKey<SlimGBufferBindingRef> Bindings{&BindingsIdentity};
    }
    class SlimGBufferFeature final : public IRenderFeature
    {
    public:
        explicit SlimGBufferFeature(GeometrySubsystem& native, FrameDebugger* debugger = nullptr)
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

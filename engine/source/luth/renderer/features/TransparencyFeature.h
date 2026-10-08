#pragma once
#include "luth/renderer/features/DeformationFeature.h"

namespace Luth
{
    class TransparencySubsystem;
    struct TransparencyBindings;
    struct FrameDebugger;
    struct TransparencyBindingRef { const TransparencyBindings* native = nullptr; };
    namespace TransparencyResources
    {
        inline constexpr ResourceKeyIdentity BindingsIdentity{"Transparency.Bindings"};
        inline constexpr RenderResourceKey<TransparencyBindingRef> Bindings{&BindingsIdentity};
        inline constexpr ResourceKeyIdentity DepthIdentity{"Transparency.Depth"};
        inline constexpr RenderResourceKey<GraphTextureRef> Depth{&DepthIdentity};
    }
    class TransparencyFeature final : public IRenderFeature
    {
    public:
        explicit TransparencyFeature(TransparencySubsystem& native, FrameDebugger* debugger = nullptr)
            : m_Native(native), m_Debugger(debugger) {}
        FeatureInfo Describe() const override;
        FeatureFrameDecision Evaluate(const FeaturePrepareContext&) const override { return {}; }
        void Build(RG::RenderGraph&, RenderFeatureContext&) override;
    private:
        TransparencySubsystem& m_Native;
        FrameDebugger* m_Debugger;
    };
}

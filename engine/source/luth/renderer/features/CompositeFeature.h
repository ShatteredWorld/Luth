#pragma once
#include "luth/renderer/features/RenderFeature.h"
#include "luth/renderer/features/CompositeBindings.h"

namespace Luth
{
    class PostProcessSubsystem;
    struct FrameDebugger;
    struct CompositeBindingRef { const CompositeBindings* native = nullptr; };
    namespace CompositeResources
    {
        inline constexpr ResourceKeyIdentity BindingsIdentity{"Composite.Bindings"};
        inline constexpr RenderResourceKey<CompositeBindingRef> Bindings{&BindingsIdentity};
    }
    class CompositeFeature final : public IRenderFeature
    {
    public:
        explicit CompositeFeature(PostProcessSubsystem& native, FrameDebugger* debugger = nullptr)
            : m_Native(native), m_Debugger(debugger) {}
        FeatureInfo Describe() const override;
        FeatureFrameDecision Evaluate(const FeaturePrepareContext&) const override { return {}; }
        void Build(RG::RenderGraph&, RenderFeatureContext&) override;
    private:
        PostProcessSubsystem& m_Native;
        FrameDebugger* m_Debugger;
    };
}

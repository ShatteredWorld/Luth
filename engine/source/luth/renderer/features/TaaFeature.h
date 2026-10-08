#pragma once
#include "luth/renderer/features/RenderFeature.h"
#include "luth/renderer/features/TaaBindings.h"

namespace Luth
{
    class PostProcessSubsystem;
    struct FrameDebugger;
    struct TaaBindingRef { const TaaBindings* native = nullptr; };
    namespace TaaResources
    {
        inline constexpr ResourceKeyIdentity BindingsIdentity{"TAA.Bindings"};
        inline constexpr RenderResourceKey<TaaBindingRef> Bindings{&BindingsIdentity};
    }
    class TaaFeature final : public IRenderFeature
    {
    public:
        explicit TaaFeature(PostProcessSubsystem& native, FrameDebugger* debugger = nullptr)
            : m_Native(native), m_Debugger(debugger) {}
        FeatureInfo Describe() const override;
        FeatureFrameDecision Evaluate(const FeaturePrepareContext&) const override { return {}; }
        void Build(RG::RenderGraph&, RenderFeatureContext&) override;
    private:
        PostProcessSubsystem& m_Native;
        FrameDebugger* m_Debugger;
    };
}

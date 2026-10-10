#pragma once
#include "luth/renderer/features/RenderFeature.h"
#include "luth/renderer/features/BloomBindings.h"

namespace Luth
{
    class PostProcessSubsystem;
    struct FrameDebugger;
    struct BloomBindingRef { const BloomBindings* native = nullptr; };
    namespace BloomResources
    {
        inline constexpr ResourceKeyIdentity BindingsIdentity{"Bloom.Bindings"};
        inline constexpr RenderResourceKey<BloomBindingRef> Bindings{&BindingsIdentity};
    }
    class BloomFeature final : public IRenderFeature
    {
    public:
        explicit BloomFeature(PostProcessSubsystem& native, FrameDebugger* debugger = nullptr)
            : m_Native(native), m_Debugger(debugger) {}
        FeatureInfo Describe() const override;
        FeatureFrameDecision Evaluate(const FeaturePrepareContext&) const override { return {}; }
        void Build(RG::RenderGraph&, RenderFeatureContext&) override;
    private:
        PostProcessSubsystem& m_Native;
        FrameDebugger* m_Debugger;
    };
}

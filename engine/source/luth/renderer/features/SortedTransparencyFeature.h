#pragma once
#include "luth/renderer/features/DeformationFeature.h"

namespace Luth
{
    class TransparencySubsystem;
    struct SortedTransparencyBindings;
    struct FrameDebugger;
    struct SortedTransparencyBindingRef { const SortedTransparencyBindings* native = nullptr; };
    namespace TransparencyResources
    {
        inline constexpr ResourceKeyIdentity BindingsIdentity{"Transparency.SortedBindings"};
        inline constexpr RenderResourceKey<SortedTransparencyBindingRef> Bindings{&BindingsIdentity};
        inline constexpr ResourceKeyIdentity DepthIdentity{"Transparency.Depth"};
        inline constexpr RenderResourceKey<GraphTextureRef> Depth{&DepthIdentity};
    }
    class SortedTransparencyFeature final : public IRenderFeature
    {
    public:
        explicit SortedTransparencyFeature(TransparencySubsystem& native, FrameDebugger* debugger = nullptr)
            : m_Native(native), m_Debugger(debugger) {}
        FeatureInfo Describe() const override;
        FeatureFrameDecision Evaluate(const FeaturePrepareContext&) const override { return {}; }
        void Build(RG::RenderGraph&, RenderFeatureContext&) override;
    private:
        TransparencySubsystem& m_Native;
        FrameDebugger* m_Debugger;
    };
}

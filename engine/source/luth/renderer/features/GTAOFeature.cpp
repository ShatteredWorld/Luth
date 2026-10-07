#include "luthpch.h"
#include "luth/renderer/features/GTAOFeature.h"
#include "luth/renderer/subsystems/GTAOSubsystem.h"
#include "luth/renderer/Renderer.h"
#include "luth/renderer/CameraParams.h"

namespace Luth
{
    GTAOFeature::GTAOFeature(GTAOSubsystem& native, GtaoViewStateStore& states, FrameDebugger* debugger)
        : m_Native(native), m_States(states), m_Debugger(debugger) {}

    FeatureInfo GTAOFeature::Describe() const
    {
        FeatureInfo info;
        info.name = "GTAO";
        info.phase = FeaturePhase::Async;
        info.activation = FeatureActivation::Conditional;
        info.resources.reads = {{RenderResources::SurfaceDepth}, {GtaoResources::Parameters}};
        info.resources.writes = {{RenderResources::AmbientOcclusion, ResourceOutputPresence::Optional}};
        return info;
    }
    FeatureFrameDecision GTAOFeature::Evaluate(const FeaturePrepareContext& ctx) const
    {
        for (const auto bindings : {ctx.frame.resources, ctx.view.resources})
            for (const auto& binding : bindings)
                if (const auto* params = binding.TryGet(GtaoResources::Parameters))
                    return {params->enabled && params->realtime};
        throw std::invalid_argument("GTAO: missing frozen parameters");
    }
    void GTAOFeature::Prepare(FeaturePrepareContext& ctx)
    {
        const auto* state = m_States.Find(ctx.view.id);
        // The compatibility host prepares native resources and the shared UBO before Build.
        // The store is domain-owned and never depends on the legacy aggregate's settings.
        if (!ctx.view.camera || !state || !*state || (*state)->width != ctx.view.width ||
            (*state)->height != ctx.view.height)
            throw std::invalid_argument("GTAO: native view state or camera was not prepared");
    }
    void GTAOFeature::Build(RG::RenderGraph& graph, RenderFeatureContext& ctx)
    {
        const auto& depth = ctx.resources.Get(RenderResources::SurfaceDepth);
        const auto& params = ctx.resources.Get(GtaoResources::Parameters);
        const auto& state = **m_States.Find(ctx.view.id);
        if (!depth.handle.IsValid() || !depth.binding.texture || depth.binding.texture != state.depthSource ||
            depth.binding.baseMip || depth.binding.mipCount != 1 ||
            depth.binding.baseLayer || depth.binding.layerCount != 1)
            throw std::invalid_argument("GTAO: depth contract does not match prepared native bindings");
        const auto linear = m_Native.AddPrefilterPass(graph, depth.handle, state, *ctx.view.camera, m_Debugger);
        const auto raw = m_Native.AddMainPass(graph, linear, state, *ctx.view.camera,
            ctx.frame.renderFrameIndex, params.shaderFrameIndex, m_Debugger);
        const auto final = m_Native.AddDenoisePass(graph, raw, linear, state, m_Debugger);
        ctx.resources.Publish(RenderResources::AmbientOcclusion, GraphTextureRef{final, state.finalBinding.binding});
    }
    void GTAOFeature::ReleaseView(RenderViewId id)
    {
        const auto* state = m_States.Find(id);
        const bool ownsNativeResources = state && *state && (*state)->pool;
        m_States.Release(id, [ownsNativeResources] {
            if (ownsNativeResources) Renderer::WaitForGPU();
        });
    }
}

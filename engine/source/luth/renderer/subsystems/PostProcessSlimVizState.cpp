#include "luthpch.h"
#include "luth/renderer/subsystems/PostProcessSubsystem.h"
#include "luth/renderer/FrameTargets.h"
#include "luth/renderer/Renderer.h"
#include <limits>
namespace Luth
{
    std::shared_ptr<SlimVizViewState> PostProcessSubsystem::EnsureSlimVizView(RenderViewId id, FrameTargets& targets)
    {
        const std::array sources{targets.GetSlimNormal(), targets.GetSlimRoughness(), targets.GetSlimMotion(), targets.GetSlimMaterialID()};
        for (const auto& source : sources) if (!source) throw std::invalid_argument("SlimViz: sampled source unavailable");
        const u32 width = sources[0]->GetWidth(), height = sources[0]->GetHeight();
        for (const auto& source : sources)
            if (source->GetWidth() != width || source->GetHeight() != height) throw std::invalid_argument("SlimViz: source extent mismatch");
        u64 generation = 1;
        if (const auto* prior = m_SlimVizStates.Find(id))
        {
            generation = (*prior)->sourceGeneration;
            if ((*prior)->sources != sources)
            {
                if (generation == std::numeric_limits<u64>::max()) throw std::runtime_error("SlimViz: source generation exhausted");
                ++generation;
            }
        }
        return m_SlimVizStates.Ensure(id, SlimVizViewState::Config(width, height, generation),
            [&](const ViewStateConfig& config) {
                auto state = SlimVizViewState::Create(id, config, m_SlimVizDescSetLayout);
                state->sources = sources; WriteSlimVizView(*state); return state;
            }, [] { Renderer::WaitForGPU(); });
    }
    void PostProcessSubsystem::ReleaseSlimVizView(RenderViewId id)
    {
        m_SlimVizStates.Release(id, [] { Renderer::WaitForGPU(); });
    }
}

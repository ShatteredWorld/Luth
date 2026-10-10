#include "luthpch.h"
#include "luth/renderer/subsystems/PostProcessSubsystem.h"
#include "luth/renderer/FrameTargets.h"
#include "luth/renderer/Renderer.h"
#include <limits>

namespace Luth
{
    std::shared_ptr<TaaViewState> PostProcessSubsystem::EnsureTaaView(RenderViewId id, FrameTargets& targets)
    {
        const std::array<std::shared_ptr<Texture>, 3> sources{
            targets.GetSceneColor(), targets.GetSlimMotion(), targets.GetSceneDepth()};
        for (const auto& source : sources)
            if (!source) throw std::invalid_argument("TAA: sampled source unavailable");
        u64 sourceGeneration = 1;
        if (const auto* prior = m_TaaStates.Find(id))
        {
            sourceGeneration = (*prior)->sourceGeneration;
            if ((*prior)->sources != sources)
            {
                if (sourceGeneration == std::numeric_limits<u64>::max())
                    throw std::runtime_error("TAA: source generation exhausted");
                ++sourceGeneration;
            }
        }
        const auto config = TaaViewState::Config(sources[0]->GetWidth(), sources[0]->GetHeight(), sourceGeneration);
        return m_TaaStates.Ensure(id, config, [&](const ViewStateConfig& requested) {
            auto state = TaaViewState::Create(id, requested, m_TaaResolveDescSetLayout);
            state->sources = sources;
            WriteTaaResolveView(*state, targets);
            return state;
        }, [] { Renderer::WaitForGPU(); });
    }
    void PostProcessSubsystem::ReleaseTaaView(RenderViewId id)
    {
        m_TaaStates.Release(id, [] { Renderer::WaitForGPU(); });
    }
}

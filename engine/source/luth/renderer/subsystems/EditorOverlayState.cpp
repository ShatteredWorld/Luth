#include "luthpch.h"
#include "luth/renderer/subsystems/EditorOverlaysSubsystem.h"
#include "luth/renderer/FrameTargets.h"
#include "luth/renderer/Renderer.h"
#include <limits>

namespace Luth
{
    std::shared_ptr<EditorOverlayViewState> EditorOverlaysSubsystem::EnsureView(RenderViewId id, FrameTargets& targets)
    {
        const std::array sources{targets.GetSelectionMask(), targets.GetSelectionDepth(), targets.GetSceneDepth()};
        for (const auto& source : sources)
            if (!source) throw std::invalid_argument("Editor overlays: sampled source unavailable");
        const u32 width = sources[2]->GetWidth(), height = sources[2]->GetHeight();
        for (const auto& source : sources)
            if (source->GetWidth() != width || source->GetHeight() != height)
                throw std::invalid_argument("Editor overlays: incompatible source extents");
        u64 generation = 1;
        if (const auto* prior = m_ViewStates.Find(id))
        {
            generation = (*prior)->sourceGeneration;
            if ((*prior)->sources != sources)
            {
                if (generation == std::numeric_limits<u64>::max())
                    throw std::runtime_error("Editor overlays: source generation exhausted");
                ++generation;
            }
        }
        return m_ViewStates.Ensure(id, EditorOverlayViewState::Config(width, height, generation),
            [&](const ViewStateConfig& config) {
                auto state = EditorOverlayViewState::Create(id, config, m_OutlineDescSetLayout, m_GridDescSetLayout);
                state->sources = sources;
                WriteOutlineView(*state); WriteGridView(*state);
                return state;
            }, [] { Renderer::WaitForGPU(); });
    }
    void EditorOverlaysSubsystem::ReleaseView(RenderViewId id)
    {
        m_ViewStates.Release(id, [] { Renderer::WaitForGPU(); });
    }
}

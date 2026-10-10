#include "luthpch.h"
#include "luth/renderer/subsystems/PostProcessSubsystem.h"
#include "luth/renderer/Renderer.h"

namespace Luth
{
    std::shared_ptr<BloomViewState> PostProcessSubsystem::EnsureBloomView(RenderViewId id, u32 width, u32 height)
    {
        return m_BloomStates.Ensure(id, BloomViewState::Config(width, height), [&](const ViewStateConfig& config) {
            auto state = BloomViewState::Create(id, config, m_BloomComputeLayout);
            WriteBloomView(*state);
            return state;
        }, [] { Renderer::WaitForGPU(); });
    }
    void PostProcessSubsystem::ReleaseBloomView(RenderViewId id)
    {
        m_BloomStates.Release(id, [] { Renderer::WaitForGPU(); });
    }
}

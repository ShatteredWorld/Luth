#include "luthpch.h"
#include "luth/renderer/subsystems/ReflectionsSubsystem.h"
#include "luth/renderer/FrameTargets.h"
#include "luth/renderer/Renderer.h"
#include "luth/renderer/backend/vulkan/VulkanTexture.h"

namespace Luth
{
    std::shared_ptr<ReflectionViewState> ReflectionsSubsystem::EnsureView(RenderViewId id,
        const FrameTargets& targets, bool half)
    {
        if (!m_SetLayout) return {};
        std::array<std::shared_ptr<Texture>, 3> sources{
            targets.GetSceneDepth(), targets.GetSlimNormal(), targets.GetSlimRoughness()};
        std::array<VkImageView, 3> views{};
        if (!sources[0]) throw std::invalid_argument("Reflections: missing depth source");
        const auto width = sources[0]->GetWidth(), height = sources[0]->GetHeight();
        for (u32 i = 0; i < sources.size(); ++i) {
            if (!sources[i] || sources[i]->GetWidth() != width || sources[i]->GetHeight() != height)
                throw std::invalid_argument("Reflections: incompatible sampled sources");
            views[i] = std::static_pointer_cast<VKTexture>(sources[i])->GetImageView();
            if (!views[i]) throw std::invalid_argument("Reflections: missing sampled image view");
        }
        const auto* prior = m_Views.Find(id);
        const u64 generation = prior && (*prior)->sources == sources && (*prior)->sourceViews == views
            ? (*prior)->sourceGeneration : m_NextSourceGeneration++;
        const auto config = ReflectionViewState::Config(width, height, half, generation);
        return m_Views.Ensure(id, config, [&](const ViewStateConfig& requested) {
            auto state = ReflectionViewState::Create(id, requested, m_SetLayout);
            state->sources = sources; state->sourceViews = views;
            WriteView(*state);
            return state;
        }, [] { Renderer::WaitForGPU(); });
    }
    void ReflectionsSubsystem::ReleaseView(RenderViewId id)
    {
        m_UpscaleViews.Release(id, [] { Renderer::WaitForGPU(); });
        m_Views.Release(id, [] { Renderer::WaitForGPU(); });
    }
}

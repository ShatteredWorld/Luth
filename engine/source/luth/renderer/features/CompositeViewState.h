#pragma once
#include "luth/core/FrameData.h"
#include "luth/renderer/features/RenderViewState.h"
#include "luth/renderer/resources/Texture.h"
#include <array>
#include <vulkan/vulkan.h>

namespace Luth
{
    struct CompositeViewState
    {
        CompositeViewState() = default;
        CompositeViewState(const CompositeViewState&) = delete;
        CompositeViewState& operator=(const CompositeViewState&) = delete;
        ~CompositeViewState();
        static ViewStateConfig Config(u32 width, u32 height, u64 sourceGeneration = 1);
        static std::shared_ptr<CompositeViewState> Create(RenderViewId, const ViewStateConfig&, VkDescriptorSetLayout);
        VkDevice device = VK_NULL_HANDLE;
        VkDescriptorPool pool = VK_NULL_HANDLE;
        std::array<VkDescriptorSet, MAX_FRAMES_IN_FLIGHT> sets{};
        // Retain the stable default HDR and bloom fallback images referenced by all slots.
        // The effective HDR source and tagged-heap UBO are rebound for the absolute render frame.
        std::array<std::shared_ptr<Texture>, 2> sources{};
        u64 sourceGeneration = 1;
    };
    using CompositeViewStateStore = FeatureViewStates<std::shared_ptr<CompositeViewState>>;
}

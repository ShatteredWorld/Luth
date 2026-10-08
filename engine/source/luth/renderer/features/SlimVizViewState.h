#pragma once
#include "luth/renderer/features/RenderViewState.h"
#include "luth/renderer/resources/Texture.h"
#include <array>
#include <vulkan/vulkan.h>
namespace Luth
{
    struct SlimVizViewState
    {
        SlimVizViewState() = default;
        SlimVizViewState(const SlimVizViewState&) = delete;
        SlimVizViewState& operator=(const SlimVizViewState&) = delete;
        ~SlimVizViewState();
        static ViewStateConfig Config(u32 width, u32 height, u64 sourceGeneration = 1);
        static std::shared_ptr<SlimVizViewState> Create(RenderViewId, const ViewStateConfig&, VkDescriptorSetLayout);
        VkDevice device = VK_NULL_HANDLE;
        VkDescriptorPool pool = VK_NULL_HANDLE;
        VkDescriptorSet set = VK_NULL_HANDLE;
        std::array<std::shared_ptr<Texture>, 4> sources{};
        u64 sourceGeneration = 1;
    };
    using SlimVizViewStateStore = FeatureViewStates<std::shared_ptr<SlimVizViewState>>;
}

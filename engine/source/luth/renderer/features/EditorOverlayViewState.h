#pragma once
#include "luth/core/FrameData.h"
#include "luth/renderer/features/RenderViewState.h"
#include "luth/renderer/resources/Texture.h"
#include <array>
#include <vulkan/vulkan.h>

namespace Luth
{
    struct EditorOverlayViewState
    {
        EditorOverlayViewState() = default;
        EditorOverlayViewState(const EditorOverlayViewState&) = delete;
        EditorOverlayViewState& operator=(const EditorOverlayViewState&) = delete;
        ~EditorOverlayViewState();
        static ViewStateConfig Config(u32 width, u32 height, u64 sourceGeneration = 1);
        static std::shared_ptr<EditorOverlayViewState> Create(RenderViewId, const ViewStateConfig&,
            VkDescriptorSetLayout outlineLayout, VkDescriptorSetLayout gridLayout);
        VkDevice device = VK_NULL_HANDLE;
        VkDescriptorPool pool = VK_NULL_HANDLE;
        VkDescriptorSet outlineSet = VK_NULL_HANDLE;
        std::array<VkDescriptorSet, MAX_FRAMES_IN_FLIGHT> gridSets{};
        // Retain FrameTargets' selection mask/depth and scene depth; preserve its public getters.
        std::array<std::shared_ptr<Texture>, 3> sources{};
        u64 sourceGeneration = 1;
    };
    using EditorOverlayViewStateStore = FeatureViewStates<std::shared_ptr<EditorOverlayViewState>>;
}

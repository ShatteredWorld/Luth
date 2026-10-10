#pragma once

#include "luth/renderer/features/RenderViewState.h"
#include "luth/core/FrameData.h"
#include "luth/renderer/features/RenderResource.h"
#include "luth/renderer/backend/vulkan/VulkanComputePipeline.h"
#include <array>

namespace Luth
{
    // Native image bindings are frozen during preparation. No graph handles survive here.
    struct GtaoImageBinding
    {
        VkImage image = VK_NULL_HANDLE;
        VkImageView view = VK_NULL_HANDLE;
        TextureBindingRef binding;
    };
    struct GtaoViewState
    {
        GtaoViewState() = default;
        ~GtaoViewState();
        GtaoViewState(const GtaoViewState&) = delete;
        GtaoViewState& operator=(const GtaoViewState&) = delete;
        u32 width = 0, height = 0, halfWidth = 0, halfHeight = 0;
        bool uniformEnabled = false; // Effective setting frozen with the shared UBO.
        const Texture* depthSource = nullptr;
        std::shared_ptr<Texture> linearDepth, rawAO, edges, finalAO;
        GtaoImageBinding linearBinding, rawBinding, finalBinding;
        VkDevice device = VK_NULL_HANDLE;
        VkDescriptorPool pool = VK_NULL_HANDLE;
        VkDescriptorSet prefilterSet = VK_NULL_HANDLE;
        std::array<VkDescriptorSet, MAX_FRAMES_IN_FLIGHT> mainSets{};
        VkDescriptorSet denoiseSet = VK_NULL_HANDLE;
    };
    using GtaoViewStateStore = FeatureViewStates<std::shared_ptr<GtaoViewState>>;
}

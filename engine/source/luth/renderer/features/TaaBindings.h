#pragma once
#include "luth/renderer/features/RenderResource.h"
#include "luth/renderer/features/TaaViewState.h"
#include "luth/core/types/LuthMath.h"
#include <cstddef>
#include <vulkan/vulkan.h>

namespace Luth
{
    struct TaaResolvePushConstants
    {
        Mat4 skyReproj{1.0f};
        f32 temporalAlpha = -1.0f;
    };
    static_assert(sizeof(TaaResolvePushConstants) == 68);
    static_assert(offsetof(TaaResolvePushConstants, temporalAlpha) == 64);
    struct TaaBindings
    {
        bool enabled = false;
        VkPipeline pipeline = VK_NULL_HANDLE;
        VkPipelineLayout layout = VK_NULL_HANDLE;
        VkDescriptorSet set = VK_NULL_HANDLE;
        std::array<TextureBindingRef, 3> sources{}; // HDR, motion, depth.
        TextureBindingRef previous, current;
        VkImage previousImage = VK_NULL_HANDLE, currentImage = VK_NULL_HANDLE;
        VkImageView previousView = VK_NULL_HANDLE, currentView = VK_NULL_HANDLE;
        u32 width = 0, height = 0;
        TaaResolvePushConstants constants;
        // Keep native resources alive for the recorded job; never retain graph handles.
        std::shared_ptr<TaaViewState> state;
    };
}

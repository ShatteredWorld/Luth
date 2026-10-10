#pragma once
#include "luth/renderer/features/RenderResource.h"
#include "luth/renderer/features/RenderViewState.h"
#include <vulkan/vulkan.h>

namespace Luth
{
    struct RtSunShadowBindings
    {
        VkPipeline pipeline = VK_NULL_HANDLE;
        VkPipelineLayout layout = VK_NULL_HANDLE;
        std::array<VkDescriptorSet, 5> sets{};
        VkImage image = VK_NULL_HANDLE;
        VkImageView imageView = VK_NULL_HANDLE;
        TextureBindingRef mask;
        const Texture* depthSource = nullptr;
        const Texture* normalSource = nullptr;
        VkAccelerationStructureKHR tlas = VK_NULL_HANDLE;
        VkDeviceAddress geometryTable = 0;
        RenderViewId view;
        u64 generation = 0, frameIndex = 0;
        u32 width = 0, height = 0;
    };
}

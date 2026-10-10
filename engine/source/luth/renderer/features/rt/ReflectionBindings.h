#pragma once
#include "luth/renderer/features/rt/ReflectionViewState.h"
#include "luth/renderer/features/RenderResource.h"
#include "luth/renderer/settings/ReflectionsSettings.h"
#include "luth/core/types/LuthMath.h"
#include "luth/memory/GPUTaggedPageAllocator.h"

namespace Luth
{
    struct ReflectionBindings
    {
        VkPipeline pipeline = VK_NULL_HANDLE;
        VkPipelineLayout layout = VK_NULL_HANDLE;
        std::array<VkDescriptorSet, 5> sets{};
        std::array<const Texture*, 3> sources{};
        std::array<VkImage, 3> sourceImages{};
        std::array<VkImageView, 3> sourceViews{};
        VkImage image = VK_NULL_HANDLE;
        VkImageView imageView = VK_NULL_HANDLE;
        TextureBindingRef output;
        Memory::GPUSubRegion lights{};
        std::shared_ptr<const ReflectionViewState> retained;
        Mat4 inverseViewProjection{1.0f};
        ReflectionsSettings settings;
        VkAccelerationStructureKHR tlas = VK_NULL_HANDLE;
        VkDeviceAddress geometryTable = 0;
        RenderViewId view;
        u64 generation = 0, frameIndex = 0;
        u32 width = 0, height = 0, fullWidth = 0, fullHeight = 0;
        bool environmentReady = false;
    };
}

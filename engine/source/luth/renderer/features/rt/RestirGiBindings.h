#pragma once
#include "luth/renderer/features/rt/RestirGiViewState.h"
#include "luth/renderer/features/RenderResource.h"
#include "luth/renderer/settings/RestirGiSettings.h"
#include "luth/core/types/LuthMath.h"
namespace Luth
{
    struct RestirGiBindings
    {
        std::array<VkPipeline, 4> pipelines{};
        std::array<VkPipelineLayout, 4> layouts{};
        std::array<VkDescriptorSet, 5> sets{};
        std::array<const Texture*, 3> sources{};
        std::array<VkImage, 3> sourceImages{};
        std::array<VkImageView, 3> sourceViews{};
        std::array<VkImage, 1> images{};
        std::array<VkImageView, 1> imageViews{};
        std::array<TextureBindingRef, 1> outputs{};
        Memory::GPUSubRegion scratch{}, spatial{}, lights{};
        std::shared_ptr<const RestirGiViewState> retained;
        Mat4 inverseViewProjection{1.0f};
        RestirGiSettings settings;
        VkAccelerationStructureKHR tlas = VK_NULL_HANDLE;
        VkDeviceAddress geometryTable = 0;
        RenderViewId view;
        u64 generation = 0, frameIndex = 0;
        u32 width = 0, height = 0, fullWidth = 0, fullHeight = 0;
    };
}

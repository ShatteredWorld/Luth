#pragma once
#include "luth/renderer/features/rt/RestirDiViewState.h"
#include "luth/renderer/features/RenderResource.h"
#include "luth/renderer/settings/RestirSettings.h"
#include "luth/core/types/LuthMath.h"
namespace Luth
{
    struct RestirDiBindings
    {
        std::array<VkPipeline, 4> pipelines{};
        std::array<VkPipelineLayout, 4> layouts{};
        std::array<VkDescriptorSet, 5> sets{};
        std::array<const Texture*, 4> sources{};
        std::array<VkImage, 4> sourceImages{};
        std::array<VkImageView, 4> sourceViews{};
        std::array<VkImage, 2> images{};
        std::array<VkImageView, 2> imageViews{};
        std::array<TextureBindingRef, 2> outputs{};
        Memory::GPUSubRegion scratch{}, spatial{}, lights{};
        std::shared_ptr<const RestirDiViewState> retained;
        Mat4 inverseViewProjection{1.0f};
        RestirSettings settings;
        bool historyValid = false;
        VkAccelerationStructureKHR tlas = VK_NULL_HANDLE;
        VkDeviceAddress geometryTable = 0;
        RenderViewId view;
        u64 generation = 0, frameIndex = 0;
        u32 width = 0, height = 0, fullWidth = 0, fullHeight = 0;
    };
}

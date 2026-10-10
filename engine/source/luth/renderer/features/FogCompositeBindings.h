#pragma once
#include "luth/renderer/features/RenderResource.h"
#include "luth/core/types/LuthMath.h"
#include <vulkan/vulkan.h>

namespace Luth
{
    struct FogCompositeBindings
    {
        VkPipeline pipeline = VK_NULL_HANDLE;
        VkPipelineLayout layout = VK_NULL_HANDLE;
        std::array<VkDescriptorSet, 2> sets{};
        Mat4 invView{1.0f};
        TextureBindingRef depth, resolved;
        bool enabled = false;
    };
}

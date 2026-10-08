#pragma once
#include "luth/renderer/features/RenderResource.h"
#include "luth/core/types/LuthMath.h"
#include "luth/renderer/resources/Texture.h"
#include <memory>
#include <vulkan/vulkan.h>
namespace Luth
{
    struct ClusterVizPushConstants { Vec2 viewport; float nearZ, farZ; };
    static_assert(sizeof(ClusterVizPushConstants) == 16);
    struct ClusterVizBindings
    {
        VkPipeline pipeline = VK_NULL_HANDLE;
        VkPipelineLayout layout = VK_NULL_HANDLE;
        std::array<VkDescriptorSet, 2> sets{};
        std::shared_ptr<Texture> depth;
        VkDescriptorBufferInfo grid{};
        ClusterVizPushConstants parameters{};
        bool enabled = false;
    };
}

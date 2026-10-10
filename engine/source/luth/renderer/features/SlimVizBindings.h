#pragma once
#include "luth/renderer/features/RenderResource.h"
#include "luth/renderer/features/SlimVizViewState.h"
#include <memory>
#include <vulkan/vulkan.h>
namespace Luth
{
    struct SlimVizPushConstants { u32 mode; float scale; };
    static_assert(sizeof(SlimVizPushConstants) == 8);
    struct SlimVizBindings
    {
        VkPipeline pipeline = VK_NULL_HANDLE;
        VkPipelineLayout layout = VK_NULL_HANDLE;
        VkDescriptorSet set = VK_NULL_HANDLE;
        std::shared_ptr<SlimVizViewState> state;
        SlimVizPushConstants parameters{};
        bool enabled = false;
    };
}

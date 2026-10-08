#pragma once
#include "luth/renderer/features/RenderResource.h"
#include "luth/renderer/resources/Texture.h"
#include "luth/memory/GPUTaggedPageAllocator.h"
#include <memory>
namespace Luth
{
    struct GiReservoirVizPushConstants { float vx, vy, resW, resH, mCap, ageCap; };
    static_assert(sizeof(GiReservoirVizPushConstants) == 24);
    struct GiReservoirVizBindings
    {
        VkPipeline pipeline = VK_NULL_HANDLE;
        VkPipelineLayout layout = VK_NULL_HANDLE;
        VkDescriptorSet set = VK_NULL_HANDLE;
        std::shared_ptr<Texture> depth;
        Memory::GPUSubRegion reservoir;
        GiReservoirVizPushConstants parameters{};
        bool enabled = false;
    };
}

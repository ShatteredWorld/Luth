#pragma once

#include "luth/renderer/features/RenderViewState.h"
#include <array>
#include <limits>
#include <stdexcept>
#include <vulkan/vulkan.h>

namespace Luth
{
    // Borrowed native bindings pinned to the captured frame. Frozen rendering does
    // not advance their heap tags; view generation validation precedes their use.
    struct CapturedReplayBindings
    {
        std::array<VkDescriptorSet, 6> sets{};
        VkBuffer indirectBuffer = VK_NULL_HANDLE;
        VkDeviceSize indirectOffset = 0, indirectSize = 0;
        u32 regionsPerView = 0, regionStride = 0;

        VkDeviceSize DrawOffset(u32 viewIndex, u32 region, u32 objectIndex) const
        {
            if (!indirectBuffer || !regionsPerView || !regionStride
                || region >= regionsPerView || objectIndex >= regionStride)
                throw std::invalid_argument("Invalid captured indirect draw binding");
            const u64 drawRegion = u64(viewIndex) * regionsPerView + region;
            if (drawRegion > (std::numeric_limits<u64>::max() - objectIndex) / regionStride)
                throw std::out_of_range("Captured indirect draw index overflow");
            const u64 index = drawRegion * regionStride + objectIndex;
            constexpr u64 bytes = sizeof(VkDrawIndexedIndirectCommand);
            if (index > std::numeric_limits<u64>::max() / bytes)
                throw std::out_of_range("Captured indirect draw offset overflow");
            const u64 relative = index * bytes;
            if (relative > indirectSize || bytes > indirectSize - relative
                || indirectOffset > std::numeric_limits<u64>::max() - relative)
                throw std::out_of_range("Captured indirect draw outside buffer slice");
            return indirectOffset + relative;
        }
    };
}

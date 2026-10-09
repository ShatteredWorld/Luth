#pragma once
#include "VulkanBarrierCapabilities.h"
#include <array>

namespace Luth
{
    // Native lighting ABI: raster prefix, followed by optional hybrid surface signals.
    struct VulkanLightBindings
    {
        std::array<VkDescriptorSetLayoutBinding, 9> bindings{};
        std::array<VkDescriptorBindingFlags, 9> flags{};
        uint32_t count = 4;
        explicit VulkanLightBindings(VulkanBarrierCapabilities capabilities = {})
        {
            if (capabilities.accelerationStructures) count = 9;
            for (uint32_t i = 0; i < count; ++i)
            {
                bindings[i] = {i, i < 3 ? VK_DESCRIPTOR_TYPE_STORAGE_BUFFER
                    : VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr};
                if (i < 3) flags[i] = VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT;
            }
            bindings[0].stageFlags |= VK_SHADER_STAGE_COMPUTE_BIT;
            if (count == 9 && capabilities.rayTracingPipelines)
                bindings[4].stageFlags |= VK_SHADER_STAGE_RAYGEN_BIT_KHR;
        }
        bool HasHybridSignals() const { return count == 9; }
    };
}

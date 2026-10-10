#pragma once
#include "VulkanBarrierCapabilities.h"
#include <array>

namespace Luth
{
    // Native Set 0 ABI. Raster and hybrid share bindings 0-5 and UBO bytes.
    struct VulkanGlobalBindings
    {
        std::array<VkDescriptorSetLayoutBinding, 7> bindings{};
        std::array<VkDescriptorBindingFlags, 7> flags{};
        uint32_t count = 6;
        explicit VulkanGlobalBindings(VulkanBarrierCapabilities capabilities = {})
        {
            const auto sharedStages = VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_COMPUTE_BIT;
            bindings[0] = {0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1,
                sharedStages | VK_SHADER_STAGE_VERTEX_BIT, nullptr};
            for (uint32_t i = 1; i <= 4; ++i)
                bindings[i] = {i, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, sharedStages, nullptr};
            bindings[5] = {5, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr};
            for (uint32_t i = 0; i < 6; ++i)
                flags[i] = VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT;
            if (capabilities.accelerationStructures)
            {
                count = 7;
                const VkShaderStageFlags pipelineStages = capabilities.rayTracingPipelines
                    ? VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR : 0;
                bindings[6] = {6, VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1,
                    sharedStages | pipelineStages, nullptr};
                flags[6] = VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT | VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT;
            }
        }
        bool HasTlasBinding() const { return count == 7; }
    };
}

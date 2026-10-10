#pragma once
#include <array>
#include <cstdint>
#include <vulkan/vulkan.h>

namespace Luth
{
    // Compatibility view pool. Independent feature pools are budgeted by their owners.
    struct VulkanViewPool
    {
        std::array<VkDescriptorPoolSize, 5> sizes{};
        uint32_t count = 3;
        uint32_t maxSets;

        VulkanViewPool(bool rt, uint32_t frames) : maxSets(4 * frames)
        {
            // Global: 2 UBOs + 4 samplers; lighting: 3 SSBOs + 1 sampler;
            // cluster build: 2 SSBOs; light assignment: 5 SSBOs, all cycled.
            sizes[0] = {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 2 * frames};
            sizes[1] = {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 5 * frames};
            sizes[2] = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 10 * frames};
            if (rt)
            {
                // Preserve the existing hybrid budget during domain migration.
                count = 5;
                maxSets = 205 - 15 * frames - 50;
                sizes[0] = {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 48 - 4 * frames};
                sizes[1] = {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 248 - 11 * frames - 118};
                sizes[2] = {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 317 - 32 * frames - 121};
                sizes[3] = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 126 - 11 * frames - 2};
                sizes[4] = {VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 8};
            }
        }
    };
}

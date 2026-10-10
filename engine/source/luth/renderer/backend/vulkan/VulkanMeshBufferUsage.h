#pragma once
#include <vulkan/vulkan.h>

namespace Luth::VulkanMeshBufferUsage
{
    // AS build input is an optional device usage, independent of raster BDA access.
    constexpr VkBufferUsageFlags BuildInput(bool accelerationStructuresEnabled)
    {
        return accelerationStructuresEnabled
            ? VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR : 0;
    }
    constexpr VkBufferUsageFlags Vertex(bool accelerationStructuresEnabled)
    {
        return VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT
            | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | BuildInput(accelerationStructuresEnabled);
    }
    constexpr VkBufferUsageFlags Index(bool accelerationStructuresEnabled)
    {
        return VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT
            | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | BuildInput(accelerationStructuresEnabled);
    }
    constexpr VkBufferUsageFlags Deformation(bool accelerationStructuresEnabled)
    {
        return VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT
            | BuildInput(accelerationStructuresEnabled);
    }
}

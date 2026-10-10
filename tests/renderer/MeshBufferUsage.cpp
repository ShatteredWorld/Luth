#include <doctest/doctest.h>
#include "luth/renderer/backend/vulkan/VulkanMeshBufferUsage.h"
#include <array>

using namespace Luth;

TEST_CASE("MeshBufferUsage: raster source and deformation allocations omit optional AS usage")
{
    const std::array raster{VulkanMeshBufferUsage::Vertex(false),
        VulkanMeshBufferUsage::Index(false), VulkanMeshBufferUsage::Deformation(false)};
    for (const auto usage : raster)
    {
        CHECK((usage & VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR) == 0);
        CHECK((usage & VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR) == 0);
        CHECK((usage & VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT) != 0);
    }
    CHECK((raster[0] & VK_BUFFER_USAGE_VERTEX_BUFFER_BIT) != 0);
    CHECK((raster[1] & VK_BUFFER_USAGE_INDEX_BUFFER_BIT) != 0);
    CHECK((raster[2] & VK_BUFFER_USAGE_STORAGE_BUFFER_BIT) != 0);
    CHECK((raster[0] & VK_BUFFER_USAGE_TRANSFER_DST_BIT) != 0);
    CHECK((raster[1] & VK_BUFFER_USAGE_TRANSFER_DST_BIT) != 0);
    CHECK((raster[2] & VK_BUFFER_USAGE_TRANSFER_DST_BIT) == 0);
}

TEST_CASE("MeshBufferUsage: hybrid allocations retain their existing build and raster usage")
{
    const std::array raster{VulkanMeshBufferUsage::Vertex(false),
        VulkanMeshBufferUsage::Index(false), VulkanMeshBufferUsage::Deformation(false)};
    const std::array hybrid{VulkanMeshBufferUsage::Vertex(true),
        VulkanMeshBufferUsage::Index(true), VulkanMeshBufferUsage::Deformation(true)};
    for (size_t i = 0; i < hybrid.size(); ++i)
    {
        CHECK((hybrid[i] & raster[i]) == raster[i]);
        CHECK((hybrid[i] ^ raster[i]) == VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR);
        CHECK((hybrid[i] & VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR) == 0);
    }
}

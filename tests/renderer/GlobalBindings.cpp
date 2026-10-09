#include <doctest/doctest.h>
#include "luth/renderer/backend/vulkan/VulkanGlobalBindings.h"
using namespace Luth;

TEST_CASE("GlobalBindings: raster omits TLAS while shared bindings retain the hybrid ABI")
{
    const VulkanGlobalBindings raster;
    const VulkanGlobalBindings hybrid({true, true});
    CHECK(raster.count == 6); CHECK_FALSE(raster.HasTlasBinding());
    CHECK(hybrid.count == 7); CHECK(hybrid.HasTlasBinding());
    for (uint32_t i = 0; i < raster.count; ++i)
    {
        CHECK(raster.bindings[i].binding == i);
        CHECK(raster.bindings[i].descriptorType == hybrid.bindings[i].descriptorType);
        CHECK(raster.bindings[i].descriptorCount == 1);
        CHECK(raster.bindings[i].stageFlags == hybrid.bindings[i].stageFlags);
        const VkShaderStageFlags expectedStages = i == 5 ? VK_SHADER_STAGE_FRAGMENT_BIT
            : VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_COMPUTE_BIT
                | (i == 0 ? VK_SHADER_STAGE_VERTEX_BIT : 0);
        CHECK(raster.bindings[i].stageFlags == expectedStages);
        CHECK(raster.bindings[i].pImmutableSamplers == nullptr);
        CHECK(raster.flags[i] == VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT);
        CHECK(raster.flags[i] == hybrid.flags[i]);
        CHECK((raster.bindings[i].stageFlags & (VK_SHADER_STAGE_RAYGEN_BIT_KHR
            | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR)) == 0);
        CHECK(raster.bindings[i].descriptorType != VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR);
    }
    CHECK(raster.bindings[0].descriptorType == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);
    CHECK(raster.bindings[5].descriptorType == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);
    for (uint32_t i = 1; i <= 4; ++i)
        CHECK(raster.bindings[i].descriptorType == VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
    CHECK(VulkanGlobalBindings::ViewPoolTypeCount(false) == 4);
    CHECK(VulkanGlobalBindings::ViewPoolTypeCount(true) == 5);
}

TEST_CASE("GlobalBindings: query-only TLAS visibility excludes RT pipeline stages")
{
    const VulkanGlobalBindings query({true, false}), full({true, true});
    REQUIRE(query.HasTlasBinding());
    CHECK(query.bindings[6].binding == 6);
    CHECK(query.bindings[6].descriptorType == VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR);
    CHECK(query.bindings[6].descriptorCount == 1);
    CHECK(query.bindings[6].stageFlags == (VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_COMPUTE_BIT));
    CHECK(query.flags[6] == (VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT | VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT));
    CHECK(full.bindings[6].stageFlags == (query.bindings[6].stageFlags | VK_SHADER_STAGE_RAYGEN_BIT_KHR
        | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR));
    CHECK(full.flags[6] == query.flags[6]);
}

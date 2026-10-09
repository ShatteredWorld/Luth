#include <doctest/doctest.h>
#include "luth/renderer/backend/vulkan/VulkanGlobalBindings.h"
#include "luth/renderer/backend/vulkan/VulkanLightBindings.h"
#include "luth/renderer/backend/vulkan/VulkanViewPool.h"
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

TEST_CASE("GlobalBindings: raster lighting prefix excludes hybrid surface signals")
{
    const Luth::VulkanLightBindings raster, hybrid({true, true}), query({true, false});
    REQUIRE(raster.count == 4); REQUIRE(hybrid.count == 9);
    CHECK_FALSE(raster.HasHybridSignals()); CHECK(hybrid.HasHybridSignals());
    for (uint32_t i = 0; i < 4; ++i)
    {
        CHECK(raster.bindings[i].binding == i);
        CHECK(raster.bindings[i].descriptorCount == 1);
        CHECK(raster.bindings[i].descriptorType == (i < 3 ? VK_DESCRIPTOR_TYPE_STORAGE_BUFFER : VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER));
        CHECK(raster.bindings[i].stageFlags == (VK_SHADER_STAGE_FRAGMENT_BIT | (i == 0 ? VK_SHADER_STAGE_COMPUTE_BIT : 0)));
        CHECK(raster.flags[i] == (i < 3 ? VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT : 0));
        CHECK(raster.bindings[i].descriptorType == hybrid.bindings[i].descriptorType);
        CHECK(raster.bindings[i].stageFlags == hybrid.bindings[i].stageFlags);
        CHECK(raster.flags[i] == hybrid.flags[i]);
    }
    for (uint32_t i = 4; i < 9; ++i)
    {
        CHECK(hybrid.bindings[i].binding == i);
        CHECK(hybrid.bindings[i].descriptorType == VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
        CHECK(hybrid.bindings[i].descriptorCount == 1);
        CHECK(hybrid.bindings[i].stageFlags == (VK_SHADER_STAGE_FRAGMENT_BIT | (i == 4 ? VK_SHADER_STAGE_RAYGEN_BIT_KHR : 0)));
        CHECK(query.bindings[i].stageFlags == VK_SHADER_STAGE_FRAGMENT_BIT);
        CHECK(hybrid.flags[i] == 0);
    }
}
TEST_CASE("GlobalBindings: raster view pool fits shared layouts without RT descriptor types")
{
    for (const uint32_t frames : {2u, 3u})
    {
        const VulkanViewPool raster(false, frames), hybrid(true, frames);
        const VulkanGlobalBindings global;
        const VulkanLightBindings light;
        REQUIRE(raster.count == 3);
        CHECK(raster.maxSets == 4 * frames);
        for (uint32_t i = 0; i < raster.count; ++i)
        {
            const auto& size = raster.sizes[i];
            CHECK(size.descriptorCount > 0);
            CHECK(size.type != VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);
            CHECK(size.type != VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR);
            uint32_t required = size.type == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER ? 7 : 0; // Cluster build + assignment.
            for (uint32_t b = 0; b < global.count; ++b)
                if (global.bindings[b].descriptorType == size.type) required += global.bindings[b].descriptorCount;
            for (uint32_t b = 0; b < light.count; ++b)
                if (light.bindings[b].descriptorType == size.type) required += light.bindings[b].descriptorCount;
            CHECK(size.descriptorCount == required * frames);
        }
        CHECK(hybrid.count == 5);
        CHECK(hybrid.maxSets == 205 - 14 * frames - 32);
        CHECK(hybrid.sizes[0].descriptorCount == 48 - 4 * frames);
        CHECK(hybrid.sizes[1].descriptorCount == 248 - 10 * frames - 65);
        CHECK(hybrid.sizes[2].descriptorCount == 317 - 29 * frames - 69);
        CHECK(hybrid.sizes[3].descriptorCount == 126 - 8 * frames - 1);
        CHECK(hybrid.sizes[4].type == VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR);
        CHECK(hybrid.sizes[4].descriptorCount == 8);
    }
}

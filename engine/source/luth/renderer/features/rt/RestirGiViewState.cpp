#include "luthpch.h"
#include "luth/renderer/features/rt/RestirGiViewState.h"
#include "luth/renderer/backend/vulkan/VulkanContext.h"
#include "luth/renderer/backend/vulkan/VulkanTexture.h"

namespace Luth
{
    ViewStateConfig RestirGiViewState::Config(u32 width, u32 height, bool half, u64 sourceGeneration)
    {
        if (!width || !height || !sourceGeneration)
            throw std::invalid_argument("ReSTIR GI: invalid extent or source generation");
        return {width, height, half ? 1u : 0u, sourceGeneration};
    }
    std::shared_ptr<RestirGiViewState> RestirGiViewState::Create(RenderViewId id,
        const ViewStateConfig& config, VkDescriptorSetLayout layout, VkDescriptorSetLayout vizLayout, u32 scratchTag, u32 spatialTag)
    {
        Config(config.width, config.height, config.signature != 0, config.resourceGeneration);
        if (!id.value || config.signature > 1 || scratchTag < 0xFFFF8000u || scratchTag == 0xFFFFFFFFu
            || spatialTag < 0xFFFF8000u || spatialTag == 0xFFFFFFFFu || scratchTag == spatialTag)
            throw std::invalid_argument("ReSTIR GI: invalid identity, configuration or reservoir tags");
        if (!layout || !vizLayout) throw std::runtime_error("ReSTIR GI: native layout unavailable");
        auto state = std::make_shared<RestirGiViewState>();
        state->id = id;
        const auto extent = WorkingExtent(config);
        state->width = extent[0]; state->height = extent[1];
        state->sourceGeneration = config.resourceGeneration;
        state->restirGiDI = std::make_shared<VKTexture>(state->width, state->height,
            TextureFormat::RGBA16F, 1, 0u, 1, VK_IMAGE_USAGE_STORAGE_BIT);
        const u64 bytes = static_cast<u64>(state->width) * state->height * 64u;
        state->restirGiReservoirTag = scratchTag;
        state->restirGiReservoir = Memory::GPUTaggedPageAllocator::Get().AllocateLargeTaggedDeviceLocal(scratchTag, bytes, 16);
        state->restirGiSpatialTag = spatialTag;
        state->restirGiSpatial = Memory::GPUTaggedPageAllocator::Get().AllocateLargeTaggedDeviceLocal(spatialTag, bytes, 16);
        if (!state->restirGiReservoir.buffer || !state->restirGiSpatial.buffer)
            throw std::runtime_error("ReSTIR GI: reservoir allocation failed");
        state->device = VulkanContext::Get().GetDevice();
        const RestirGiPoolBudget budget(MAX_FRAMES_IN_FLIGHT);
        VkDescriptorPoolCreateInfo pool{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        pool.flags = VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT;
        pool.maxSets = budget.maxSets;
        pool.poolSizeCount = static_cast<u32>(budget.sizes.size()); pool.pPoolSizes = budget.sizes.data();
        if (vkCreateDescriptorPool(state->device, &pool, nullptr, &state->pool) != VK_SUCCESS)
            throw std::runtime_error("ReSTIR GI: descriptor pool allocation failed");
        std::array<VkDescriptorSetLayout, MAX_FRAMES_IN_FLIGHT> layouts; layouts.fill(layout);
        VkDescriptorSetAllocateInfo alloc{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        alloc.descriptorPool = state->pool; alloc.descriptorSetCount = MAX_FRAMES_IN_FLIGHT;
        alloc.pSetLayouts = layouts.data();
        if (vkAllocateDescriptorSets(state->device, &alloc, state->restirGiDescSet.data()) != VK_SUCCESS)
            throw std::runtime_error("ReSTIR GI: descriptor allocation failed");
        alloc.descriptorSetCount = 1; alloc.pSetLayouts = &vizLayout;
        if (vkAllocateDescriptorSets(state->device, &alloc, &state->giReservoirVizDescSet) != VK_SUCCESS)
            throw std::runtime_error("ReSTIR GI: visualization descriptor allocation failed");
        for (u32 slot = 0; slot < MAX_FRAMES_IN_FLIGHT; ++slot) {
            const auto name = "RestirGI.View" + std::to_string(id.value) + ".Slot" + std::to_string(slot);
            VulkanContext::SetDebugName(state->restirGiDescSet[slot], name.c_str());
        }
        return state;
    }
    RestirGiViewState::~RestirGiViewState()
    {
        if (pool) VulkanContext::Get().PushDeletion([device = device, pool = pool] { vkDestroyDescriptorPool(device, pool, nullptr); });
        if (restirGiReservoirTag) Memory::GPUTaggedPageAllocator::Get().FreeTagAndDestroy(restirGiReservoirTag);
        if (restirGiSpatialTag) Memory::GPUTaggedPageAllocator::Get().FreeTagAndDestroy(restirGiSpatialTag);
    }
}

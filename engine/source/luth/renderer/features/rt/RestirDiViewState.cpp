#include "luthpch.h"
#include "luth/renderer/features/rt/RestirDiViewState.h"
#include "luth/renderer/backend/vulkan/VulkanContext.h"
#include "luth/renderer/backend/vulkan/VulkanTexture.h"

namespace Luth
{
    ViewStateConfig RestirDiViewState::Config(u32 width, u32 height, bool half, u64 sourceGeneration)
    {
        if (!width || !height || !sourceGeneration)
            throw std::invalid_argument("ReSTIR DI: invalid extent or source generation");
        return {width, height, half ? 1u : 0u, sourceGeneration};
    }
    std::shared_ptr<RestirDiViewState> RestirDiViewState::Create(RenderViewId id,
        const ViewStateConfig& config, VkDescriptorSetLayout layout, u32 scratchTag, u32 spatialTag)
    {
        Config(config.width, config.height, config.signature != 0, config.resourceGeneration);
        if (!id.value || config.signature > 1 || scratchTag < 0xFFFF0000u || scratchTag >= 0xFFFF8000u
            || spatialTag < 0xFFFF0000u || spatialTag >= 0xFFFF8000u || scratchTag == spatialTag)
            throw std::invalid_argument("ReSTIR DI: invalid identity, configuration or reservoir tags");
        if (!layout) throw std::runtime_error("ReSTIR DI: native layout unavailable");
        auto state = std::make_shared<RestirDiViewState>();
        state->id = id;
        const auto extent = WorkingExtent(config);
        state->width = extent[0]; state->height = extent[1];
        state->sourceGeneration = config.resourceGeneration;
        state->restirDI = std::make_shared<VKTexture>(state->width, state->height,
            TextureFormat::RGBA16F, 1, 0u, 1, VK_IMAGE_USAGE_STORAGE_BIT);
        state->restirDISpec = std::make_shared<VKTexture>(state->width, state->height,
            TextureFormat::RGBA16F, 1, 0u, 1, VK_IMAGE_USAGE_STORAGE_BIT);
        const u64 bytes = static_cast<u64>(state->width) * state->height * 32u;
        state->restirReservoirTag = scratchTag;
        state->restirReservoir = Memory::GPUTaggedPageAllocator::Get().AllocateLargeTaggedDeviceLocal(scratchTag, bytes, 16);
        state->restirSpatialTag = spatialTag;
        state->restirSpatial = Memory::GPUTaggedPageAllocator::Get().AllocateLargeTaggedDeviceLocal(spatialTag, bytes, 16);
        if (!state->restirReservoir.buffer || !state->restirSpatial.buffer)
            throw std::runtime_error("ReSTIR DI: reservoir allocation failed");
        state->device = VulkanContext::Get().GetDevice();
        const RestirDiPoolBudget budget(MAX_FRAMES_IN_FLIGHT);
        VkDescriptorPoolCreateInfo pool{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        pool.flags = VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT;
        pool.maxSets = budget.maxSets;
        pool.poolSizeCount = static_cast<u32>(budget.sizes.size()); pool.pPoolSizes = budget.sizes.data();
        if (vkCreateDescriptorPool(state->device, &pool, nullptr, &state->pool) != VK_SUCCESS)
            throw std::runtime_error("ReSTIR DI: descriptor pool allocation failed");
        std::array<VkDescriptorSetLayout, MAX_FRAMES_IN_FLIGHT> layouts; layouts.fill(layout);
        VkDescriptorSetAllocateInfo alloc{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        alloc.descriptorPool = state->pool; alloc.descriptorSetCount = MAX_FRAMES_IN_FLIGHT;
        alloc.pSetLayouts = layouts.data();
        if (vkAllocateDescriptorSets(state->device, &alloc, state->restirDescSet.data()) != VK_SUCCESS)
            throw std::runtime_error("ReSTIR DI: descriptor allocation failed");
        for (u32 slot = 0; slot < MAX_FRAMES_IN_FLIGHT; ++slot) {
            const auto name = "RestirDI.View" + std::to_string(id.value) + ".Slot" + std::to_string(slot);
            VulkanContext::SetDebugName(state->restirDescSet[slot], name.c_str());
        }
        return state;
    }
    RestirDiViewState::~RestirDiViewState()
    {
        if (pool) VulkanContext::Get().PushDeletion([device = device, pool = pool] { vkDestroyDescriptorPool(device, pool, nullptr); });
        if (restirReservoirTag) Memory::GPUTaggedPageAllocator::Get().FreeTagAndDestroy(restirReservoirTag);
        if (restirSpatialTag) Memory::GPUTaggedPageAllocator::Get().FreeTagAndDestroy(restirSpatialTag);
    }
}

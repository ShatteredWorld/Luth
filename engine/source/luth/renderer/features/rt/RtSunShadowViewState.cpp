#include "luthpch.h"
#include "luth/renderer/features/rt/RtSunShadowViewState.h"
#include "luth/renderer/backend/vulkan/VulkanContext.h"
#include "luth/renderer/backend/vulkan/VulkanTexture.h"

namespace Luth
{
    ViewStateConfig RtSunShadowViewState::Config(u32 width, u32 height, u64 depthIdentity, u64 normalIdentity)
    {
        if (!width || !height || !depthIdentity || !normalIdentity)
            throw std::invalid_argument("RtSunShadow: invalid extent or source identities");
        return {width, height, depthIdentity, normalIdentity};
    }
    std::shared_ptr<RtSunShadowViewState> RtSunShadowViewState::Create(RenderViewId id,
        const ViewStateConfig& config, VkDescriptorSetLayout layout)
    {
        Config(config.width, config.height, config.signature, config.resourceGeneration);
        if (!id.value) throw std::invalid_argument("RtSunShadow: invalid view identity");
        if (!layout) throw std::runtime_error("RtSunShadow: native layout unavailable");
        auto state = std::make_shared<RtSunShadowViewState>();
        state->id = id; state->width = config.width; state->height = config.height;
        state->mask = std::make_shared<VKTexture>(config.width, config.height, TextureFormat::R8,
            1, 0u, 1, VK_IMAGE_USAGE_STORAGE_BIT);
        state->device = VulkanContext::Get().GetDevice();
        const RtSunShadowPoolBudget budget(MAX_FRAMES_IN_FLIGHT);
        VkDescriptorPoolCreateInfo pool{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        pool.flags = VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT;
        pool.maxSets = budget.maxSets;
        pool.poolSizeCount = static_cast<u32>(budget.sizes.size());
        pool.pPoolSizes = budget.sizes.data();
        if (vkCreateDescriptorPool(state->device, &pool, nullptr, &state->pool) != VK_SUCCESS)
            throw std::runtime_error("RtSunShadow: descriptor pool allocation failed");
        std::array<VkDescriptorSetLayout, MAX_FRAMES_IN_FLIGHT> layouts;
        layouts.fill(layout);
        VkDescriptorSetAllocateInfo alloc{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        alloc.descriptorPool = state->pool; alloc.descriptorSetCount = MAX_FRAMES_IN_FLIGHT;
        alloc.pSetLayouts = layouts.data();
        if (vkAllocateDescriptorSets(state->device, &alloc, state->sets.data()) != VK_SUCCESS)
            throw std::runtime_error("RtSunShadow: descriptor allocation failed");
        for (u32 slot = 0; slot < MAX_FRAMES_IN_FLIGHT; ++slot) {
            const auto name = "RtShadow.View" + std::to_string(id.value) + ".Slot" + std::to_string(slot);
            VulkanContext::SetDebugName(state->sets[slot], name.c_str());
        }
        return state;
    }
    RtSunShadowViewState::~RtSunShadowViewState()
    {
        if (pool) VulkanContext::Get().PushDeletion([device = device, pool = pool] { vkDestroyDescriptorPool(device, pool, nullptr); });
        mask.reset(); depthSource.reset(); normalSource.reset();
    }
}

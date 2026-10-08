#include "luthpch.h"
#include "luth/renderer/features/TaaViewState.h"
#include "luth/renderer/backend/vulkan/VulkanContext.h"

namespace Luth
{
    ViewStateConfig TaaViewState::Config(u32 width, u32 height, u64 sourceGeneration)
    {
        if (!width || !height || !sourceGeneration)
            throw std::invalid_argument("TAA: invalid extent or source generation");
        return {width, height, sourceGeneration};
    }
    std::shared_ptr<TaaViewState> TaaViewState::Create(RenderViewId id,
        const ViewStateConfig& config, VkDescriptorSetLayout layout)
    {
        Config(config.width, config.height, config.signature);
        if (!id.value) throw std::invalid_argument("TAA: invalid view identity");
        if (!layout) throw std::runtime_error("TAA: native layout unavailable");
        auto state = std::make_shared<TaaViewState>();
        state->sourceGeneration = config.signature;
        state->historyA = Texture::Create(config.width, config.height, TextureFormat::RGBA16F);
        state->historyB = Texture::Create(config.width, config.height, TextureFormat::RGBA16F);
        state->device = VulkanContext::Get().GetDevice();
        VkDescriptorPoolSize sizes[] = {
            {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 4 * MAX_FRAMES_IN_FLIGHT},
            {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, MAX_FRAMES_IN_FLIGHT}
        };
        VkDescriptorPoolCreateInfo pool{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        pool.flags = VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT;
        pool.maxSets = MAX_FRAMES_IN_FLIGHT; pool.poolSizeCount = 2; pool.pPoolSizes = sizes;
        if (vkCreateDescriptorPool(state->device, &pool, nullptr, &state->pool) != VK_SUCCESS)
            throw std::runtime_error("TAA: descriptor pool allocation failed");
        std::array<VkDescriptorSetLayout, MAX_FRAMES_IN_FLIGHT> layouts;
        layouts.fill(layout);
        VkDescriptorSetAllocateInfo alloc{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        alloc.descriptorPool = state->pool; alloc.descriptorSetCount = MAX_FRAMES_IN_FLIGHT;
        alloc.pSetLayouts = layouts.data();
        if (vkAllocateDescriptorSets(state->device, &alloc, state->resolveSets.data()) != VK_SUCCESS)
            throw std::runtime_error("TAA: descriptor allocation failed");
        for (u32 slot = 0; slot < MAX_FRAMES_IN_FLIGHT; ++slot)
        {
            const auto name = "TAA.View" + std::to_string(id.value) + ".Slot" + std::to_string(slot);
            VulkanContext::SetDebugName(state->resolveSets[slot], name.c_str());
        }
        return state;
    }
    TaaViewState::~TaaViewState()
    {
        if (pool)
            VulkanContext::Get().PushDeletion([device = device, pool = pool] { vkDestroyDescriptorPool(device, pool, nullptr); });
    }
}

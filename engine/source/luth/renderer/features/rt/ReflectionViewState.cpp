#include "luthpch.h"
#include "luth/renderer/features/rt/ReflectionViewState.h"
#include "luth/renderer/backend/vulkan/VulkanContext.h"
#include "luth/renderer/backend/vulkan/VulkanTexture.h"

namespace Luth
{
    ViewStateConfig ReflectionViewState::Config(u32 width, u32 height, bool half, u64 generation)
    {
        if (!width || !height || !generation)
            throw std::invalid_argument("Reflections: invalid extent or source generation");
        return {width, height, half ? 1u : 0u, generation};
    }
    std::array<u32, 2> ReflectionViewState::WorkingExtent(const ViewStateConfig& config)
    {
        Config(config.width, config.height, config.signature != 0, config.resourceGeneration);
        if (config.signature > 1) throw std::invalid_argument("Reflections: invalid resolution mode");
        return {config.signature ? std::max(config.width / 2, 1u) : config.width,
                config.signature ? std::max(config.height / 2, 1u) : config.height};
    }
    std::shared_ptr<ReflectionViewState> ReflectionViewState::Create(RenderViewId id,
        const ViewStateConfig& config, VkDescriptorSetLayout layout)
    {
        const auto extent = WorkingExtent(config);
        if (!id.value) throw std::invalid_argument("Reflections: invalid view identity");
        if (!layout) throw std::runtime_error("Reflections: native layout unavailable");
        auto state = std::make_shared<ReflectionViewState>();
        state->id = id; state->width = extent[0]; state->height = extent[1];
        state->sourceGeneration = config.resourceGeneration;
        // Fully overwritten by RtReflections, including fallback pixels; no history clear.
        state->radiance = std::make_shared<VKTexture>(state->width, state->height,
            TextureFormat::RGBA16F, 1, 0u, 1, VK_IMAGE_USAGE_STORAGE_BIT);
        state->device = VulkanContext::Get().GetDevice();
        const ReflectionPoolBudget budget;
        VkDescriptorPoolCreateInfo pool{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        pool.maxSets = budget.maxSets;
        pool.poolSizeCount = static_cast<u32>(budget.sizes.size()); pool.pPoolSizes = budget.sizes.data();
        if (vkCreateDescriptorPool(state->device, &pool, nullptr, &state->pool) != VK_SUCCESS)
            throw std::runtime_error("Reflections: descriptor pool allocation failed");
        VkDescriptorSetAllocateInfo alloc{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        alloc.descriptorPool = state->pool; alloc.descriptorSetCount = 1; alloc.pSetLayouts = &layout;
        if (vkAllocateDescriptorSets(state->device, &alloc, &state->descriptorSet) != VK_SUCCESS)
            throw std::runtime_error("Reflections: descriptor allocation failed");
        const auto name = "Reflections.View" + std::to_string(id.value);
        VulkanContext::SetDebugName(state->descriptorSet, name.c_str());
        return state;
    }
    ReflectionViewState::~ReflectionViewState()
    {
        if (pool) VulkanContext::Get().PushDeletion([device = device, pool = pool] {
            vkDestroyDescriptorPool(device, pool, nullptr);
        });
    }
}

#include "luthpch.h"
#include "luth/renderer/features/EditorOverlayViewState.h"
#include "luth/renderer/backend/vulkan/VulkanContext.h"

namespace Luth
{
    ViewStateConfig EditorOverlayViewState::Config(u32 width, u32 height, u64 generation)
    {
        if (!width || !height || !generation)
            throw std::invalid_argument("Editor overlays: invalid extent or source generation");
        return {width, height, generation};
    }
    std::shared_ptr<EditorOverlayViewState> EditorOverlayViewState::Create(RenderViewId id,
        const ViewStateConfig& config, VkDescriptorSetLayout outlineLayout, VkDescriptorSetLayout gridLayout)
    {
        Config(config.width, config.height, config.signature);
        if (!id.value) throw std::invalid_argument("Editor overlays: invalid view identity");
        if (!outlineLayout || !gridLayout) throw std::runtime_error("Editor overlays: native layouts unavailable");
        auto state = std::make_shared<EditorOverlayViewState>();
        state->sourceGeneration = config.signature;
        state->device = VulkanContext::Get().GetDevice();
        VkDescriptorPoolSize sizes[] = {
            {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 3 + MAX_FRAMES_IN_FLIGHT},
            {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, MAX_FRAMES_IN_FLIGHT}
        };
        VkDescriptorPoolCreateInfo pool{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        pool.flags = VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT;
        pool.maxSets = 1 + MAX_FRAMES_IN_FLIGHT; pool.poolSizeCount = 2; pool.pPoolSizes = sizes;
        if (vkCreateDescriptorPool(state->device, &pool, nullptr, &state->pool) != VK_SUCCESS)
            throw std::runtime_error("Editor overlays: descriptor pool allocation failed");
        std::array<VkDescriptorSetLayout, 1 + MAX_FRAMES_IN_FLIGHT> layouts;
        layouts.fill(gridLayout); layouts[0] = outlineLayout;
        std::array<VkDescriptorSet, 1 + MAX_FRAMES_IN_FLIGHT> sets{};
        VkDescriptorSetAllocateInfo alloc{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        alloc.descriptorPool = state->pool; alloc.descriptorSetCount = (u32)sets.size(); alloc.pSetLayouts = layouts.data();
        if (vkAllocateDescriptorSets(state->device, &alloc, sets.data()) != VK_SUCCESS)
            throw std::runtime_error("Editor overlays: descriptor allocation failed");
        state->outlineSet = sets[0]; std::copy_n(sets.begin() + 1, MAX_FRAMES_IN_FLIGHT, state->gridSets.begin());
        const auto prefix = "EditorOverlays.View" + std::to_string(id.value);
        VulkanContext::SetDebugName(state->outlineSet, (prefix + ".Outline").c_str());
        for (u32 slot = 0; slot < MAX_FRAMES_IN_FLIGHT; ++slot)
            VulkanContext::SetDebugName(state->gridSets[slot], (prefix + ".Grid.Slot" + std::to_string(slot)).c_str());
        return state;
    }
    EditorOverlayViewState::~EditorOverlayViewState()
    {
        if (pool)
            VulkanContext::Get().PushDeletion([device = device, pool = pool] { vkDestroyDescriptorPool(device, pool, nullptr); });
    }
}

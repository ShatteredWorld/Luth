#include "luthpch.h"
#include "luth/renderer/features/SlimVizViewState.h"
#include "luth/renderer/backend/vulkan/VulkanContext.h"
namespace Luth
{
    ViewStateConfig SlimVizViewState::Config(u32 width, u32 height, u64 generation)
    {
        if (!width || !height || !generation) throw std::invalid_argument("SlimViz: invalid extent or generation");
        return {width, height, generation};
    }
    std::shared_ptr<SlimVizViewState> SlimVizViewState::Create(RenderViewId id, const ViewStateConfig& config, VkDescriptorSetLayout layout)
    {
        Config(config.width, config.height, config.signature);
        if (!id.value) throw std::invalid_argument("SlimViz: invalid view identity");
        if (!layout) throw std::runtime_error("SlimViz: native layout unavailable");
        auto state = std::make_shared<SlimVizViewState>(); state->sourceGeneration = config.signature;
        state->device = VulkanContext::Get().GetDevice();
        VkDescriptorPoolSize size{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 4};
        VkDescriptorPoolCreateInfo pool{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        pool.maxSets = 1; pool.poolSizeCount = 1; pool.pPoolSizes = &size;
        if (vkCreateDescriptorPool(state->device, &pool, nullptr, &state->pool) != VK_SUCCESS)
            throw std::runtime_error("SlimViz: descriptor pool allocation failed");
        VkDescriptorSetAllocateInfo alloc{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        alloc.descriptorPool = state->pool; alloc.descriptorSetCount = 1; alloc.pSetLayouts = &layout;
        if (vkAllocateDescriptorSets(state->device, &alloc, &state->set) != VK_SUCCESS)
            throw std::runtime_error("SlimViz: descriptor allocation failed");
        const auto name = "SlimViz.View" + std::to_string(id.value);
        VulkanContext::SetDebugName(state->set, name.c_str()); return state;
    }
    SlimVizViewState::~SlimVizViewState()
    {
        if (pool) VulkanContext::Get().PushDeletion([device = device, pool = pool] { vkDestroyDescriptorPool(device, pool, nullptr); });
    }
}

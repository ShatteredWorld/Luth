#include "luthpch.h"
#include "luth/renderer/features/ClusterVizViewState.h"
#include "luth/renderer/backend/vulkan/VulkanContext.h"
namespace Luth
{
    ViewStateConfig ClusterVizViewState::Config(u32 width, u32 height, u64 generation)
    {
        if (!width || !height || !generation) throw std::invalid_argument("ClusterViz: invalid extent or generation");
        return {width, height, generation};
    }
    std::shared_ptr<ClusterVizViewState> ClusterVizViewState::Create(RenderViewId id, const ViewStateConfig& config, VkDescriptorSetLayout layout)
    {
        Config(config.width, config.height, config.signature);
        if (!id.value) throw std::invalid_argument("ClusterViz: invalid view identity");
        if (!layout) throw std::runtime_error("ClusterViz: native layout unavailable");
        auto state = std::make_shared<ClusterVizViewState>(); state->sourceGeneration = config.signature;
        state->device = VulkanContext::Get().GetDevice();
        VkDescriptorPoolSize size{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1};
        VkDescriptorPoolCreateInfo pool{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        pool.maxSets = 1; pool.poolSizeCount = 1; pool.pPoolSizes = &size;
        if (vkCreateDescriptorPool(state->device, &pool, nullptr, &state->pool) != VK_SUCCESS)
            throw std::runtime_error("ClusterViz: descriptor pool allocation failed");
        VkDescriptorSetAllocateInfo alloc{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        alloc.descriptorPool = state->pool; alloc.descriptorSetCount = 1; alloc.pSetLayouts = &layout;
        if (vkAllocateDescriptorSets(state->device, &alloc, &state->set) != VK_SUCCESS)
            throw std::runtime_error("ClusterViz: descriptor allocation failed");
        const auto name = "ClusterViz.View" + std::to_string(id.value);
        VulkanContext::SetDebugName(state->set, name.c_str()); return state;
    }
    ClusterVizViewState::~ClusterVizViewState()
    {
        if (pool) VulkanContext::Get().PushDeletion([device = device, pool = pool] { vkDestroyDescriptorPool(device, pool, nullptr); });
    }
}

#include "luthpch.h"
#include "luth/renderer/features/BloomViewState.h"
#include "luth/renderer/backend/vulkan/VulkanContext.h"
#include "luth/renderer/backend/vulkan/VulkanTexture.h"

namespace Luth
{
    ViewStateConfig BloomViewState::Config(u32 width, u32 height)
    {
        if (!width || !height) throw std::invalid_argument("Bloom: invalid view extent");
        return {width, height, kMipCount};
    }
    std::pair<u32, u32> BloomViewState::MipExtent(const ViewStateConfig& config, u32 mip)
    {
        Config(config.width, config.height);
        if (config.signature != kMipCount || mip >= kMipCount)
            throw std::invalid_argument("Bloom: invalid pyramid configuration or mip");
        return {std::max((config.width / 2) >> mip, 1u), std::max((config.height / 2) >> mip, 1u)};
    }
    std::shared_ptr<BloomViewState> BloomViewState::Create(RenderViewId id,
        const ViewStateConfig& config, VkDescriptorSetLayout layout)
    {
        MipExtent(config, 0);
        if (!id.value) throw std::invalid_argument("Bloom: invalid view identity");
        if (!layout) throw std::runtime_error("Bloom: native layout unavailable");
        auto state = std::make_shared<BloomViewState>();
        state->width = config.width; state->height = config.height;
        for (u32 mip = 0; mip < kMipCount; ++mip)
        {
            const auto [width, height] = MipExtent(config, mip);
            // Preserve SAMPLED/COLOR usage and the shader-read bootstrap from VKTexture,
            // adding STORAGE for the existing compute pyramid's imageStore operations.
            state->mips[mip] = std::make_shared<VKTexture>(width, height, TextureFormat::RGBA16F,
                1, 0u, 1, VK_IMAGE_USAGE_STORAGE_BIT);
        }
        state->device = VulkanContext::Get().GetDevice();
        VkDescriptorPoolSize sizes[] = {
            {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, kSetCount},
            {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, kSetCount}
        };
        VkDescriptorPoolCreateInfo pool{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        pool.flags = VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT;
        pool.maxSets = kSetCount; pool.poolSizeCount = 2; pool.pPoolSizes = sizes;
        if (vkCreateDescriptorPool(state->device, &pool, nullptr, &state->pool) != VK_SUCCESS)
            throw std::runtime_error("Bloom: descriptor pool allocation failed");
        std::array<VkDescriptorSetLayout, kSetCount> layouts;
        layouts.fill(layout);
        std::array<VkDescriptorSet, kSetCount> sets{};
        VkDescriptorSetAllocateInfo alloc{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        alloc.descriptorPool = state->pool; alloc.descriptorSetCount = kSetCount; alloc.pSetLayouts = layouts.data();
        if (vkAllocateDescriptorSets(state->device, &alloc, sets.data()) != VK_SUCCESS)
            throw std::runtime_error("Bloom: descriptor allocation failed");
        std::copy_n(sets.begin(), MAX_FRAMES_IN_FLIGHT, state->prefilterSets.begin());
        std::copy_n(sets.begin() + MAX_FRAMES_IN_FLIGHT, kMipCount - 1, state->downSets.begin());
        std::copy_n(sets.begin() + MAX_FRAMES_IN_FLIGHT + kMipCount - 1, kMipCount - 1, state->upSets.begin());
        for (u32 i = 0; i < kSetCount; ++i)
        {
            const auto name = "Bloom.View" + std::to_string(id.value) + ".Set" + std::to_string(i);
            VulkanContext::SetDebugName(sets[i], name.c_str());
        }
        return state;
    }
    BloomViewState::~BloomViewState()
    {
        if (pool)
            VulkanContext::Get().PushDeletion([device = device, pool = pool] { vkDestroyDescriptorPool(device, pool, nullptr); });
    }
}

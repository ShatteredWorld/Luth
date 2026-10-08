#include "luthpch.h"
#include <limits>
#include "luth/renderer/features/TransparencyViewState.h"
#include "luth/renderer/features/FogViewState.h"
#include "luth/renderer/backend/vulkan/VulkanContext.h"
#include "luth/renderer/backend/vulkan/VulkanTexture.h"

namespace Luth
{
    ViewStateConfig TransparencyViewState::Config(u32 width, u32 height, u32 layers)
    {
        if (!width || !height || !layers || layers > 16 ||
            u64(width) * height > std::numeric_limits<u32>::max() / layers)
            throw std::invalid_argument("Transparency: invalid extent, layer budget or node capacity");
        return {width, height, layers};
    }
    u64 TransparencyViewState::NodeBytes(const ViewStateConfig& config)
    {
        if (config.signature > 16) throw std::invalid_argument("Transparency: invalid layer signature");
        Config(config.width, config.height, static_cast<u32>(config.signature));
        return 16ull + u64(config.width) * config.height * config.signature * 16ull;
    }
    std::shared_ptr<TransparencyViewState> TransparencyViewState::Create(RenderViewId id,
        const ViewStateConfig& config, const std::array<VkDescriptorSetLayout, 2>& layouts, u32 reservedTag)
    {
        const auto bytes = NodeBytes(config);
        if (!id.value || reservedTag < 0xFFFFC000u)
            throw std::invalid_argument("Transparency: invalid view identity or reserved tag");
        for (auto layout : layouts)
            if (!layout) throw std::runtime_error("Transparency: native layout is unavailable");
        auto state = std::make_shared<TransparencyViewState>();
        state->width = config.width; state->height = config.height;
        state->oitLayersCached = static_cast<u32>(config.signature);
        state->refractionBackdrop = Texture::Create(config.width, config.height, TextureFormat::RGBA16F);
        state->oitHeads = std::make_shared<VKTexture>(config.width, config.height, TextureFormat::R32_Uint,
            1, 0u, 1, VK_IMAGE_USAGE_STORAGE_BIT);
        state->oitNodesTag = reservedTag;
        state->oitNodes = Memory::GPUTaggedPageAllocator::Get().AllocateLargeTaggedDeviceLocal(reservedTag, bytes, 16);
        state->device = VulkanContext::Get().GetDevice();
        VkDescriptorPoolSize sizes[] = {
            {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, MAX_FRAMES_IN_FLIGHT + 1},
            {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, MAX_FRAMES_IN_FLIGHT + 1},
            {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 2 * MAX_FRAMES_IN_FLIGHT}
        };
        VkDescriptorPoolCreateInfo pool{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        pool.flags = VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT;
        pool.maxSets = MAX_FRAMES_IN_FLIGHT + 1; pool.poolSizeCount = 3; pool.pPoolSizes = sizes;
        if (vkCreateDescriptorPool(state->device, &pool, nullptr, &state->pool) != VK_SUCCESS)
            throw std::runtime_error("Transparency: descriptor pool allocation failed");
        std::array<VkDescriptorSetLayout, MAX_FRAMES_IN_FLIGHT> cycled;
        cycled.fill(layouts[0]);
        VkDescriptorSetAllocateInfo alloc{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        alloc.descriptorPool = state->pool; alloc.descriptorSetCount = MAX_FRAMES_IN_FLIGHT;
        alloc.pSetLayouts = cycled.data();
        if (vkAllocateDescriptorSets(state->device, &alloc, state->transparentDescSet.data()) != VK_SUCCESS)
            throw std::runtime_error("Transparency: cycled descriptor allocation failed");
        alloc.descriptorSetCount = 1; alloc.pSetLayouts = &layouts[1];
        if (vkAllocateDescriptorSets(state->device, &alloc, &state->oitResolveDescSet) != VK_SUCCESS)
            throw std::runtime_error("Transparency: resolve descriptor allocation failed");
        for (u32 slot = 0; slot < MAX_FRAMES_IN_FLIGHT; ++slot)
        {
            const auto name = "Transparency.View" + std::to_string(id.value) + ".Slot" + std::to_string(slot);
            VulkanContext::SetDebugName(state->transparentDescSet[slot], name.c_str());
        }
        const auto resolveName = "Transparency.Resolve.View" + std::to_string(id.value);
        VulkanContext::SetDebugName(state->oitResolveDescSet, resolveName.c_str());
        // Preserve the OIT_EMPTY bootstrap and GENERAL layout before the first FragmentStorageRead import.
        const auto image = static_cast<VKTexture*>(state->oitHeads.get())->GetImage();
        VulkanContext::Get().ImmediateSubmit([image](VkCommandBuffer cmd) {
            VkImageMemoryBarrier barrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
            barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED; barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            barrier.srcQueueFamilyIndex = barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barrier.image = image; barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
            barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                0, 0, nullptr, 0, nullptr, 1, &barrier);
            VkClearColorValue clear{}; clear.uint32[0] = 0xFFFFFFFFu;
            vkCmdClearColorImage(cmd, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &clear, 1, &barrier.subresourceRange);
            barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL; barrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
            barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
            vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                0, 0, nullptr, 0, nullptr, 1, &barrier);
        });
        return state;
    }
    TransparencyViewState::~TransparencyViewState()
    {
        if (pool)
            VulkanContext::Get().PushDeletion([device = device, pool = pool] { vkDestroyDescriptorPool(device, pool, nullptr); });
        refractionBackdrop.reset(); oitHeads.reset(); fogBindings.fill({});
        if (oitNodesTag) Memory::GPUTaggedPageAllocator::Get().FreeTagAndDestroy(oitNodesTag);
    }
}

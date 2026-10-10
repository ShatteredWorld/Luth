#include "luthpch.h"
#include "luth/renderer/features/rt/ReflectionDenoiserViewState.h"
#include "luth/renderer/backend/vulkan/VulkanContext.h"
#include "luth/renderer/backend/vulkan/VulkanTexture.h"

namespace Luth
{
    std::shared_ptr<ReflectionDenoiserViewState> ReflectionDenoiserViewState::Create(RenderViewId id,
        const ViewStateConfig& config, const std::array<VkDescriptorSetLayout, 4>& layouts)
    {
        const auto extent = ReflectionViewState::WorkingExtent(config);
        if (!id.value) throw std::invalid_argument("Reflection denoiser: invalid view identity");
        for (auto layout : layouts)
            if (!layout) throw std::runtime_error("Reflection denoiser: native layout unavailable");
        auto state = std::make_shared<ReflectionDenoiserViewState>();
        state->id = id; state->width = extent[0]; state->height = extent[1];
        state->sourceGeneration = config.resourceGeneration;
        auto texture = [](u32 w, u32 h) {
            return std::make_shared<VKTexture>(w, h, TextureFormat::RGBA16F, 1, 0u, 1, VK_IMAGE_USAGE_STORAGE_BIT);
        };
        state->svgfDenoised = texture(config.width, config.height);
        state->svgfHalf = texture(state->width, state->height);
        std::array<VkImage, 8> histories{};
        for (u32 i = 0; i < 2; ++i) {
            state->svgfColorHist[i] = texture(state->width, state->height);
            state->svgfMoments[i] = texture(state->width, state->height);
            state->svgfGeom[i] = texture(state->width, state->height);
            state->svgfAtrous[i] = texture(state->width, state->height);
            histories[i * 4] = std::static_pointer_cast<VKTexture>(state->svgfColorHist[i])->GetImage();
            histories[i * 4 + 1] = std::static_pointer_cast<VKTexture>(state->svgfMoments[i])->GetImage();
            histories[i * 4 + 2] = std::static_pointer_cast<VKTexture>(state->svgfGeom[i])->GetImage();
            histories[i * 4 + 3] = std::static_pointer_cast<VKTexture>(state->svgfAtrous[i])->GetImage();
        }
        // Preserve the original zero bootstrap and GENERAL storage-image history layout.
        VulkanContext::Get().ImmediateSubmit([&](VkCommandBuffer cmd) {
            std::array<VkImageMemoryBarrier, 8> barriers{};
            for (u32 i = 0; i < histories.size(); ++i) {
                auto& b = barriers[i]; b.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
                b.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED; b.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
                b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                b.image = histories[i]; b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
                b.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            }
            vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                0, 0, nullptr, 0, nullptr, 8, barriers.data());
            VkClearColorValue zero{}; VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
            for (auto image : histories) vkCmdClearColorImage(cmd, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &zero, 1, &range);
            for (auto& b : barriers) {
                b.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL; b.newLayout = VK_IMAGE_LAYOUT_GENERAL;
                b.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT; b.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
            }
            vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                0, 0, nullptr, 0, nullptr, 8, barriers.data());
        });
        state->device = VulkanContext::Get().GetDevice();
        const ReflectionDenoiserPoolBudget budget;
        VkDescriptorPoolCreateInfo pool{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        pool.maxSets = budget.maxSets; pool.poolSizeCount = 2; pool.pPoolSizes = budget.sizes.data();
        if (vkCreateDescriptorPool(state->device, &pool, nullptr, &state->pool) != VK_SUCCESS)
            throw std::runtime_error("Reflection denoiser: descriptor pool allocation failed");
        auto allocate = [&](VkDescriptorSetLayout layout, u32 count, VkDescriptorSet* sets) {
            std::array<VkDescriptorSetLayout, 2> pair{layout, layout};
            VkDescriptorSetAllocateInfo ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
            ai.descriptorPool = state->pool; ai.descriptorSetCount = count; ai.pSetLayouts = pair.data();
            if (vkAllocateDescriptorSets(state->device, &ai, sets) != VK_SUCCESS)
                throw std::runtime_error("Reflection denoiser: descriptor allocation failed");
        };
        allocate(layouts[0], 1, &state->svgfPassthroughDescSet);
        allocate(layouts[1], 2, state->svgfReprojectDescSet);
        allocate(layouts[2], 2, state->svgfMomentsDescSet);
        allocate(layouts[3], 2, state->svgfAtrousDescSet);
        return state;
    }
    ReflectionDenoiserViewState::~ReflectionDenoiserViewState()
    {
        if (pool) VulkanContext::Get().PushDeletion([device = device, pool = pool] { vkDestroyDescriptorPool(device, pool, nullptr); });
    }
}

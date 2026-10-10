#include "luthpch.h"
#include "luth/renderer/features/rt/ReflectionUpscaleViewState.h"
#include "luth/renderer/backend/vulkan/VulkanTexture.h"
#include "luth/renderer/backend/vulkan/VulkanContext.h"

namespace Luth
{
    std::shared_ptr<ReflectionUpscaleViewState> ReflectionUpscaleViewState::Create(RenderViewId id,
        const ViewStateConfig& config, VkDescriptorSetLayout layout, VkSampler sampler,
        const std::shared_ptr<ReflectionDenoiserViewState>& denoiser,
        const std::array<std::shared_ptr<Texture>, 2>& sources)
    {
        if (!id.value || !config.width || !config.height || !config.resourceGeneration)
            throw std::invalid_argument("Reflection upscale: invalid view configuration");
        if (!denoiser || denoiser->id != id || !denoiser->svgfHalf || !denoiser->svgfDenoised ||
            !sources[0] || !sources[1])
            throw std::invalid_argument("Reflection upscale: incompatible source owner");
        if (!layout || !sampler) throw std::runtime_error("Reflection upscale: native layout/sampler unavailable");
        const auto& d = *denoiser;
        if (!d.width || !d.height || d.svgfHalf->GetWidth() != d.width || d.svgfHalf->GetHeight() != d.height ||
            d.svgfDenoised->GetWidth() != config.width || d.svgfDenoised->GetHeight() != config.height)
            throw std::invalid_argument("Reflection upscale: incompatible signal extent");
        for (const auto& source : sources)
            if (source->GetWidth() != config.width || source->GetHeight() != config.height)
                throw std::invalid_argument("Reflection upscale: incompatible guide extent");
        auto state = std::make_shared<ReflectionUpscaleViewState>();
        state->id = id; state->sourceGeneration = config.resourceGeneration;
        state->denoiser = denoiser; state->sources = sources;
        auto view = [](const std::shared_ptr<Texture>& texture) {
            return std::static_pointer_cast<VKTexture>(texture)->GetImageView();
        };
        for (u32 i = 0; i < 2; ++i) state->sourceViews[i] = view(sources[i]);
        if (!state->sourceViews[0] || !state->sourceViews[1] || !view(d.svgfHalf) || !view(d.svgfDenoised))
            throw std::invalid_argument("Reflection upscale: missing native image view");
        state->device = VulkanContext::Get().GetDevice();
        const ReflectionUpscalePoolBudget budget;
        VkDescriptorPoolCreateInfo pool{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        pool.maxSets = budget.maxSets; pool.poolSizeCount = 2; pool.pPoolSizes = budget.sizes.data();
        if (vkCreateDescriptorPool(state->device, &pool, nullptr, &state->pool) != VK_SUCCESS)
            throw std::runtime_error("Reflection upscale: descriptor pool allocation failed");
        VkDescriptorSetAllocateInfo allocate{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        allocate.descriptorPool = state->pool; allocate.descriptorSetCount = 1; allocate.pSetLayouts = &layout;
        if (vkAllocateDescriptorSets(state->device, &allocate, &state->set) != VK_SUCCESS)
            throw std::runtime_error("Reflection upscale: descriptor allocation failed");
        VkDescriptorImageInfo images[]{
            {sampler, view(d.svgfHalf), VK_IMAGE_LAYOUT_GENERAL},
            {sampler, state->sourceViews[0], VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL},
            {sampler, state->sourceViews[1], VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL},
            {VK_NULL_HANDLE, view(d.svgfDenoised), VK_IMAGE_LAYOUT_GENERAL}};
        VkWriteDescriptorSet writes[4]{};
        for (u32 i = 0; i < 4; ++i) {
            auto& w = writes[i]; w.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            w.dstSet = state->set; w.dstBinding = i; w.descriptorCount = 1;
            w.descriptorType = i == 3 ? VK_DESCRIPTOR_TYPE_STORAGE_IMAGE : VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            w.pImageInfo = &images[i];
        }
        vkUpdateDescriptorSets(state->device, 4, writes, 0, nullptr);
        return state;
    }
    ReflectionUpscaleViewState::~ReflectionUpscaleViewState()
    {
        if (pool) VulkanContext::Get().PushDeletion([device = device, pool = pool] { vkDestroyDescriptorPool(device, pool, nullptr); });
    }
}

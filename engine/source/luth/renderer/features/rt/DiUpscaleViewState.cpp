#include "luthpch.h"
#include "luth/renderer/features/rt/DiUpscaleViewState.h"
#include "luth/renderer/backend/vulkan/VulkanTexture.h"
#include "luth/renderer/backend/vulkan/VulkanContext.h"

namespace Luth
{
    std::shared_ptr<DiUpscaleViewState> DiUpscaleViewState::Create(RenderViewId id,
        const ViewStateConfig& config, VkDescriptorSetLayout layout, VkSampler sampler,
        const std::array<std::shared_ptr<DiDenoiserViewState>, 2>& denoisers,
        const std::array<std::shared_ptr<Texture>, 2>& sources)
    {
        if (!id.value || !config.width || !config.height)
            throw std::invalid_argument("DI upscale: invalid view configuration");
        for (u32 i = 0; i < denoisers.size(); ++i) {
            const auto& d = denoisers[i];
            if (!d || d->id != id || d->signal != (i ? DiDenoiserSignal::Specular : DiDenoiserSignal::Diffuse) ||
                !d->svgfDiHalf || !d->svgfDenoised || !sources[i])
                throw std::invalid_argument("DI upscale: incompatible source owner");
        }
        if (!layout || !sampler) throw std::runtime_error("DI upscale: native layout/sampler unavailable");
        for (u32 i = 0; i < 2; ++i) {
            const auto& d = denoisers[i];
            if (!d->width || !d->height ||
                d->svgfDiHalf->GetWidth() != d->width || d->svgfDiHalf->GetHeight() != d->height ||
                d->svgfDenoised->GetWidth() != config.width || d->svgfDenoised->GetHeight() != config.height ||
                sources[i]->GetWidth() != config.width || sources[i]->GetHeight() != config.height)
                throw std::invalid_argument("DI upscale: incompatible source extent");
        }
        auto state = std::make_shared<DiUpscaleViewState>();
        state->id = id; state->sourceGeneration = config.resourceGeneration;
        state->denoisers = denoisers; state->sources = sources;
        state->device = VulkanContext::Get().GetDevice();
        const DiUpscalePoolBudget budget;
        VkDescriptorPoolCreateInfo pool{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        pool.maxSets = budget.maxSets; pool.poolSizeCount = 2; pool.pPoolSizes = budget.sizes.data();
        if (vkCreateDescriptorPool(state->device, &pool, nullptr, &state->pool) != VK_SUCCESS)
            throw std::runtime_error("DI upscale: descriptor pool allocation failed");
        const VkDescriptorSetLayout layouts[]{layout, layout};
        VkDescriptorSetAllocateInfo allocate{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        allocate.descriptorPool = state->pool; allocate.descriptorSetCount = 2; allocate.pSetLayouts = layouts;
        if (vkAllocateDescriptorSets(state->device, &allocate, state->sets.data()) != VK_SUCCESS)
            throw std::runtime_error("DI upscale: descriptor allocation failed");
        auto view = [](const std::shared_ptr<Texture>& texture) {
            return std::static_pointer_cast<VKTexture>(texture)->GetImageView();
        };
        for (u32 i = 0; i < 2; ++i) state->sourceViews[i] = view(sources[i]);
        for (u32 channel = 0; channel < 2; ++channel) {
            const auto& d = denoisers[channel];
            VkDescriptorImageInfo images[]{
                {sampler, view(d->svgfDiHalf), VK_IMAGE_LAYOUT_GENERAL},
                {sampler, state->sourceViews[0], VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL},
                {sampler, state->sourceViews[1], VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL},
                {VK_NULL_HANDLE, view(d->svgfDenoised), VK_IMAGE_LAYOUT_GENERAL}};
            VkWriteDescriptorSet writes[4]{};
            for (u32 i = 0; i < 4; ++i) {
                auto& w = writes[i]; w.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                w.dstSet = state->sets[channel]; w.dstBinding = i; w.descriptorCount = 1;
                w.descriptorType = i == 3 ? VK_DESCRIPTOR_TYPE_STORAGE_IMAGE : VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                w.pImageInfo = &images[i];
            }
            vkUpdateDescriptorSets(state->device, 4, writes, 0, nullptr);
        }
        return state;
    }
    DiUpscaleViewState::~DiUpscaleViewState()
    {
        if (pool) VulkanContext::Get().PushDeletion([device = device, pool = pool] { vkDestroyDescriptorPool(device, pool, nullptr); });
    }
}

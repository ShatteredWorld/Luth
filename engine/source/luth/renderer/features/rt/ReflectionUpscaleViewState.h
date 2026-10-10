#pragma once
#include "luth/renderer/features/rt/ReflectionDenoiserViewState.h"
#include "luth/renderer/features/RenderResource.h"

namespace Luth
{
    struct ReflectionUpscalePoolBudget
    {
        std::array<VkDescriptorPoolSize, 2> sizes{{
            {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 3},
            {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1}}};
        u32 maxSets = 1;
    };
    // Borrows the reflection denoiser owner; owns only immutable upscale descriptors.
    struct ReflectionUpscaleViewState
    {
        ReflectionUpscaleViewState() = default;
        ~ReflectionUpscaleViewState();
        ReflectionUpscaleViewState(const ReflectionUpscaleViewState&) = delete;
        ReflectionUpscaleViewState& operator=(const ReflectionUpscaleViewState&) = delete;
        static std::shared_ptr<ReflectionUpscaleViewState> Create(RenderViewId, const ViewStateConfig&,
            VkDescriptorSetLayout, VkSampler, const std::shared_ptr<ReflectionDenoiserViewState>&,
            const std::array<std::shared_ptr<Texture>, 2>&);
        RenderViewId id;
        u64 sourceGeneration = 0;
        std::shared_ptr<ReflectionDenoiserViewState> denoiser;
        std::array<std::shared_ptr<Texture>, 2> sources; // Full depth and normal.
        std::array<VkImageView, 2> sourceViews{};
        VkDescriptorSet set = VK_NULL_HANDLE;
        VkDevice device = VK_NULL_HANDLE;
        VkDescriptorPool pool = VK_NULL_HANDLE;
    };
    using ReflectionUpscaleViewStates = FeatureViewStates<std::shared_ptr<ReflectionUpscaleViewState>>;

    struct ReflectionUpscaleBindings
    {
        RenderViewId view;
        u64 generation = 0, frameIndex = 0;
        VkPipeline pipeline = VK_NULL_HANDLE;
        VkPipelineLayout layout = VK_NULL_HANDLE;
        VkDescriptorSet globalSet = VK_NULL_HANDLE, set = VK_NULL_HANDLE;
        std::array<TextureBindingRef, 3> sources; // Working signal, full depth/normal.
        std::array<VkImage, 3> sourceImages{};
        std::array<VkImageView, 3> sourceViews{};
        TextureBindingRef output;
        VkImage outputImage = VK_NULL_HANDLE;
        VkImageView outputView = VK_NULL_HANDLE;
        u32 width = 0, height = 0, fullWidth = 0, fullHeight = 0;
        f32 phiDepth = 0, phiNormal = 32.0f;
        std::shared_ptr<const ReflectionUpscaleViewState> retained;
        bool Ready() const { return pipeline && layout && globalSet && set; }
    };
}

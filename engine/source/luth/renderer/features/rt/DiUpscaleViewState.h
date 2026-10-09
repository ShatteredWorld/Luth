#pragma once
#include "luth/renderer/features/rt/DiDenoiserViewState.h"
#include "luth/renderer/features/RenderResource.h"

namespace Luth
{
    struct DiUpscalePoolBudget
    {
        std::array<VkDescriptorPoolSize, 2> sizes{{
            {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 6},
            {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 2}}};
        u32 maxSets = 2;
    };
    // Borrows both denoiser owners; owns only immutable upscale descriptors.
    struct DiUpscaleViewState
    {
        DiUpscaleViewState() = default;
        ~DiUpscaleViewState();
        DiUpscaleViewState(const DiUpscaleViewState&) = delete;
        DiUpscaleViewState& operator=(const DiUpscaleViewState&) = delete;
        static std::shared_ptr<DiUpscaleViewState> Create(RenderViewId, const ViewStateConfig&,
            VkDescriptorSetLayout, VkSampler, const std::array<std::shared_ptr<DiDenoiserViewState>, 2>&,
            const std::array<std::shared_ptr<Texture>, 2>&);
        RenderViewId id;
        u64 sourceGeneration = 0;
        std::array<std::shared_ptr<DiDenoiserViewState>, 2> denoisers;
        std::array<std::shared_ptr<Texture>, 2> sources; // Full depth and normal.
        std::array<VkImageView, 2> sourceViews{};
        std::array<VkDescriptorSet, 2> sets{};
        VkDevice device = VK_NULL_HANDLE;
        VkDescriptorPool pool = VK_NULL_HANDLE;
    };
    using DiUpscaleViewStates = FeatureViewStates<std::shared_ptr<DiUpscaleViewState>>;

    struct DiUpscaleBindings
    {
        DiDenoiserSignal signal = DiDenoiserSignal::Diffuse;
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
        std::shared_ptr<const DiUpscaleViewState> retained;
        bool Ready() const { return pipeline && layout && globalSet && set; }
    };
}

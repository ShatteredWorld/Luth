#pragma once
#include "luth/renderer/features/rt/RestirDiViewState.h"

namespace Luth
{
    // One motion-reprojection diffuse channel. Physical ownership only; no RG handles.
    struct DiDenoiserPoolBudget
    {
        std::array<VkDescriptorPoolSize, 2> sizes{{
            {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 21},
            {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 25}}};
        u32 maxSets = 7;
    };
    struct DiDenoiserViewState
    {
        DiDenoiserViewState() = default;
        ~DiDenoiserViewState();
        DiDenoiserViewState(const DiDenoiserViewState&) = delete;
        DiDenoiserViewState& operator=(const DiDenoiserViewState&) = delete;
        static ViewStateConfig Config(u32 width, u32 height, bool half, u64 sourceGeneration)
        { return RestirDiViewState::Config(width, height, half, sourceGeneration); }
        static std::shared_ptr<DiDenoiserViewState> Create(RenderViewId, const ViewStateConfig&,
            const std::array<VkDescriptorSetLayout, 4>&);
        RenderViewId id;
        u32 width = 0, height = 0;
        u64 sourceGeneration = 0;
        std::shared_ptr<Texture> svgfDenoised, svgfDiHalf;
        std::shared_ptr<Texture> svgfColorHist[2], svgfMoments[2], svgfGeom[2], svgfAtrous[2];
        VkDescriptorSet svgfPassthroughDescSet = VK_NULL_HANDLE;
        VkDescriptorSet svgfReprojectDescSet[2]{}, svgfMomentsDescSet[2]{}, svgfAtrousDescSet[2]{};
        std::shared_ptr<RestirDiViewState> input;
        // Retain every descriptor source, including material ID and roughness.
        std::array<std::shared_ptr<Texture>, 5> sources;
        std::array<VkImageView, 5> sourceViews{};
        VkDevice device = VK_NULL_HANDLE;
        VkDescriptorPool pool = VK_NULL_HANDLE;
    };
    using DiDenoiserViewStates = FeatureViewStates<std::shared_ptr<DiDenoiserViewState>>;
}

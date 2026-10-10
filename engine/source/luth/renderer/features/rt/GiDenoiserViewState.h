#pragma once
#include "luth/renderer/features/rt/RestirGiViewState.h"

namespace Luth
{
    // GI motion-reprojection state; native resources remain local to each view.
    struct GiDenoiserPoolBudget
    {
        std::array<VkDescriptorPoolSize, 2> sizes{{
            {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 21},
            {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 25}}};
        u32 maxSets = 7;
    };
    struct GiDenoiserViewState
    {
        GiDenoiserViewState() = default;
        ~GiDenoiserViewState();
        GiDenoiserViewState(const GiDenoiserViewState&) = delete;
        GiDenoiserViewState& operator=(const GiDenoiserViewState&) = delete;
        static ViewStateConfig Config(u32 width, u32 height, bool half, u64 sourceGeneration)
        { return RestirGiViewState::Config(width, height, half, sourceGeneration); }
        static std::shared_ptr<GiDenoiserViewState> Create(RenderViewId, const ViewStateConfig&,
            const std::array<VkDescriptorSetLayout, 4>&);
        std::shared_ptr<Texture>* Noisy() const {
            if (!input) return nullptr;
            return &input->restirGiDI;
        }
        RenderViewId id;
        u32 width = 0, height = 0;
        TemporalSignalHistory history;
        u64 sourceGeneration = 0;
        std::shared_ptr<Texture> svgfDenoised, svgfGiHalf;
        std::shared_ptr<Texture> svgfColorHist[2], svgfMoments[2], svgfGeom[2], svgfAtrous[2];
        VkDescriptorSet svgfPassthroughDescSet = VK_NULL_HANDLE;
        VkDescriptorSet svgfReprojectDescSet[2]{}, svgfMomentsDescSet[2]{}, svgfAtrousDescSet[2]{};
        std::shared_ptr<RestirGiViewState> input;
        // Retain every descriptor source, including material ID and roughness.
        std::array<std::shared_ptr<Texture>, 5> sources;
        std::array<VkImageView, 5> sourceViews{};
        VkDevice device = VK_NULL_HANDLE;
        VkDescriptorPool pool = VK_NULL_HANDLE;
    };
    using GiDenoiserViewStates = FeatureViewStates<std::shared_ptr<GiDenoiserViewState>>;
}

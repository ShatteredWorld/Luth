#pragma once
#include "luth/renderer/features/rt/ReflectionViewState.h"

namespace Luth
{
    // Reflection hit-distance virtual-reprojection state; native resources remain local to each view.
    struct ReflectionDenoiserPoolBudget
    {
        std::array<VkDescriptorPoolSize, 2> sizes{{
            {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 21},
            {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 25}}};
        u32 maxSets = 7;
    };
    struct ReflectionDenoiserViewState
    {
        ReflectionDenoiserViewState() = default;
        ~ReflectionDenoiserViewState();
        ReflectionDenoiserViewState(const ReflectionDenoiserViewState&) = delete;
        ReflectionDenoiserViewState& operator=(const ReflectionDenoiserViewState&) = delete;
        static ViewStateConfig Config(u32 width, u32 height, bool half, u64 sourceGeneration)
        { return ReflectionViewState::Config(width, height, half, sourceGeneration); }
        static std::shared_ptr<ReflectionDenoiserViewState> Create(RenderViewId, const ViewStateConfig&,
            const std::array<VkDescriptorSetLayout, 4>&);
        std::shared_ptr<Texture>* Noisy() const {
            if (!input) return nullptr;
            return &input->radiance;
        }
        std::shared_ptr<Texture>* WorkingOutput() {
            if (!svgfDenoised) return nullptr;
            return width != svgfDenoised->GetWidth() || height != svgfDenoised->GetHeight()
                ? &svgfHalf : &svgfDenoised;
        }
        RenderViewId id;
        u32 width = 0, height = 0;
        TemporalSignalHistory history;
        u64 sourceGeneration = 0;
        std::shared_ptr<Texture> svgfDenoised, svgfHalf;
        std::shared_ptr<Texture> svgfColorHist[2], svgfMoments[2], svgfGeom[2], svgfAtrous[2];
        VkDescriptorSet svgfPassthroughDescSet = VK_NULL_HANDLE;
        VkDescriptorSet svgfReprojectDescSet[2]{}, svgfMomentsDescSet[2]{}, svgfAtrousDescSet[2]{};
        std::shared_ptr<ReflectionViewState> input;
        // Depth, normal, roughness (virtual reprojection and edge stop), material ID.
        std::array<std::shared_ptr<Texture>, 4> sources;
        std::array<VkImageView, 4> sourceViews{};
        VkDevice device = VK_NULL_HANDLE;
        VkDescriptorPool pool = VK_NULL_HANDLE;
    };
    using ReflectionDenoiserViewStates = FeatureViewStates<std::shared_ptr<ReflectionDenoiserViewState>>;
}

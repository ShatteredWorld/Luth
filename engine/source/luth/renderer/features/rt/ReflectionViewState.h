#pragma once
#include <vulkan/vulkan.h>
#include "luth/renderer/features/rt/TemporalSignalHistory.h"
#include "luth/renderer/resources/Texture.h"
#include <array>
#include <algorithm>

namespace Luth
{
    struct ReflectionPoolBudget
    {
        std::array<VkDescriptorPoolSize, 2> sizes{{
            {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 3},
            {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1}}};
        u32 maxSets = 1;
    };

    // Raw demodulated specular radiance (RGB) and ray hit distance (A).
    // Physical resources only; retained guides keep the immutable trace set valid.
    struct ReflectionViewState
    {
        ReflectionViewState() = default;
        ~ReflectionViewState();
        ReflectionViewState(const ReflectionViewState&) = delete;
        ReflectionViewState& operator=(const ReflectionViewState&) = delete;
        static ViewStateConfig Config(u32 width, u32 height, bool half, u64 sourceGeneration);
        static std::array<u32, 2> WorkingExtent(const ViewStateConfig& config);
        static std::shared_ptr<ReflectionViewState> Create(RenderViewId, const ViewStateConfig&, VkDescriptorSetLayout);
        RenderViewId id;
        u32 width = 0, height = 0;
        TemporalSignalHistory history;
        u64 sourceGeneration = 0;
        std::shared_ptr<Texture> radiance;
        std::array<std::shared_ptr<Texture>, 3> sources; // depth, normal, roughness
        std::array<VkImageView, 3> sourceViews{};
        VkDevice device = VK_NULL_HANDLE;
        VkDescriptorPool pool = VK_NULL_HANDLE;
        VkDescriptorSet descriptorSet = VK_NULL_HANDLE;
    };
    using ReflectionViewStates = FeatureViewStates<std::shared_ptr<ReflectionViewState>>;
}

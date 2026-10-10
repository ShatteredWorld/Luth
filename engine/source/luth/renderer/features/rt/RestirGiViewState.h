#pragma once
#include "luth/renderer/features/rt/TemporalSignalHistory.h"

#include "luth/renderer/features/RenderViewState.h"
#include "luth/renderer/resources/Texture.h"
#include "luth/memory/GPUTaggedPageAllocator.h"
#include "luth/core/FrameData.h"
#include <array>
#include <algorithm>

namespace Luth
{
    struct RestirGiPoolBudget
    {
        std::array<VkDescriptorPoolSize, 3> sizes;
        u32 maxSets;
        explicit RestirGiPoolBudget(u32 frames) : sizes{{
            {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 3 * frames + 1},
            {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, frames},
            {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 3 * frames + 1}}}, maxSets(frames + 1)
        {
            if (!frames) throw std::invalid_argument("ReSTIR GI: invalid descriptor frame count");
        }
    };

    // Domain-owned physical resources; no graph handles. Spatial output is temporal
    // history, scratch is same-frame only. Neither buffer is ping-ponged.
    struct RestirGiViewState
    {
        RestirGiViewState() = default;
        ~RestirGiViewState();
        RestirGiViewState(const RestirGiViewState&) = delete;
        RestirGiViewState& operator=(const RestirGiViewState&) = delete;
        static ViewStateConfig Config(u32 width, u32 height, bool half, u64 sourceGeneration);
        static std::array<u32, 2> WorkingExtent(const ViewStateConfig& config) {
            Config(config.width, config.height, config.signature != 0, config.resourceGeneration);
            if (config.signature > 1) throw std::invalid_argument("ReSTIR GI: invalid resolution mode");
            return {config.signature ? std::max(config.width / 2, 1u) : config.width,
                    config.signature ? std::max(config.height / 2, 1u) : config.height};
        }
        static std::shared_ptr<RestirGiViewState> Create(RenderViewId, const ViewStateConfig&,
            VkDescriptorSetLayout, VkDescriptorSetLayout vizLayout, u32 scratchTag, u32 spatialTag);
        RenderViewId id;
        u32 width = 0, height = 0;
        TemporalSignalHistory history;
        Memory::GPUSubRegion restirGiReservoir{}, restirGiSpatial{};
        u32 restirGiReservoirTag = 0, restirGiSpatialTag = 0;
        std::shared_ptr<Texture> restirGiDI;
        // depth, normal, motion; retained for stable descriptor lifetime.
        std::array<std::shared_ptr<Texture>, 3> sources;
        std::array<VkImageView, 3> sourceViews{};
        u64 sourceGeneration = 0;
        VkDevice device = VK_NULL_HANDLE;
        VkDescriptorPool pool = VK_NULL_HANDLE;
        VkDescriptorSet giReservoirVizDescSet = VK_NULL_HANDLE;
        std::array<VkDescriptorSet, MAX_FRAMES_IN_FLIGHT> restirGiDescSet{};
    };
    using RestirGiViewStates = FeatureViewStates<std::shared_ptr<RestirGiViewState>>;
}

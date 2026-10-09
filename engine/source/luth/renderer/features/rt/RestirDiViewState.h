#pragma once
#include "luth/renderer/features/RenderViewState.h"
#include "luth/renderer/resources/Texture.h"
#include "luth/memory/GPUTaggedPageAllocator.h"
#include "luth/core/FrameData.h"
#include <array>
#include <algorithm>

namespace Luth
{
    struct RestirDiPoolBudget
    {
        std::array<VkDescriptorPoolSize, 3> sizes;
        u32 maxSets;
        explicit RestirDiPoolBudget(u32 frames) : sizes{{
            {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 4 * frames},
            {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 2 * frames},
            {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 3 * frames}}}, maxSets(frames)
        {
            if (!frames) throw std::invalid_argument("ReSTIR DI: invalid descriptor frame count");
        }
    };

    // Domain-owned physical resources; no graph handles. Spatial output is temporal
    // history, scratch is same-frame only. Neither buffer is ping-ponged.
    struct RestirDiViewState
    {
        RestirDiViewState() = default;
        ~RestirDiViewState();
        RestirDiViewState(const RestirDiViewState&) = delete;
        RestirDiViewState& operator=(const RestirDiViewState&) = delete;
        static ViewStateConfig Config(u32 width, u32 height, bool half, u64 sourceGeneration);
        static std::array<u32, 2> WorkingExtent(const ViewStateConfig& config) {
            Config(config.width, config.height, config.signature != 0, config.resourceGeneration);
            if (config.signature > 1) throw std::invalid_argument("ReSTIR DI: invalid resolution mode");
            return {config.signature ? std::max(config.width / 2, 1u) : config.width,
                    config.signature ? std::max(config.height / 2, 1u) : config.height};
        }
        static std::shared_ptr<RestirDiViewState> Create(RenderViewId, const ViewStateConfig&,
            VkDescriptorSetLayout, u32 scratchTag, u32 spatialTag);
        RenderViewId id;
        u32 width = 0, height = 0;
        Memory::GPUSubRegion restirReservoir{}, restirSpatial{};
        u32 restirReservoirTag = 0, restirSpatialTag = 0;
        std::shared_ptr<Texture> restirDI, restirDISpec;
        // depth, normal, motion, roughness; retained for stable descriptor lifetime.
        std::array<std::shared_ptr<Texture>, 4> sources;
        std::array<VkImageView, 4> sourceViews{};
        u64 sourceGeneration = 0;
        VkDevice device = VK_NULL_HANDLE;
        VkDescriptorPool pool = VK_NULL_HANDLE;
        std::array<VkDescriptorSet, MAX_FRAMES_IN_FLIGHT> restirDescSet{};
    };
    using RestirDiViewStates = FeatureViewStates<std::shared_ptr<RestirDiViewState>>;
}

#pragma once
#include "luth/core/FrameData.h"
#include "luth/renderer/features/RenderViewState.h"
#include "luth/renderer/resources/Texture.h"
#include <array>
#include <vulkan/vulkan.h>

namespace Luth
{
    struct TaaViewState
    {
        TaaViewState() = default;
        TaaViewState(const TaaViewState&) = delete;
        TaaViewState& operator=(const TaaViewState&) = delete;
        ~TaaViewState();
        static ViewStateConfig Config(u32 width, u32 height, u64 sourceGeneration = 1);
        static std::shared_ptr<TaaViewState> Create(RenderViewId, const ViewStateConfig&, VkDescriptorSetLayout);
        VkDevice device = VK_NULL_HANDLE;
        VkDescriptorPool pool = VK_NULL_HANDLE;
        std::array<VkDescriptorSet, MAX_FRAMES_IN_FLIGHT> resolveSets{};
        std::shared_ptr<Texture> historyA, historyB;
        // Retain the stable sampled inputs together with their descriptor pool.
        std::array<std::shared_ptr<Texture>, 3> sources{};
        u64 sourceGeneration = 1;
        u64 shaderGeneration = 0;
        void ApplyShaderGeneration(u64 generation)
        {
            if (shaderGeneration == generation) return;
            history.Invalidate(); shaderGeneration = generation;
        }
        ViewHistoryState history;
        bool recorded = false;
    };
    using TaaViewStateStore = FeatureViewStates<std::shared_ptr<TaaViewState>>;
}

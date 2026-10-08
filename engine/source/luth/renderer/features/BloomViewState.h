#pragma once
#include "luth/core/FrameData.h"
#include "luth/renderer/features/RenderViewState.h"
#include "luth/renderer/resources/Texture.h"
#include <array>
#include <utility>
#include <vulkan/vulkan.h>

namespace Luth
{
    struct BloomViewState
    {
        static constexpr u32 kMipCount = 6;
        static constexpr u32 kSetCount = MAX_FRAMES_IN_FLIGHT + 2 * (kMipCount - 1);
        BloomViewState() = default;
        BloomViewState(const BloomViewState&) = delete;
        BloomViewState& operator=(const BloomViewState&) = delete;
        ~BloomViewState();
        static ViewStateConfig Config(u32 width, u32 height);
        static std::pair<u32, u32> MipExtent(const ViewStateConfig&, u32 mip);
        static std::shared_ptr<BloomViewState> Create(RenderViewId, const ViewStateConfig&, VkDescriptorSetLayout);
        VkDevice device = VK_NULL_HANDLE;
        VkDescriptorPool pool = VK_NULL_HANDLE;
        u32 width = 0, height = 0;
        std::array<std::shared_ptr<Texture>, kMipCount> mips{};
        std::array<VkDescriptorSet, MAX_FRAMES_IN_FLIGHT> prefilterSets{};
        std::array<VkDescriptorSet, kMipCount - 1> downSets{}, upSets{};
    };
    using BloomViewStateStore = FeatureViewStates<std::shared_ptr<BloomViewState>>;
}

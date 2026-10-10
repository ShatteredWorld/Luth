#pragma once
#include "luth/renderer/features/RenderResource.h"
#include "luth/renderer/features/BloomViewState.h"
#include "luth/core/types/LuthMath.h"

namespace Luth
{
    struct BloomDownPC
    {
        Vec2 srcTexel{};
        IVec2 dstSize{};
        f32 threshold = 0, knee = 0;
        u32 prefilter = 0, pad = 0;
    };
    struct BloomUpPC
    {
        Vec2 srcTexel{};
        IVec2 dstSize{};
        f32 radius = 0, pad0 = 0, pad1 = 0, pad2 = 0;
    };
    static_assert(sizeof(BloomDownPC) == 32);
    static_assert(sizeof(BloomUpPC) == 32);
    struct BloomBindings
    {
        bool enabled = false;
        VkPipeline downPipeline = VK_NULL_HANDLE, upPipeline = VK_NULL_HANDLE;
        VkPipelineLayout downLayout = VK_NULL_HANDLE, upLayout = VK_NULL_HANDLE;
        VkDescriptorSet prefilterSet = VK_NULL_HANDLE;
        std::array<VkDescriptorSet, BloomViewState::kMipCount - 1> downSets{}, upSets{};
        TextureBindingRef source;
        std::array<TextureBindingRef, BloomViewState::kMipCount> mips{};
        std::array<VkImage, BloomViewState::kMipCount> images{};
        std::array<VkImageView, BloomViewState::kMipCount> views{};
        u32 width = 0, height = 0;
        f32 threshold = 0, radius = 0;
        std::shared_ptr<BloomViewState> state;
    };
}

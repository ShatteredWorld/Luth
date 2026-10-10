#pragma once
#include "luth/renderer/features/FogViewState.h"
namespace Luth
{
    struct FogVizPushConstants { u32 mode = 0; float scale = 1, overlayAlpha = 1; };
    static_assert(sizeof(FogVizPushConstants) == 12);
    struct FogVizBindings
    {
        VkPipeline pipeline = VK_NULL_HANDLE;
        VkPipelineLayout layout = VK_NULL_HANDLE;
        std::array<VkDescriptorSet, 2> sets{};
        std::shared_ptr<FogViewState> state;
        u64 renderFrameIndex = 0;
        FogVizPushConstants parameters{};
        bool enabled = false;
    };
}

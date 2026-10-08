#pragma once
#include "luth/renderer/features/EditorOverlayViewState.h"
#include "luth/renderer/features/RenderResource.h"
namespace Luth
{
    struct GridPushConstants
    {
        float axisXColor[4], axisZColor[4], gridColor[4];
        float majorScale, fadeStart, fadeEnd, lineThickness;
        float jitter[2];
    };
    static_assert(sizeof(GridPushConstants) == 72);
    struct GridBindings
    {
        VkPipeline pipeline = VK_NULL_HANDLE;
        VkPipelineLayout layout = VK_NULL_HANDLE;
        VkDescriptorSet set = VK_NULL_HANDLE;
        TextureBindingRef depth;
        GridPushConstants parameters{};
        std::shared_ptr<EditorOverlayViewState> state;
        bool enabled = false;
    };
}

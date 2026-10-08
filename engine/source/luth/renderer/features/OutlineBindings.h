#pragma once
#include "luth/renderer/features/EditorOverlayViewState.h"
#include "luth/renderer/features/RenderResource.h"
namespace Luth
{
    struct CameraParams;
    struct OutlinePushConstants
    {
        float width, texelSizeX, texelSizeY;
        float colorR, colorG, colorB, colorA, occludedAlpha;
    };
    static_assert(sizeof(OutlinePushConstants) == 32);
    OutlinePushConstants MakeOutlineParameters(const CameraParams&, u32 width, u32 height);
    struct OutlineBindings
    {
        VkPipeline pipeline = VK_NULL_HANDLE;
        VkPipelineLayout layout = VK_NULL_HANDLE;
        VkDescriptorSet set = VK_NULL_HANDLE;
        std::shared_ptr<EditorOverlayViewState> state;
        OutlinePushConstants parameters{};
        bool enabled = false;
    };
}

#pragma once
#include "luth/renderer/features/RenderResource.h"
#include <vulkan/vulkan.h>

namespace Luth
{
    struct RefractionBackdropBindings
    {
        VkImage image = VK_NULL_HANDLE;
        VkImageView view = VK_NULL_HANDLE;
        TextureBindingRef binding;
        u32 width = 0, height = 0;
        bool enabled = false;
    };
}

#include "luthpch.h"
#include "luth/renderer/features/GtaoViewState.h"
#include "luth/renderer/backend/vulkan/VulkanContext.h"
#include "luth/renderer/resources/Texture.h"

namespace Luth
{
    GtaoViewState::~GtaoViewState()
    {
        if (!pool) return; // Headless states own no Vulkan resources.
        // Queue the pool first. Texture destructors enqueue their native handles in the
        // same completion slot below; retaining textures inside this callback would nest
        // deletions while shutdown is already draining that slot.
        VulkanContext::Get().PushDeletion([device = device, pool = pool]() {
            vkDestroyDescriptorPool(device, pool, nullptr);
        });
        linearDepth.reset(); rawAO.reset(); edges.reset(); finalAO.reset();
    }
}

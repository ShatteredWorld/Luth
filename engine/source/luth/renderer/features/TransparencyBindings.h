#pragma once
#include "luth/renderer/subsystems/GeometrySubsystem.h"

namespace Luth
{
    class RtSubsystem;
    struct TransparencyBindings
    {
        std::array<VkDescriptorSet, 7> sets{};
        VkPolygonMode polygon = VK_POLYGON_MODE_FILL;
        std::vector<ForwardDrawPacket> draws;
        TextureBindingRef fog, backdrop;
        const RtSubsystem* rayScene = nullptr; // Temporary paired geometry-table lookup until M13.
        bool captureDraws = false;
        bool oit = false;
        VkImage headsImage = VK_NULL_HANDLE;
        VkImageView headsView = VK_NULL_HANDLE;
        TextureBindingRef heads;
        Memory::GPUSubRegion nodes{};
        u32 width = 0, height = 0, capacity = 0, maxResolveK = 8;
        VkPipeline resolvePipeline = VK_NULL_HANDLE;
        VkPipelineLayout resolveLayout = VK_NULL_HANDLE;
        VkDescriptorSet resolveSet = VK_NULL_HANDLE;
    };
}

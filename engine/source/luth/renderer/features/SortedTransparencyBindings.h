#pragma once
#include "luth/renderer/subsystems/GeometrySubsystem.h"

namespace Luth
{
    class RtSubsystem;
    struct SortedTransparencyBindings
    {
        std::array<VkDescriptorSet, 7> sets{};
        VkPolygonMode polygon = VK_POLYGON_MODE_FILL;
        std::vector<ForwardDrawPacket> draws;
        TextureBindingRef fog, backdrop;
        const RtSubsystem* rayScene = nullptr; // Temporary paired geometry-table lookup until M13.
        bool captureDraws = false;
    };
}

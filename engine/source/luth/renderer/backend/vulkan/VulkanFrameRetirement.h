#pragma once
#include <cstdint>
#include <optional>
#include <stdexcept>

namespace Luth
{
    // App renders 0, 1 synchronously, then renders N-1. The first pipelined
    // iteration reuses frame 1 descriptors before the normal ring wait starts.
    inline std::optional<uint32_t> DescriptorRetirementSlot(uint64_t frameIndex, uint32_t frames)
    {
        if (!frames) throw std::invalid_argument("Frame retirement requires a nonempty ring");
        if (frameIndex >= frames) return static_cast<uint32_t>((frameIndex - frames + 1) % frames);
        if (frameIndex == 2) return 1 % frames;
        return std::nullopt;
    }
}

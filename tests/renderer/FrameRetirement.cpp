#include <doctest/doctest.h>
#include "luth/renderer/backend/vulkan/VulkanFrameRetirement.h"
#include <array>
#include <algorithm>
using namespace Luth;

TEST_CASE("FrameRetirement: completion covers bootstrap replay and steady descriptor writers")
{
    for (uint32_t frames : {2u, 3u, 4u}) {
        std::array<uint64_t, 4> completions{}, descriptorReaders{}, commandUsers{};
        uint64_t completed = 0;
        for (uint64_t game = 0; game < 24; ++game) {
            const auto render = game < 2 ? game : game - 1;
            if (const auto wait = DescriptorRetirementSlot(game, frames))
                completed = std::max(completed, completions[*wait]);
            CHECK(completed >= descriptorReaders[render % frames]);
            CHECK(completed >= descriptorReaders[game % frames]);
            CHECK(completed >= commandUsers[game % frames]);
            descriptorReaders[render % frames] = game + 1;
            commandUsers[game % frames] = game + 1;
            completions[game % frames] = game + 1;
        }
    }
}
TEST_CASE("FrameRetirement: first pipeline iteration waits even before the three-slot ring wraps")
{
    CHECK_FALSE(DescriptorRetirementSlot(0, 3));
    CHECK_FALSE(DescriptorRetirementSlot(1, 3));
    CHECK(DescriptorRetirementSlot(2, 3) == 1);
    CHECK_THROWS_AS(DescriptorRetirementSlot(2, 0), std::invalid_argument);
}

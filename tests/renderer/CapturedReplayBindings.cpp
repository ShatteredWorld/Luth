#include <doctest/doctest.h>
#include "luth/renderer/rendergraph/FrameCapture.h"

using namespace Luth;

TEST_CASE("CapturedReplayBindings: frozen bindings preserve per-view and cascade slices")
{
    CapturedReplayBindings live;
    live.sets[0] = reinterpret_cast<VkDescriptorSet>(uintptr_t(10));
    live.indirectBuffer = reinterpret_cast<VkBuffer>(uintptr_t(20));
    live.indirectOffset = 4096;
    live.regionsPerView = 5; live.regionStride = 4096;
    live.indirectSize = 2 * 5 * 4096 * sizeof(VkDrawIndexedIndirectCommand);
    const auto frozen = live;
    live.sets[0] = reinterpret_cast<VkDescriptorSet>(uintptr_t(11));
    live.indirectBuffer = reinterpret_cast<VkBuffer>(uintptr_t(21));
    live.indirectOffset = 8192;
    CHECK(frozen.sets[0] != live.sets[0]);
    CHECK(frozen.indirectBuffer != live.indirectBuffer);
    CHECK(frozen.DrawOffset(0, 0, 0) == 4096);
    CHECK(frozen.DrawOffset(1, 0, 2) == 4096 + (5 * 4096 + 2) * sizeof(VkDrawIndexedIndirectCommand));
    for (u32 cascade = 0; cascade < 4; ++cascade)
        CHECK(frozen.DrawOffset(1, cascade + 1, 4095)
            == 4096 + ((5 + cascade + 1) * 4096 + 4095) * sizeof(VkDrawIndexedIndirectCommand));
    CHECK_THROWS_AS(frozen.DrawOffset(2, 0, 0), std::out_of_range);
    CHECK_THROWS_AS(frozen.DrawOffset(0, 5, 0), std::invalid_argument);
    CHECK_THROWS_AS(frozen.DrawOffset(0, 0, 4096), std::invalid_argument);
}

TEST_CASE("CapturedReplayBindings: invalid slices and offset overflow are rejected")
{
    CapturedReplayBindings bindings;
    CHECK_THROWS_AS(bindings.DrawOffset(0, 0, 0), std::invalid_argument);
    bindings.indirectBuffer = reinterpret_cast<VkBuffer>(uintptr_t(1));
    bindings.regionStride = 2; bindings.regionsPerView = 1;
    bindings.indirectSize = sizeof(VkDrawIndexedIndirectCommand) - 1;
    CHECK_THROWS_AS(bindings.DrawOffset(0, 0, 0), std::out_of_range);
    bindings.indirectSize = std::numeric_limits<u64>::max();
    bindings.indirectOffset = std::numeric_limits<u64>::max();
    CHECK_THROWS_AS(bindings.DrawOffset(0, 0, 1), std::out_of_range);
    bindings.indirectOffset = 0;
    bindings.regionsPerView = bindings.regionStride = std::numeric_limits<u32>::max();
    CHECK_THROWS_AS(bindings.DrawOffset(std::numeric_limits<u32>::max(), 0, 0), std::out_of_range);
    bindings.regionsPerView = 1;
    CHECK_THROWS_AS(bindings.DrawOffset(std::numeric_limits<u32>::max(), 0, 0), std::out_of_range);
}

TEST_CASE("CapturedReplayBindings: capture reset clears borrowed bindings and identity")
{
    RG::CapturedFrame frame;
    frame.replayBindings.sets[0] = reinterpret_cast<VkDescriptorSet>(uintptr_t(10));
    frame.replayBindings.indirectBuffer = reinterpret_cast<VkBuffer>(uintptr_t(20));
    frame.replayBindings.regionStride = 4096;
    frame.capturedView.id = {7}; frame.capturedView.resourceGeneration = 9;
    frame.Clear();
    CHECK(frame.replayBindings.sets[0] == VK_NULL_HANDLE);
    CHECK(frame.replayBindings.indirectBuffer == VK_NULL_HANDLE);
    CHECK(frame.replayBindings.regionStride == 0);
    CHECK(frame.capturedView.id.value == 0);
    CHECK(frame.capturedView.resourceGeneration == 0);
}

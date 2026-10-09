#include <doctest/doctest.h>
#include "luth/renderer/features/rt/DiDenoiserViewState.h"
#include "luth/renderer/features/rt/RtSunShadowViewState.h"
#include "luth/renderer/backend/vulkan/VulkanViewPool.h"
#include "luth/renderer/subsystems/RtSubsystem.h"
#include "luth/renderer/FrameTargets.h"
using namespace Luth;
namespace
{
    auto MakeShadow(const ViewStateConfig& config)
    {
        auto state = std::make_shared<RtSunShadowViewState>();
        state->width = config.width; state->height = config.height;
        return state;
    }
}
TEST_CASE("RtSunShadowViewState: extent and each sampled source replace only the affected view")
{
    RtSunShadowViewStates states; int waits = 0;
    auto safe = [&] { ++waits; };
    const auto config = RtSunShadowViewState::Config(800, 600, 10, 20);
    auto scene = states.Ensure({1}, config, MakeShadow, safe);
    auto game = states.Ensure({2}, config, MakeShadow, safe);
    CHECK(scene != game); CHECK(waits == 0);
    CHECK(states.Ensure({1}, config, MakeShadow, safe) == scene); CHECK(waits == 0);
    auto depthChanged = states.Ensure({1}, RtSunShadowViewState::Config(800, 600, 11, 20), MakeShadow, safe);
    CHECK(depthChanged != scene); CHECK(waits == 1); CHECK(*states.Find({2}) == game);
    auto normalChanged = states.Ensure({1}, RtSunShadowViewState::Config(800, 600, 11, 21), MakeShadow, safe);
    CHECK(normalChanged != depthChanged); CHECK(waits == 2); CHECK(*states.Find({2}) == game);
    auto resized = states.Ensure({1}, RtSunShadowViewState::Config(1600, 900, 12, 22), MakeShadow, safe);
    CHECK(resized != normalChanged); CHECK(waits == 3); CHECK(resized->width == 1600);
    CHECK(resized->height == 900); CHECK(*states.Find({2}) == game);
    CHECK(states.Ensure({1}, RtSunShadowViewState::Config(1600, 900, 12, 22), MakeShadow, safe) == resized);
    CHECK(waits == 3);
}
TEST_CASE("RtSunShadowViewState: failure preserves prior state and release respects borrowed lifetime")
{
    RtSunShadowViewStates states; int waits = 0; auto safe = [&] { ++waits; };
    const auto config = RtSunShadowViewState::Config(800, 600, 10, 20);
    auto prior = states.Ensure({1}, config, MakeShadow, safe);
    CHECK_THROWS_AS(states.Ensure({1}, RtSunShadowViewState::Config(800, 600, 11, 20),
        [](const ViewStateConfig&) -> std::shared_ptr<RtSunShadowViewState> { throw std::runtime_error("allocation failed"); }, safe), std::runtime_error);
    CHECK(*states.Find({1}) == prior); CHECK(waits == 1);
    std::weak_ptr<RtSunShadowViewState> retained = prior;
    states.Release({1}, safe); CHECK(waits == 2); CHECK(states.Find({1}) == nullptr);
    CHECK_FALSE(retained.expired()); prior.reset(); CHECK(retained.expired());
    states.Release({1}, safe); CHECK(waits == 2);
    states.Ensure({2}, config, MakeShadow, safe);
    states.ReleaseAll(safe); CHECK(waits == 3); CHECK(states.Find({2}) == nullptr);
}
TEST_CASE("RtSunShadowViewState: invalid inputs and dormant domain need no device")
{
    CHECK_THROWS_AS(RtSunShadowViewState::Config(0, 600, 10, 20), std::invalid_argument);
    CHECK_THROWS_AS(RtSunShadowViewState::Config(800, 0, 10, 20), std::invalid_argument);
    CHECK_THROWS_AS(RtSunShadowViewState::Config(800, 600, 0, 20), std::invalid_argument);
    CHECK_THROWS_AS(RtSunShadowViewState::Config(800, 600, 10, 0), std::invalid_argument);
    const auto config = RtSunShadowViewState::Config(800, 600, 10, 20);
    CHECK_THROWS_AS(RtSunShadowViewState::Create({}, config, {}), std::invalid_argument);
    CHECK_THROWS_AS(RtSunShadowViewState::Create({1}, config, {}), std::runtime_error);
    RtSubsystem dormant; FrameTargets targets;
    CHECK(dormant.EnsureShadowView({1}, targets) == nullptr);
    dormant.ReleaseShadowView({1});
}
TEST_CASE("RtSunShadowViewState: local pool removes exactly its descriptors from the shared pool")
{
    CHECK_THROWS_AS(RtSunShadowPoolBudget(0), std::invalid_argument);
    for (const u32 frames : {2u, 3u}) {
        const RtSunShadowPoolBudget local(frames);
        const VulkanViewPool shared(true, frames);
        CHECK(local.maxSets == frames);
        CHECK(local.sizes[0].type == VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
        CHECK(local.sizes[0].descriptorCount == 2 * frames);
        CHECK(local.sizes[1].type == VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);
        CHECK(local.sizes[1].descriptorCount == frames);
        const DiDenoiserPoolBudget diffuse;
        CHECK(shared.maxSets + local.maxSets + diffuse.maxSets == 205 - 13 * frames - 16);
        CHECK(shared.sizes[1].descriptorCount + local.sizes[1].descriptorCount + diffuse.sizes[1].descriptorCount == 248 - 9 * frames - 13);
        CHECK(shared.sizes[2].descriptorCount + local.sizes[0].descriptorCount + diffuse.sizes[0].descriptorCount == 317 - 27 * frames - 21);
    }
}

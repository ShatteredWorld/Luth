#include "luth/renderer/features/rt/ReflectionViewState.h"
#include "luth/renderer/features/rt/GiUpscaleViewState.h"
#include <doctest/doctest.h>
#include "luth/renderer/features/rt/GiDenoiserViewState.h"
#include "luth/renderer/features/rt/RestirGiViewState.h"
#include "luth/renderer/backend/vulkan/VulkanViewPool.h"
#include "luth/renderer/subsystems/RtRestirGiSubsystem.h"
#include "luth/renderer/FrameTargets.h"
using namespace Luth;
namespace
{
    auto MakeGi(const ViewStateConfig& config)
    {
        auto state = std::make_shared<RestirGiViewState>();
        const auto extent = RestirGiViewState::WorkingExtent(config);
        state->width = extent[0]; state->height = extent[1];
        return state;
    }
}
TEST_CASE("RestirGiViewState: working extent preserves full half odd and minimum resolutions")
{
    for (const auto [width, height] : {std::pair{800u, 600u}, std::pair{801u, 601u}, std::pair{1u, 1u}}) {
        auto full = RestirGiViewState::WorkingExtent(RestirGiViewState::Config(width, height, false, 1));
        auto half = RestirGiViewState::WorkingExtent(RestirGiViewState::Config(width, height, true, 1));
        CHECK(full[0] == width); CHECK(full[1] == height);
        CHECK(half[0] == std::max(width / 2, 1u)); CHECK(half[1] == std::max(height / 2, 1u));
    }
}
TEST_CASE("RestirGiViewState: local replacement isolates views and failure preserves prior resources")
{
    RestirGiViewStates states; int waits = 0; auto safe = [&] { ++waits; };
    auto config = RestirGiViewState::Config(800, 600, false, 1);
    auto scene = states.Ensure({1}, config, MakeGi, safe);
    auto game = states.Ensure({2}, config, MakeGi, safe);
    CHECK(scene != game); CHECK(waits == 0);
    CHECK(states.Ensure({1}, config, MakeGi, safe) == scene); CHECK(waits == 0);
    config.signature = 1;
    auto half = states.Ensure({1}, config, MakeGi, safe);
    CHECK(half != scene); CHECK(half->width == 400); CHECK(half->height == 300);
    CHECK(waits == 1); CHECK(*states.Find({2}) == game);
    // Any changed sampled source gets a new generation, independently of extent.
    ++config.resourceGeneration;
    auto rebound = states.Ensure({1}, config, MakeGi, safe);
    CHECK(rebound != half); CHECK(waits == 2); CHECK(*states.Find({2}) == game);
    config.width = 1600;
    CHECK_THROWS_AS(states.Ensure({1}, config,
        [](const ViewStateConfig&) -> std::shared_ptr<RestirGiViewState> { throw std::runtime_error("allocation failed"); }, safe), std::runtime_error);
    CHECK(*states.Find({1}) == rebound); CHECK(waits == 3);
    std::weak_ptr<RestirGiViewState> retained = rebound;
    states.Release({1}, safe); CHECK(waits == 4); CHECK(states.Find({1}) == nullptr);
    CHECK_FALSE(retained.expired()); rebound.reset(); CHECK(retained.expired());
    states.Release({1}, safe); CHECK(waits == 4);
    states.ReleaseAll(safe); CHECK(waits == 5); CHECK(states.Find({2}) == nullptr);
}
TEST_CASE("RestirGiViewState: invalid inputs and dormant native domain require no device")
{
    CHECK_THROWS_AS(RestirGiViewState::Config(0, 600, false, 1), std::invalid_argument);
    CHECK_THROWS_AS(RestirGiViewState::Config(800, 0, false, 1), std::invalid_argument);
    CHECK_THROWS_AS(RestirGiViewState::Config(800, 600, false, 0), std::invalid_argument);
    const auto config = RestirGiViewState::Config(800, 600, false, 1);
    CHECK_THROWS_AS(RestirGiViewState::Create({}, config, {}, {}, 0xFFFF8000u, 0xFFFF8001u), std::invalid_argument);
    CHECK_THROWS_AS(RestirGiViewState::Create({1}, config, {}, {}, 0xFFFF8000u, 0xFFFF8000u), std::invalid_argument);
    CHECK_THROWS_AS(RestirGiViewState::Create({1}, config, {}, {}, 0xFFFF0000u, 0xFFFF8001u), std::invalid_argument);
    CHECK_THROWS_AS(RestirGiViewState::Create({1}, config, {}, {}, 0xFFFF8000u, 0xFFFF8001u), std::runtime_error);
    RtRestirGiSubsystem dormant; FrameTargets targets;
    CHECK(dormant.EnsureView({1}, targets, false) == nullptr);
    dormant.ReleaseView({1});
}
TEST_CASE("RestirGiViewState: local pool includes cycled raw and reservoir visualization bindings")
{
    CHECK_THROWS_AS(RestirGiPoolBudget(0), std::invalid_argument);
    for (const u32 frames : {2u, 3u}) {
        const RestirGiPoolBudget local(frames);
        CHECK(local.maxSets == frames + 1);
        CHECK(local.sizes[0].type == VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
        CHECK(local.sizes[0].descriptorCount == 3 * frames + 1);
        CHECK(local.sizes[1].type == VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);
        CHECK(local.sizes[1].descriptorCount == frames);
        CHECK(local.sizes[2].type == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
        CHECK(local.sizes[2].descriptorCount == 3 * frames + 1);
        const VulkanViewPool shared(true, frames);
        CHECK(shared.maxSets + ReflectionPoolBudget{}.maxSets + GiUpscalePoolBudget{}.maxSets + GiDenoiserPoolBudget{}.maxSets + local.maxSets == 205 - 14 * frames - 32);
        CHECK(shared.sizes[1].descriptorCount + ReflectionPoolBudget{}.sizes[1].descriptorCount + GiUpscalePoolBudget{}.sizes[1].descriptorCount + GiDenoiserPoolBudget{}.sizes[1].descriptorCount + local.sizes[1].descriptorCount == 248 - 10 * frames - 65);
        CHECK(shared.sizes[2].descriptorCount + ReflectionPoolBudget{}.sizes[0].descriptorCount + GiUpscalePoolBudget{}.sizes[0].descriptorCount + GiDenoiserPoolBudget{}.sizes[0].descriptorCount + local.sizes[0].descriptorCount == 317 - 29 * frames - 69);
        CHECK(shared.sizes[3].descriptorCount + local.sizes[2].descriptorCount == 126 - 8 * frames - 1);
    }
}

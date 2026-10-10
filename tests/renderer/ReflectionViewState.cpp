#include "luth/renderer/features/rt/ReflectionUpscaleViewState.h"
#include "luth/renderer/features/rt/ReflectionDenoiserViewState.h"
#include <doctest/doctest.h>
#include "luth/renderer/features/rt/ReflectionViewState.h"
#include "luth/renderer/backend/vulkan/VulkanViewPool.h"
#include "luth/renderer/subsystems/ReflectionsSubsystem.h"
#include "luth/renderer/FrameTargets.h"
using namespace Luth;
namespace {
    auto MakeReflection(const ViewStateConfig& config) {
        auto state = std::make_shared<ReflectionViewState>();
        const auto extent = ReflectionViewState::WorkingExtent(config);
        state->width = extent[0]; state->height = extent[1];
        return state;
    }
}
TEST_CASE("ReflectionViewState: full half odd and narrow extents retain native allocation semantics")
{
    for (const auto [w, h] : {std::pair{800u, 600u}, std::pair{801u, 601u}, std::pair{1u, 600u}, std::pair{800u, 1u}, std::pair{1u, 1u}}) {
        const auto full = ReflectionViewState::WorkingExtent(ReflectionViewState::Config(w, h, false, 1));
        const auto half = ReflectionViewState::WorkingExtent(ReflectionViewState::Config(w, h, true, 1));
        CHECK(full[0] == w); CHECK(full[1] == h);
        CHECK(half[0] == std::max(w / 2, 1u)); CHECK(half[1] == std::max(h / 2, 1u));
    }
}
TEST_CASE("ReflectionViewState: replacement release and failed allocation isolate views")
{
    ReflectionViewStates states; int waits = 0; auto safe = [&] { ++waits; };
    auto config = ReflectionViewState::Config(801, 601, false, 1);
    auto scene = states.Ensure({1}, config, MakeReflection, safe);
    auto game = states.Ensure({2}, config, MakeReflection, safe);
    CHECK(scene != game); CHECK(waits == 0);
    CHECK(states.Ensure({1}, config, MakeReflection, safe) == scene);
    config.signature = 1;
    auto half = states.Ensure({1}, config, MakeReflection, safe);
    CHECK(half != scene); CHECK(half->width == 400); CHECK(half->height == 300); CHECK(waits == 1);
    ++config.resourceGeneration;
    auto rebound = states.Ensure({1}, config, MakeReflection, safe);
    CHECK(rebound != half); CHECK(waits == 2); CHECK(*states.Find({2}) == game);
    ++config.width;
    CHECK_THROWS_AS(states.Ensure({1}, config,
        [](const ViewStateConfig&) -> std::shared_ptr<ReflectionViewState> { throw std::runtime_error("failed"); }, safe), std::runtime_error);
    CHECK(*states.Find({1}) == rebound); CHECK(waits == 3);
    std::weak_ptr<ReflectionViewState> retained = rebound;
    states.Release({1}, safe); CHECK(waits == 4); CHECK(states.Find({1}) == nullptr);
    CHECK_FALSE(retained.expired()); rebound.reset(); CHECK(retained.expired());
    states.Release({1}, safe); CHECK(waits == 4);
    states.ReleaseAll(safe); CHECK(waits == 5); CHECK(states.Find({2}) == nullptr);
}
TEST_CASE("ReflectionViewState: invalid configuration and dormant domain require no Vulkan device")
{
    CHECK_THROWS_AS(ReflectionViewState::Config(0, 1, false, 1), std::invalid_argument);
    CHECK_THROWS_AS(ReflectionViewState::Config(1, 0, false, 1), std::invalid_argument);
    CHECK_THROWS_AS(ReflectionViewState::Config(1, 1, false, 0), std::invalid_argument);
    auto config = ReflectionViewState::Config(1, 1, false, 1);
    CHECK_THROWS_AS(ReflectionViewState::Create({}, config, {}), std::invalid_argument);
    CHECK_THROWS_AS(ReflectionViewState::Create({1}, config, {}), std::runtime_error);
    config.signature = 2;
    CHECK_THROWS_AS(ReflectionViewState::WorkingExtent(config), std::invalid_argument);
    ReflectionsSubsystem dormant; FrameTargets targets;
    CHECK(dormant.EnsureView({1}, targets, false) == nullptr);
    dormant.ReleaseView({1});
}
TEST_CASE("ReflectionViewState: trace descriptors move out of the compatibility pool")
{
    const ReflectionPoolBudget local;
    CHECK(local.maxSets == 1);
    CHECK(local.sizes[0].type == VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
    CHECK(local.sizes[0].descriptorCount == 3);
    CHECK(local.sizes[1].type == VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);
    CHECK(local.sizes[1].descriptorCount == 1);
    for (const u32 frames : {2u, 3u}) {
        const VulkanViewPool shared(true, frames), raster(false, frames);
        CHECK(shared.maxSets + ReflectionUpscalePoolBudget{}.maxSets + ReflectionDenoiserPoolBudget{}.maxSets + local.maxSets == 205 - 15 * frames - 41);
        CHECK(shared.sizes[1].descriptorCount + ReflectionUpscalePoolBudget{}.sizes[1].descriptorCount + ReflectionDenoiserPoolBudget{}.sizes[1].descriptorCount + local.sizes[1].descriptorCount == 248 - 11 * frames - 91);
        CHECK(shared.sizes[2].descriptorCount + ReflectionUpscalePoolBudget{}.sizes[0].descriptorCount + ReflectionDenoiserPoolBudget{}.sizes[0].descriptorCount + local.sizes[0].descriptorCount == 317 - 32 * frames - 94);
        CHECK(raster.maxSets == 4 * frames);
        CHECK(raster.count == 3);
    }
}

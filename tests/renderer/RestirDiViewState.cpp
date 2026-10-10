#include "luth/renderer/features/rt/ReflectionUpscaleViewState.h"
#include "luth/renderer/features/rt/ReflectionDenoiserViewState.h"
#include "luth/renderer/features/rt/ReflectionViewState.h"
#include "luth/renderer/features/rt/GiUpscaleViewState.h"
#include <doctest/doctest.h>
#include "luth/renderer/features/rt/GiDenoiserViewState.h"
#include "luth/renderer/features/rt/RestirGiViewState.h"
#include "luth/renderer/features/rt/DiUpscaleViewState.h"
#include "luth/renderer/features/rt/RestirDiViewState.h"
#include "luth/renderer/backend/vulkan/VulkanViewPool.h"
#include "luth/renderer/subsystems/RtRestirSubsystem.h"
#include "luth/renderer/FrameTargets.h"
using namespace Luth;
namespace
{
    auto MakeDi(const ViewStateConfig& config)
    {
        auto state = std::make_shared<RestirDiViewState>();
        const auto extent = RestirDiViewState::WorkingExtent(config);
        state->width = extent[0]; state->height = extent[1];
        return state;
    }
}
TEST_CASE("RestirDiViewState: working extent preserves full half odd and minimum resolutions")
{
    for (const auto [width, height] : {std::pair{800u, 600u}, std::pair{801u, 601u}, std::pair{1u, 1u}}) {
        auto full = RestirDiViewState::WorkingExtent(RestirDiViewState::Config(width, height, false, 1));
        auto half = RestirDiViewState::WorkingExtent(RestirDiViewState::Config(width, height, true, 1));
        CHECK(full[0] == width); CHECK(full[1] == height);
        CHECK(half[0] == std::max(width / 2, 1u)); CHECK(half[1] == std::max(height / 2, 1u));
    }
}
TEST_CASE("RestirDiViewState: local replacement isolates views and failure preserves prior resources")
{
    RestirDiViewStates states; int waits = 0; auto safe = [&] { ++waits; };
    auto config = RestirDiViewState::Config(800, 600, false, 1);
    auto scene = states.Ensure({1}, config, MakeDi, safe);
    auto game = states.Ensure({2}, config, MakeDi, safe);
    CHECK(scene != game); CHECK(waits == 0);
    CHECK(states.Ensure({1}, config, MakeDi, safe) == scene); CHECK(waits == 0);
    config.signature = 1;
    auto half = states.Ensure({1}, config, MakeDi, safe);
    CHECK(half != scene); CHECK(half->width == 400); CHECK(half->height == 300);
    CHECK(waits == 1); CHECK(*states.Find({2}) == game);
    // Any changed sampled source gets a new generation, independently of extent.
    ++config.resourceGeneration;
    auto rebound = states.Ensure({1}, config, MakeDi, safe);
    CHECK(rebound != half); CHECK(waits == 2); CHECK(*states.Find({2}) == game);
    config.width = 1600;
    CHECK_THROWS_AS(states.Ensure({1}, config,
        [](const ViewStateConfig&) -> std::shared_ptr<RestirDiViewState> { throw std::runtime_error("allocation failed"); }, safe), std::runtime_error);
    CHECK(*states.Find({1}) == rebound); CHECK(waits == 3);
    std::weak_ptr<RestirDiViewState> retained = rebound;
    states.Release({1}, safe); CHECK(waits == 4); CHECK(states.Find({1}) == nullptr);
    CHECK_FALSE(retained.expired()); rebound.reset(); CHECK(retained.expired());
    states.Release({1}, safe); CHECK(waits == 4);
    states.ReleaseAll(safe); CHECK(waits == 5); CHECK(states.Find({2}) == nullptr);
}
TEST_CASE("RestirDiViewState: invalid inputs and dormant native domain require no device")
{
    CHECK_THROWS_AS(RestirDiViewState::Config(0, 600, false, 1), std::invalid_argument);
    CHECK_THROWS_AS(RestirDiViewState::Config(800, 0, false, 1), std::invalid_argument);
    CHECK_THROWS_AS(RestirDiViewState::Config(800, 600, false, 0), std::invalid_argument);
    const auto config = RestirDiViewState::Config(800, 600, false, 1);
    CHECK_THROWS_AS(RestirDiViewState::Create({}, config, {}, 0xFFFF0000u, 0xFFFF0001u), std::invalid_argument);
    CHECK_THROWS_AS(RestirDiViewState::Create({1}, config, {}, 0xFFFF0000u, 0xFFFF0000u), std::invalid_argument);
    CHECK_THROWS_AS(RestirDiViewState::Create({1}, config, {}, 0xFFFF8000u, 0xFFFF0001u), std::invalid_argument);
    CHECK_THROWS_AS(RestirDiViewState::Create({1}, config, {}, 0xFFFF0000u, 0xFFFF0001u), std::runtime_error);
    RtRestirSubsystem dormant; FrameTargets targets;
    CHECK(dormant.EnsureView({1}, targets, false) == nullptr);
    dormant.ReleaseView({1});
}
TEST_CASE("RestirDiViewState: native pool removes exactly DI bindings from the compatibility budget")
{
    CHECK_THROWS_AS(RestirDiPoolBudget(0), std::invalid_argument);
    for (const u32 frames : {2u, 3u}) {
        const RestirDiPoolBudget local(frames); const VulkanViewPool shared(true, frames);
        CHECK(local.maxSets == frames);
        CHECK(local.sizes[0].type == VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
        CHECK(local.sizes[0].descriptorCount == 4 * frames);
        CHECK(local.sizes[1].type == VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);
        CHECK(local.sizes[1].descriptorCount == 2 * frames);
        CHECK(local.sizes[2].type == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
        CHECK(local.sizes[2].descriptorCount == 3 * frames);
        const DiDenoiserPoolBudget diffuse;
        CHECK(shared.maxSets + ReflectionUpscalePoolBudget{}.maxSets + ReflectionDenoiserPoolBudget{}.maxSets + ReflectionPoolBudget{}.maxSets + GiUpscalePoolBudget{}.maxSets + GiDenoiserPoolBudget{}.maxSets + RestirGiPoolBudget(frames).maxSets + local.maxSets + 2 * diffuse.maxSets + DiUpscalePoolBudget{}.maxSets == 205 - 13 * frames - 16);
        CHECK(shared.sizes[1].descriptorCount + ReflectionUpscalePoolBudget{}.sizes[1].descriptorCount + ReflectionDenoiserPoolBudget{}.sizes[1].descriptorCount + ReflectionPoolBudget{}.sizes[1].descriptorCount + GiUpscalePoolBudget{}.sizes[1].descriptorCount + GiDenoiserPoolBudget{}.sizes[1].descriptorCount + RestirGiPoolBudget(frames).sizes[1].descriptorCount + local.sizes[1].descriptorCount + 2 * diffuse.sizes[1].descriptorCount + DiUpscalePoolBudget{}.sizes[1].descriptorCount == 248 - 8 * frames - 13);
        CHECK(shared.sizes[2].descriptorCount + ReflectionUpscalePoolBudget{}.sizes[0].descriptorCount + ReflectionDenoiserPoolBudget{}.sizes[0].descriptorCount + ReflectionPoolBudget{}.sizes[0].descriptorCount + GiUpscalePoolBudget{}.sizes[0].descriptorCount + GiDenoiserPoolBudget{}.sizes[0].descriptorCount + RestirGiPoolBudget(frames).sizes[0].descriptorCount + local.sizes[0].descriptorCount + 2 * diffuse.sizes[0].descriptorCount + DiUpscalePoolBudget{}.sizes[0].descriptorCount == 317 - 25 * frames - 21);
        CHECK(shared.sizes[3].descriptorCount + RestirGiPoolBudget(frames).sizes[2].descriptorCount + local.sizes[2].descriptorCount == 126 - 5 * frames - 1);
    }
}

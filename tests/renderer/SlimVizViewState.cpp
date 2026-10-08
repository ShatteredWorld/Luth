#include <doctest/doctest.h>
#include "luth/renderer/features/SlimVizViewState.h"
#include "luth/renderer/features/BloomViewState.h"
#include "luth/renderer/features/TaaViewState.h"
#include "luth/renderer/subsystems/PostProcessSubsystem.h"
#include "luth/renderer/FrameTargets.h"
using namespace Luth;
namespace
{
    auto MakeState(const ViewStateConfig& config)
    {
        auto state = std::make_shared<SlimVizViewState>(); state->sourceGeneration = config.signature; return state;
    }
}
TEST_CASE("SlimVizViewState: independent views extent and source replacement preserve other domains")
{
    SlimVizViewStateStore states; BloomViewStateStore bloom; TaaViewStateStore taa;
    int waits = 0, allocations = 0; auto safe = [&] { ++waits; };
    auto create = [&](const ViewStateConfig& config) { ++allocations; return MakeState(config); };
    const auto config = SlimVizViewState::Config(800, 600);
    auto scene = states.Ensure({1}, config, create, safe);
    auto game = states.Ensure({2}, SlimVizViewState::Config(1280, 720), create, safe);
    auto pyramid = bloom.Ensure({1}, BloomViewState::Config(800, 600),
        [](const ViewStateConfig&) { return std::make_shared<BloomViewState>(); }, safe);
    auto history = taa.Ensure({1}, TaaViewState::Config(800, 600),
        [](const ViewStateConfig&) { return std::make_shared<TaaViewState>(); }, safe);
    history->history.Commit(5, 1);
    CHECK(scene != game); CHECK(allocations == 2); CHECK(waits == 0);
    CHECK(states.Ensure({1}, config, create, safe) == scene); CHECK(waits == 0); CHECK(allocations == 2);
    auto resized = states.Ensure({1}, SlimVizViewState::Config(1600, 900), create, [&] {
        CHECK(*states.Find({1}) == scene); ++waits;
    });
    CHECK(resized != scene); CHECK(waits == 1);
    auto rebound = states.Ensure({1}, SlimVizViewState::Config(1600, 900, 2), create, safe);
    CHECK(rebound != resized); CHECK(rebound->sourceGeneration == 2); CHECK(waits == 2); CHECK(allocations == 4);
    CHECK(states.Ensure({1}, SlimVizViewState::Config(1600, 900, 2), create, safe) == rebound); CHECK(waits == 2);
    CHECK(*states.Find({2}) == game); CHECK(*bloom.Find({1}) == pyramid);
    CHECK(*taa.Find({1}) == history); CHECK(history->history.CanReuse(6, 1));
}
TEST_CASE("SlimVizViewState: failed replacement release and reopening retain frozen owners")
{
    SlimVizViewStateStore states; int waits = 0; auto safe = [&] { ++waits; };
    const auto config = SlimVizViewState::Config(800, 600);
    auto prior = states.Ensure({1}, config, MakeState, safe);
    CHECK_THROWS_AS(states.Ensure({1}, SlimVizViewState::Config(800, 600, 2),
        [](const ViewStateConfig&) -> std::shared_ptr<SlimVizViewState> { throw std::runtime_error("allocation"); }, safe), std::runtime_error);
    CHECK(*states.Find({1}) == prior); CHECK(waits == 1);
    std::weak_ptr<SlimVizViewState> retained = prior;
    states.Release({1}, safe); CHECK_FALSE(states.Find({1})); CHECK_FALSE(retained.expired()); CHECK(waits == 2);
    states.Release({1}, safe); CHECK(waits == 2);
    auto reopened = states.Ensure({2}, config, MakeState, safe); CHECK(reopened != prior);
    prior.reset(); CHECK(retained.expired()); states.ReleaseAll(safe); CHECK_FALSE(states.Find({2})); CHECK(waits == 3);
}
TEST_CASE("SlimVizViewState: invalid configuration fails before native device access")
{
    CHECK_THROWS_AS(SlimVizViewState::Config(0, 600), std::invalid_argument);
    CHECK_THROWS_AS(SlimVizViewState::Config(800, 0), std::invalid_argument);
    CHECK_THROWS_AS(SlimVizViewState::Config(800, 600, 0), std::invalid_argument);
    const auto config = SlimVizViewState::Config(800, 600);
    CHECK_THROWS_AS(SlimVizViewState::Create({0}, config, VK_NULL_HANDLE), std::invalid_argument);
    CHECK_THROWS_AS(SlimVizViewState::Create({1}, config, VK_NULL_HANDLE), std::runtime_error);
}
TEST_CASE("SlimVizViewState: missing targets and descriptor sources are headless failures")
{
    PostProcessSubsystem native; FrameTargets targets; SlimVizViewState state;
    CHECK_THROWS_AS(native.EnsureSlimVizView({1}, targets), std::invalid_argument);
    CHECK_THROWS_AS(native.WriteSlimVizView(state), std::invalid_argument);
    native.ReleaseSlimVizView({1}); CHECK_FALSE(state.pool); CHECK_FALSE(state.set);
}

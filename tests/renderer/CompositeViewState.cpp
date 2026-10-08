#include <doctest/doctest.h>
#include "luth/renderer/features/CompositeViewState.h"
#include "luth/renderer/features/BloomViewState.h"
#include "luth/renderer/features/TaaViewState.h"
#include "luth/renderer/subsystems/PostProcessSubsystem.h"
#include "luth/renderer/FrameTargets.h"

using namespace Luth;
namespace
{
    auto MakeComposite(const ViewStateConfig& config)
    {
        auto state = std::make_shared<CompositeViewState>();
        state->sourceGeneration = config.signature;
        return state;
    }
}
TEST_CASE("CompositeViewState: local resize and source replacement preserve other domains")
{
    CompositeViewStateStore states; BloomViewStateStore bloom; TaaViewStateStore taa;
    int waits = 0, allocations = 0;
    auto safe = [&] { ++waits; };
    auto factory = [&](const ViewStateConfig& config) { ++allocations; return MakeComposite(config); };
    const auto config = CompositeViewState::Config(800, 600);
    auto scene = states.Ensure({1}, config, factory, safe);
    auto game = states.Ensure({2}, CompositeViewState::Config(1280, 720), factory, safe);
    auto pyramid = bloom.Ensure({1}, BloomViewState::Config(800, 600),
        [](const ViewStateConfig&) { return std::make_shared<BloomViewState>(); }, safe);
    auto history = taa.Ensure({1}, TaaViewState::Config(800, 600),
        [](const ViewStateConfig&) { return std::make_shared<TaaViewState>(); }, safe);
    history->history.Commit(5, 1);
    CHECK(scene != game); CHECK(allocations == 2); CHECK(waits == 0);
    CHECK(states.Ensure({1}, config, factory, safe) == scene); CHECK(allocations == 2); CHECK(waits == 0);
    auto resized = states.Ensure({1}, CompositeViewState::Config(1600, 900), factory, [&] {
        CHECK(*states.Find({1}) == scene); ++waits;
    });
    CHECK(resized != scene); CHECK(waits == 1); CHECK(allocations == 3);
    auto rebound = states.Ensure({1}, CompositeViewState::Config(1600, 900, 2), factory, safe);
    CHECK(rebound != resized); CHECK(rebound->sourceGeneration == 2); CHECK(waits == 2); CHECK(allocations == 4);
    CHECK(states.Ensure({1}, CompositeViewState::Config(1600, 900, 2), factory, safe) == rebound);
    CHECK(waits == 2); CHECK(allocations == 4);
    CHECK(*states.Find({2}) == game); CHECK(*bloom.Find({1}) == pyramid);
    CHECK(*taa.Find({1}) == history); CHECK(history->history.CanReuse(6, 1));
}
TEST_CASE("CompositeViewState: failed replacement and release preserve borrowed state")
{
    CompositeViewStateStore states; int waits = 0; auto safe = [&] { ++waits; };
    auto config = CompositeViewState::Config(800, 600);
    auto prior = states.Ensure({1}, config, MakeComposite, safe);
    CHECK_THROWS_AS(states.Ensure({1}, CompositeViewState::Config(800, 600, 2),
        [](const ViewStateConfig&) -> std::shared_ptr<CompositeViewState> { throw std::runtime_error("allocation failed"); }, safe), std::runtime_error);
    CHECK(*states.Find({1}) == prior); CHECK(waits == 1);
    CHECK(states.Ensure({1}, config, MakeComposite, safe) == prior); CHECK(waits == 1);
    std::weak_ptr<CompositeViewState> retained = prior;
    states.Release({1}, safe); CHECK(waits == 2); CHECK(states.Find({1}) == nullptr); CHECK_FALSE(retained.expired());
    prior.reset(); CHECK(retained.expired());
    states.Release({1}, safe); CHECK(waits == 2);
    auto reopened = states.Ensure({2}, config, MakeComposite, safe);
    CHECK(reopened->sourceGeneration == 1); CHECK(states.Find({1}) == nullptr);
    states.ReleaseAll(safe); CHECK(waits == 3); CHECK(states.Find({2}) == nullptr);
}
TEST_CASE("CompositeViewState: invalid configuration and native inputs fail before device access")
{
    CHECK_THROWS_AS(CompositeViewState::Config(0, 600), std::invalid_argument);
    CHECK_THROWS_AS(CompositeViewState::Config(800, 0), std::invalid_argument);
    CHECK_THROWS_AS(CompositeViewState::Config(800, 600, 0), std::invalid_argument);
    CHECK_THROWS_AS(CompositeViewState::Create({}, CompositeViewState::Config(800, 600), {}), std::invalid_argument);
    CHECK_THROWS_AS(CompositeViewState::Create({1}, CompositeViewState::Config(800, 600), {}), std::runtime_error);
    CHECK_THROWS_AS(CompositeViewState::Create({1}, {800, 600, 0}, {}), std::invalid_argument);
}
TEST_CASE("CompositeViewState: missing stable bindings fail before native writes")
{
    PostProcessSubsystem native; CompositeViewState state; FrameTargets targets;
    CHECK_THROWS_AS(native.EnsureCompositeView({1}, targets, {}), std::invalid_argument);
    CHECK_THROWS_AS(native.EnsureCompositeView({1}, targets, std::make_shared<BloomViewState>()), std::invalid_argument);
    CHECK_THROWS_AS(native.WriteCompositeView(state), std::invalid_argument);
    native.ReleaseCompositeView({99});
}

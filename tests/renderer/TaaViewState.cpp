#include <doctest/doctest.h>
#include "luth/renderer/features/TaaViewState.h"
using namespace Luth;
namespace
{
    auto MakeTaa(const ViewStateConfig& config)
    {
        auto state = std::make_shared<TaaViewState>();
        state->sourceGeneration = config.signature;
        return state;
    }
}
TEST_CASE("TaaViewState: local replacement preserves the other view and rejects stale history")
{
    TaaViewStateStore states; int waits = 0;
    auto safe = [&] { ++waits; };
    const auto config = TaaViewState::Config(800, 600);
    auto scene = states.Ensure({1}, config, MakeTaa, safe);
    auto game = states.Ensure({2}, config, MakeTaa, safe);
    CHECK(scene != game); CHECK(waits == 0);
    scene->history.Commit(10, scene->sourceGeneration);
    CHECK(scene->history.CanReuse(11, scene->sourceGeneration));
    CHECK_FALSE(scene->history.CanReuse(12, scene->sourceGeneration));
    CHECK(states.Ensure({1}, config, MakeTaa, safe) == scene); CHECK(waits == 0);
    auto replacement = states.Ensure({1}, TaaViewState::Config(800, 600, 2), MakeTaa, safe);
    CHECK(replacement != scene); CHECK(waits == 1);
    CHECK(*states.Find({2}) == game); CHECK_FALSE(replacement->history.valid);
    CHECK_FALSE(scene->history.CanReuse(11, replacement->sourceGeneration));
    replacement->history.Commit(11, replacement->sourceGeneration);
    replacement->history.Invalidate();
    CHECK_FALSE(replacement->history.CanReuse(12, replacement->sourceGeneration));
    auto resized = states.Ensure({1}, TaaViewState::Config(1600, 900, 3), MakeTaa, safe);
    CHECK(resized != replacement); CHECK(waits == 2); CHECK_FALSE(resized->history.valid);
}
TEST_CASE("TaaViewState: failed replacement and release retain borrowed state safely")
{
    TaaViewStateStore states; int waits = 0; auto safe = [&] { ++waits; };
    auto prior = states.Ensure({1}, TaaViewState::Config(800, 600), MakeTaa, safe);
    CHECK_THROWS_AS(states.Ensure({1}, TaaViewState::Config(1600, 900),
        [](const ViewStateConfig&) -> std::shared_ptr<TaaViewState> { throw std::runtime_error("allocation failed"); }, safe), std::runtime_error);
    CHECK(*states.Find({1}) == prior); CHECK(waits == 1);
    std::weak_ptr<TaaViewState> retained = prior;
    states.Release({1}, safe); CHECK(waits == 2); CHECK(states.Find({1}) == nullptr);
    CHECK_FALSE(retained.expired()); prior.reset(); CHECK(retained.expired());
    states.Release({1}, safe); CHECK(waits == 2);
    auto reopened = states.Ensure({2}, TaaViewState::Config(800, 600), MakeTaa, safe);
    CHECK_FALSE(reopened->history.valid);
    states.ReleaseAll(safe); CHECK(waits == 3); CHECK(states.Find({2}) == nullptr);
}
TEST_CASE("TaaViewState: invalid configuration fails before device access")
{
    CHECK_THROWS_AS(TaaViewState::Config(0, 600), std::invalid_argument);
    CHECK_THROWS_AS(TaaViewState::Config(800, 0), std::invalid_argument);
    CHECK_THROWS_AS(TaaViewState::Config(800, 600, 0), std::invalid_argument);
    const auto config = TaaViewState::Config(800, 600);
    CHECK_THROWS_AS(TaaViewState::Create({}, config, {}), std::invalid_argument);
    CHECK_THROWS_AS(TaaViewState::Create({1}, config, {}), std::runtime_error);
}

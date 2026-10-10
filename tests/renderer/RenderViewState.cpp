#include <doctest/doctest.h>
#include "luth/renderer/features/RenderViewState.h"
#include <memory>
#include <vector>

using namespace Luth;

TEST_CASE("RenderViewState: resize and reopen reject stale replay")
{
    RenderViewRegistry views;
    int sceneTargets = 0, gameTargets = 0;
    auto scene = views.Register(&sceneTargets);
    auto game = views.Register(&gameTargets);
    CHECK(scene != game);
    CHECK(views.Register(&sceneTargets) == scene);
    CHECK(views.Matches(game, &gameTargets, 1));
    CHECK_FALSE(views.Matches(scene, &gameTargets, 1));
    CHECK(views.Invalidate(game) == 2);
    CHECK(views.Find(&gameTargets) == game);
    CHECK_FALSE(views.Matches(game, &gameTargets, 1));
    CHECK(views.Get(scene)->generation == 1);
    views.Release(game);
    CHECK_FALSE(views.Matches(game, &gameTargets, 2));
    auto reopened = views.Register(&gameTargets);
    CHECK(reopened != game);
    CHECK_FALSE(views.Matches(game, &gameTargets, 1));
    CHECK(views.Matches(reopened, &gameTargets, 1));
    CHECK_THROWS_AS(views.Register(nullptr), std::invalid_argument);
    CHECK_THROWS_AS(views.Invalidate(game), std::invalid_argument);
}

TEST_CASE("RenderViewState: local replacement waits before native retirement")
{
    std::vector<int> events;
    struct Native
    {
        std::vector<int>* events;
        int value;
        ~Native() { events->push_back(-value); }
    };
    FeatureViewStates<std::unique_ptr<Native>> states;
    auto factory = [&](const ViewStateConfig& config) {
        events.push_back(static_cast<int>(config.signature));
        return std::unique_ptr<Native>(new Native{&events, static_cast<int>(config.signature)});
    };
    auto safePoint = [&] { events.push_back(0); };
    RenderViewId scene{10}, game{11};
    ViewStateConfig config{800, 600, 1, 1};
    auto* first = states.Ensure(scene, config, factory, safePoint).get();
    CHECK(states.Ensure(scene, config, factory, safePoint).get() == first);
    states.Ensure(game, config, factory, safePoint);
    CHECK(events == std::vector<int>{1, 1});
    config.signature = 2;
    states.Ensure(scene, config, factory, safePoint);
    CHECK(events == std::vector<int>{1, 1, 0, 2, -1});
    CHECK(states.Find(game)->get()->value == 1);
    states.Release(scene, safePoint);
    CHECK(events == std::vector<int>{1, 1, 0, 2, -1, 0, -2});
    CHECK(states.Find(scene) == nullptr);
    states.Release(scene, safePoint);
    CHECK(events.size() == 7);
    states.Release(game, safePoint);
    CHECK(events.back() == -1);
    CHECK_THROWS_AS(states.Ensure({}, config, factory, safePoint), std::invalid_argument);
}

TEST_CASE("RenderViewState: failed replacement retains prior state")
{
    FeatureViewStates<int> states;
    ViewStateConfig config{100, 100, 1, 1};
    states.Ensure({1}, config, [](const auto&) { return 42; }, [] {});
    ++config.resourceGeneration;
    CHECK_THROWS_AS(states.Ensure({1}, config, [](const auto&) -> int {
        throw std::runtime_error("allocation failed");
    }, [] {}), std::runtime_error);
    REQUIRE(states.Find({1}));
    CHECK(*states.Find({1}) == 42);
}

TEST_CASE("RenderViewState: history uses absolute render frames and generations")
{
    ViewHistoryState history;
    CHECK_FALSE(history.CanReuse(0, 1));
    history.Commit(100, 1);
    CHECK(history.CanReuse(101, 1));
    CHECK_FALSE(history.CanReuse(102, 1)); // missed view frame
    CHECK_FALSE(history.CanReuse(101, 2)); // resize/reconfiguration
    CHECK_FALSE(history.CanReuse(100, 1)); // repeated or unsubmitted frame
    history.Invalidate(); // camera cut, disable, shader reload
    CHECK_FALSE(history.CanReuse(101, 1));
    history.Commit(200, 2);
    CHECK(history.CanReuse(201, 2));
}

TEST_CASE("RenderViewState: extent generation and shutdown have explicit safe points")
{
    FeatureViewStates<int> states;
    int waits = 0, allocations = 0;
    auto wait = [&] { ++waits; };
    auto allocate = [&](const auto&) { return ++allocations; };
    ViewStateConfig config{640, 480, 0, 1};
    states.Ensure({1}, config, allocate, wait);
    states.Ensure({2}, config, allocate, wait);
    CHECK(waits == 0);
    config.width = 800;
    CHECK(states.Ensure({1}, config, allocate, wait) == 3);
    CHECK(waits == 1);
    CHECK(*states.Find({2}) == 2);
    ++config.resourceGeneration;
    CHECK_THROWS_AS(states.Ensure({1}, config, allocate, [] {
        throw std::runtime_error("GPU safe point failed");
    }), std::runtime_error);
    CHECK(allocations == 3);
    CHECK(*states.Find({1}) == 3);
    CHECK(states.Ensure({1}, config, allocate, wait) == 4);
    CHECK(waits == 2);
    states.ReleaseAll(wait);
    CHECK(waits == 3);
    CHECK(states.Find({1}) == nullptr);
    CHECK(states.Find({2}) == nullptr);
    states.ReleaseAll(wait);
    CHECK(waits == 3);
}

#include <doctest/doctest.h>
#include "luth/renderer/features/rt/DiDenoiserViewState.h"
#include "luth/renderer/subsystems/SvgfDenoiser.h"
#include "luth/renderer/FrameTargets.h"

using namespace Luth;

TEST_CASE("DiDenoiserViewState: invalid configuration fails before native allocation")
{
    CHECK_THROWS_AS(DiDenoiserViewState::Config(0, 1, false, 1), std::invalid_argument);
    CHECK_THROWS_AS(DiDenoiserViewState::Config(1, 0, false, 1), std::invalid_argument);
    CHECK_THROWS_AS(DiDenoiserViewState::Config(1, 1, false, 0), std::invalid_argument);
    const auto config = DiDenoiserViewState::Config(321, 181, true, 4);
    CHECK(RestirDiViewState::WorkingExtent(config) == std::array<u32, 2>{160, 90});
    CHECK_THROWS_AS(DiDenoiserViewState::Create({}, config, {}), std::invalid_argument);
    CHECK_THROWS_AS(DiDenoiserViewState::Create({1}, config, {}), std::runtime_error);
    SvgfDenoiser dormant;
    FrameTargets targets;
    CHECK_FALSE(dormant.EnsureDiView({1}, targets, {}));
    dormant.ReleaseDiView({1});
}

TEST_CASE("DiDenoiserViewState: view replacement retains old bindings until borrowers release")
{
    DiDenoiserViewStates states;
    int waits = 0, creates = 0;
    auto safe = [&] { ++waits; };
    auto factory = [&](const ViewStateConfig& c) {
        ++creates;
        auto state = std::make_shared<DiDenoiserViewState>();
        state->sourceGeneration = c.resourceGeneration;
        return state;
    };
    auto config = DiDenoiserViewState::Config(321, 181, false, 1);
    auto first = states.Ensure({1}, config, factory, safe);
    std::weak_ptr<DiDenoiserViewState> old = first;
    CHECK(states.Ensure({1}, config, factory, safe) == first);
    CHECK(creates == 1); CHECK(waits == 0);
    auto secondView = states.Ensure({2}, config, factory, safe);
    CHECK(secondView != first);
    config.resourceGeneration = 2; // Different raw DI or G-buffer binding, same extent.
    auto replacement = states.Ensure({1}, config, factory, safe);
    CHECK(replacement != first); CHECK(waits == 1);
    CHECK_FALSE(old.expired()); first.reset(); CHECK(old.expired());
    CHECK(*states.Find({2}) == secondView);
    config.signature = 1;
    CHECK_THROWS_AS(states.Ensure({1}, config, [](const ViewStateConfig&) -> std::shared_ptr<DiDenoiserViewState> {
        throw std::runtime_error("allocation failed");
    }, safe), std::runtime_error);
    CHECK(*states.Find({1}) == replacement);
    states.Release({1}, safe); CHECK_FALSE(states.Find({1}));
    CHECK(*states.Find({2}) == secondView);
    states.ReleaseAll(safe); CHECK_FALSE(states.Find({2}));
    CHECK(waits == 4);
}

TEST_CASE("DiDenoiserViewState: local descriptor budget covers both temporal and iteration parities")
{
    const DiDenoiserPoolBudget budget;
    CHECK(budget.maxSets == 1 + 2 + 2 + 2);
    CHECK(budget.sizes[0].type == VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
    CHECK(budget.sizes[0].descriptorCount == 1 + 2 * 5 + 2 * 2 + 2 * 3);
    CHECK(budget.sizes[1].type == VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);
    CHECK(budget.sizes[1].descriptorCount == 1 + 2 * 6 + 2 * 3 + 2 * 3);
}

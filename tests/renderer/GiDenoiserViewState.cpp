#include <doctest/doctest.h>
#include "luth/renderer/features/rt/GiDenoiserViewState.h"
#include "luth/renderer/subsystems/SvgfDenoiser.h"
#include "luth/renderer/FrameTargets.h"

using namespace Luth;

TEST_CASE("GiDenoiserViewState: invalid configuration fails before native allocation")
{
    CHECK_THROWS_AS(GiDenoiserViewState::Config(0, 1, false, 1), std::invalid_argument);
    CHECK_THROWS_AS(GiDenoiserViewState::Config(1, 0, false, 1), std::invalid_argument);
    CHECK_THROWS_AS(GiDenoiserViewState::Config(1, 1, false, 0), std::invalid_argument);
    const auto config = GiDenoiserViewState::Config(321, 181, true, 4);
    CHECK(RestirGiViewState::WorkingExtent(config) == std::array<u32, 2>{160, 90});
    CHECK_THROWS_AS(GiDenoiserViewState::Create({}, config, {}), std::invalid_argument);
    CHECK_THROWS_AS(GiDenoiserViewState::Create({1}, config, {}), std::runtime_error);
    SvgfDenoiser dormant;
    FrameTargets targets;
    CHECK_FALSE(dormant.EnsureGiView({1}, targets, {}));
    dormant.ReleaseGiView({1});
}

TEST_CASE("GiDenoiserViewState: view replacement retains old bindings until borrowers release")
{
    GiDenoiserViewStates states;
    int waits = 0, creates = 0;
    auto safe = [&] { ++waits; };
    auto factory = [&](const ViewStateConfig& c) {
        ++creates;
        auto state = std::make_shared<GiDenoiserViewState>();
        state->sourceGeneration = c.resourceGeneration;
        return state;
    };
    auto config = GiDenoiserViewState::Config(321, 181, false, 1);
    auto first = states.Ensure({1}, config, factory, safe);
    std::weak_ptr<GiDenoiserViewState> old = first;
    CHECK(states.Ensure({1}, config, factory, safe) == first);
    CHECK(creates == 1); CHECK(waits == 0);
    auto secondView = states.Ensure({2}, config, factory, safe);
    CHECK(secondView != first);
    config.resourceGeneration = 2; // Different raw GI or G-buffer binding, same extent.
    auto replacement = states.Ensure({1}, config, factory, safe);
    CHECK(replacement != first); CHECK(waits == 1);
    CHECK_FALSE(old.expired()); first.reset(); CHECK(old.expired());
    CHECK(*states.Find({2}) == secondView);
    config.signature = 1;
    CHECK_THROWS_AS(states.Ensure({1}, config, [](const ViewStateConfig&) -> std::shared_ptr<GiDenoiserViewState> {
        throw std::runtime_error("allocation failed");
    }, safe), std::runtime_error);
    CHECK(*states.Find({1}) == replacement);
    states.Release({1}, safe); CHECK_FALSE(states.Find({1}));
    CHECK(*states.Find({2}) == secondView);
    states.ReleaseAll(safe); CHECK_FALSE(states.Find({2}));
    CHECK(waits == 4);
}

TEST_CASE("GiDenoiserViewState: local descriptor budget covers both temporal and iteration parities")
{
    const GiDenoiserPoolBudget budget;
    CHECK(budget.maxSets == 1 + 2 + 2 + 2);
    CHECK(budget.sizes[0].type == VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
    CHECK(budget.sizes[0].descriptorCount == 1 + 2 * 5 + 2 * 2 + 2 * 3);
    CHECK(budget.sizes[1].type == VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);
    CHECK(budget.sizes[1].descriptorCount == 1 + 2 * 6 + 2 * 3 + 2 * 3);
}

TEST_CASE("GiDenoiserViewState: retained raw input is independent of DI and released with borrowers")
{
    auto raw = std::make_shared<RestirGiViewState>();
    raw->restirGiDI = std::shared_ptr<Texture>(reinterpret_cast<Texture*>(1), [](Texture*) {});
    GiDenoiserViewState state;
    CHECK(state.Noisy() == nullptr);
    state.input = raw;
    CHECK(state.Noisy() == &raw->restirGiDI);
    std::weak_ptr<RestirGiViewState> retained = raw;
    raw.reset(); CHECK_FALSE(retained.expired());
    state.input.reset(); CHECK(retained.expired()); CHECK(state.Noisy() == nullptr);
    SvgfDenoiser dormant(DenoiserChannel::Gi); FrameTargets targets;
    CHECK_FALSE(dormant.EnsureGiView({1}, targets, {})); dormant.ReleaseGiView({1});
}

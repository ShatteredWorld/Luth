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
TEST_CASE("DiDenoiserViewState: signals select distinct raw producers without native allocation")
{
    auto raw = std::make_shared<RestirDiViewState>();
    raw->restirDI = std::shared_ptr<Texture>(reinterpret_cast<Texture*>(1), [](Texture*) {});
    raw->restirDISpec = std::shared_ptr<Texture>(reinterpret_cast<Texture*>(2), [](Texture*) {});
    DiDenoiserViewState diffuse, specular;
    diffuse.input = specular.input = raw;
    specular.signal = DiDenoiserSignal::Specular;
    CHECK(diffuse.Noisy() == &raw->restirDI);
    CHECK(specular.Noisy() == &raw->restirDISpec);
    CHECK(*diffuse.Noisy() != *specular.Noisy());
    specular.input.reset(); CHECK(specular.Noisy() == nullptr);
    const auto config = DiDenoiserViewState::Config(1279, 719, true, 1);
    CHECK_THROWS_AS(DiDenoiserViewState::Create({1}, config, {}, static_cast<DiDenoiserSignal>(2)), std::invalid_argument);
    CHECK_THROWS_AS(DiDenoiserViewState::Create({1}, config, {}, DiDenoiserSignal::Specular), std::runtime_error);
    SvgfDenoiser dormant(DenoiserChannel::DiSpecular);
    FrameTargets targets;
    CHECK_FALSE(dormant.EnsureDiView({1}, targets, raw)); dormant.ReleaseDiView({1});
}

TEST_CASE("DiDenoiserViewState: diffuse and specular stores preserve channel and view isolation")
{
    DiDenoiserViewStates diffuse, specular;
    int waits = 0;
    auto safe = [&] { ++waits; };
    auto make = [](DiDenoiserSignal signal) {
        return [signal](const ViewStateConfig& config) {
            auto state = std::make_shared<DiDenoiserViewState>();
            state->signal = signal; state->sourceGeneration = config.resourceGeneration;
            return state;
        };
    };
    auto config = DiDenoiserViewState::Config(1279, 719, false, 1);
    auto d = diffuse.Ensure({1}, config, make(DiDenoiserSignal::Diffuse), safe);
    auto s = specular.Ensure({1}, config, make(DiDenoiserSignal::Specular), safe);
    auto game = specular.Ensure({2}, config, make(DiDenoiserSignal::Specular), safe);
    CHECK(d != s); CHECK(s != game);
    CHECK(d->signal == DiDenoiserSignal::Diffuse); CHECK(s->signal == DiDenoiserSignal::Specular);
    CHECK(specular.Ensure({1}, config, make(DiDenoiserSignal::Specular), safe) == s);
    CHECK(waits == 0);
    config.signature = 1; // Half resolution changes only this channel's store entry.
    auto replacement = specular.Ensure({1}, config, make(DiDenoiserSignal::Specular), safe);
    CHECK(replacement != s); CHECK(*diffuse.Find({1}) == d); CHECK(*specular.Find({2}) == game);
    CHECK(waits == 1);
    std::weak_ptr<DiDenoiserViewState> retained = s;
    CHECK_FALSE(retained.expired()); s.reset(); CHECK(retained.expired());
    config.resourceGeneration = 2;
    CHECK_THROWS_AS(specular.Ensure({1}, config, [](const auto&) -> std::shared_ptr<DiDenoiserViewState> {
        throw std::runtime_error("allocation failed");
    }, safe), std::runtime_error);
    CHECK(*specular.Find({1}) == replacement); CHECK(*diffuse.Find({1}) == d);
    specular.Release({1}, safe); CHECK_FALSE(specular.Find({1})); CHECK(*specular.Find({2}) == game);
    diffuse.ReleaseAll(safe); specular.ReleaseAll(safe);
    CHECK_FALSE(diffuse.Find({1})); CHECK_FALSE(specular.Find({2})); CHECK(waits == 5);
}

#include <doctest/doctest.h>
#include "luth/core/diagnostics/Log.h"
#include "luth/renderer/features/TransparencyViewState.h"
#include "luth/renderer/features/FogViewState.h"
#include "luth/renderer/subsystems/TransparencySubsystem.h"

using namespace Luth;
namespace
{
    auto Factory(const ViewStateConfig& config)
    {
        auto state = std::make_shared<TransparencyViewState>();
        state->width = config.width; state->height = config.height;
        state->oitLayersCached = static_cast<u32>(config.signature);
        state->oitNodes.size = TransparencyViewState::NodeBytes(config);
        return state;
    }
}
TEST_CASE("TransparencyViewState: budget replacement is local and GPU safe")
{
    TransparencyViewStateStore states; int waits = 0, allocations = 0;
    auto factory = [&](const ViewStateConfig& config) { ++allocations; return Factory(config); };
    auto safe = [&] { ++waits; };
    const auto config = TransparencyViewState::Config(800, 600, 4);
    auto scene = states.Ensure({1}, config, factory, safe);
    auto game = states.Ensure({2}, config, factory, safe);
    CHECK(scene != game); CHECK(waits == 0); CHECK(allocations == 2);
    CHECK(states.Ensure({1}, config, factory, safe) == scene);
    CHECK(waits == 0); CHECK(allocations == 2);
    auto larger = TransparencyViewState::Config(800, 600, 8);
    auto replacement = states.Ensure({1}, larger, factory, safe);
    CHECK(replacement != scene); CHECK(*states.Find({2}) == game);
    CHECK(waits == 1); CHECK(allocations == 3);
    CHECK(replacement->oitNodes.size == 16ull + 800ull * 600 * 8 * 16);
    CHECK(replacement->oitLayersCached == 8);
    // Ordinary mode/resolve-count edits do not enter the allocation signature.
    CHECK(states.Ensure({1}, larger, factory, safe) == replacement);
    CHECK(waits == 1);
}
TEST_CASE("TransparencyViewState: resize release reopen preserve borrowed state")
{
    TransparencyViewStateStore states; int waits = 0; auto safe = [&] { ++waits; };
    auto config = TransparencyViewState::Config(800, 600, 4);
    auto borrowed = states.Ensure({10}, config, Factory, safe);
    std::weak_ptr<TransparencyViewState> old = borrowed;
    auto resized = TransparencyViewState::Config(1600, 900, 4);
    auto current = states.Ensure({10}, resized, Factory, safe);
    CHECK(current != borrowed); CHECK(waits == 1); CHECK_FALSE(old.expired());
    borrowed.reset(); CHECK(old.expired());
    states.Release({10}, safe); CHECK(waits == 2); CHECK(states.Find({10}) == nullptr);
    states.Release({10}, safe); CHECK(waits == 2);
    CHECK(states.Ensure({11}, resized, Factory, safe) != current);
    CHECK(states.Find({10}) == nullptr);
    states.ReleaseAll(safe); CHECK(waits == 3); CHECK(states.Find({11}) == nullptr);
}
TEST_CASE("TransparencyViewState: failed replacement keeps prior resources")
{
    TransparencyViewStateStore states; int waits = 0;
    auto config = TransparencyViewState::Config(800, 600, 4);
    auto prior = states.Ensure({1}, config, Factory, [] {});
    auto changed = TransparencyViewState::Config(800, 600, 8);
    CHECK_THROWS_AS(states.Ensure({1}, changed,
        [](const ViewStateConfig&) -> std::shared_ptr<TransparencyViewState> { throw std::runtime_error("allocation failed"); },
        [&] { ++waits; }), std::runtime_error);
    CHECK(*states.Find({1}) == prior); CHECK(waits == 1);
    CHECK(states.Ensure({1}, config, Factory, [&] { ++waits; }) == prior); CHECK(waits == 1);
}
TEST_CASE("TransparencyViewState: node layout and invalid input checked before device access")
{
    CHECK(TransparencyViewState::NodeBytes(TransparencyViewState::Config(1, 1, 1)) == 32);
    CHECK(TransparencyViewState::NodeBytes(TransparencyViewState::Config(1920, 1080, 16)) == 16ull + 1920ull * 1080 * 16 * 16);
    CHECK_THROWS_AS(TransparencyViewState::Config(0, 600, 4), std::invalid_argument);
    CHECK_THROWS_AS(TransparencyViewState::Config(800, 0, 4), std::invalid_argument);
    CHECK_THROWS_AS(TransparencyViewState::Config(800, 600, 0), std::invalid_argument);
    CHECK_THROWS_AS(TransparencyViewState::Config(800, 600, 17), std::invalid_argument);
    CHECK_THROWS_AS(TransparencyViewState::Config(~0u, ~0u, 16), std::invalid_argument);
    CHECK_THROWS_AS(TransparencyViewState::NodeBytes({800, 600, u64(1) << 32}), std::invalid_argument);
    const auto config = TransparencyViewState::Config(800, 600, 4);
    CHECK_THROWS_AS(TransparencyViewState::Create({}, config, {}, 0xFFFFC000u), std::invalid_argument);
    CHECK_THROWS_AS(TransparencyViewState::Create({1}, config, {}, 1), std::invalid_argument);
    CHECK_THROWS_AS(TransparencyViewState::Create({1}, config, {}, 0xFFFFC000u), std::runtime_error);
}
TEST_CASE("TransparencyViewState: descriptor slots retain sampled fog ownership")
{
    TransparencyViewState state;
    auto source = std::make_shared<FogViewState>(); std::weak_ptr<FogViewState> old = source;
    state.fogBindings[0] = source; state.fogBindings[1] = source;
    source.reset(); CHECK_FALSE(old.expired());
    state.fogBindings[0] = std::make_shared<FogViewState>(); CHECK_FALSE(old.expired());
    state.fogBindings[1].reset(); CHECK(old.expired());
}
TEST_CASE("TransparencyViewState: node tags stay in existing persistent range")
{
    TransparencySubsystem native;
    CHECK(native.NextNodePoolTag() == 0xFFFFC000u);
    CHECK(native.NextNodePoolTag() == 0xFFFFC001u);
}

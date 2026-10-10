#include <doctest/doctest.h>
#include "luth/core/diagnostics/Log.h"
#include "luth/renderer/features/FogViewState.h"

using namespace Luth;

TEST_CASE("FogViewState: quality and physical depth control local replacement")
{
    FogViewStateStore states;
    int waits = 0, allocations = 0;
    auto factory = [&](const ViewStateConfig& config) {
        ++allocations;
        auto state = std::make_shared<FogViewState>(); // No Vulkan device in headless tests.
        const auto dims = Volumetric::GetAtlasDims(static_cast<VolumetricSettings::Quality>(config.signature));
        state->volDimX = dims.x; state->volDimY = dims.y; state->volDimZ = dims.z;
        state->volQualityCached = static_cast<u32>(config.signature);
        return state;
    };
    auto safePoint = [&] { ++waits; };
    auto config = FogViewState::Config(1920, 1080, VolumetricSettings::Quality::High);
    config.resourceGeneration = 100; // Native physical depth identity.
    auto scene = states.Ensure({1}, config, factory, safePoint);
    auto game = states.Ensure({2}, config, factory, safePoint);
    CHECK(scene != game);
    CHECK(scene->volDimX == 240);
    CHECK(scene->volDimY == 135);
    CHECK(scene->volDimZ == 192);
    CHECK(states.Ensure({1}, config, factory, safePoint) == scene);
    CHECK(waits == 0);
    CHECK(allocations == 2);
    auto low = FogViewState::Config(1920, 1080, VolumetricSettings::Quality::Low);
    low.resourceGeneration = config.resourceGeneration;
    auto replacement = states.Ensure({1}, low, factory, safePoint);
    CHECK(replacement != scene);
    CHECK(*states.Find({2}) == game);
    CHECK(replacement->volDimX == 80);
    CHECK(replacement->volDimY == 45);
    CHECK(replacement->volDimZ == 64);
    CHECK(waits == 1);
    // Rebinding a new depth image at the same extent still needs a safe point.
    ++low.resourceGeneration;
    CHECK(states.Ensure({1}, low, factory, safePoint) != replacement);
    CHECK(waits == 2);
    CHECK(allocations == 4);
}

TEST_CASE("FogViewState: resize release and reopen isolate borrowed state")
{
    FogViewStateStore states;
    int waits = 0;
    auto safePoint = [&] { ++waits; };
    auto factory = [](const ViewStateConfig&) { return std::make_shared<FogViewState>(); };
    auto config = FogViewState::Config(800, 600, VolumetricSettings::Quality::Medium);
    auto borrowed = states.Ensure({10}, config, factory, safePoint);
    std::weak_ptr<FogViewState> previous = borrowed;
    auto resized = FogViewState::Config(1600, 900, VolumetricSettings::Quality::Medium);
    CHECK(states.Ensure({10}, resized, factory, safePoint) != borrowed);
    CHECK_FALSE(previous.expired()); // The compatibility bridge can retain old bindings.
    borrowed.reset();
    CHECK(previous.expired());
    states.Release({10}, safePoint);
    CHECK(states.Find({10}) == nullptr);
    CHECK(waits == 2);
    states.Release({10}, safePoint);
    CHECK(waits == 2);
    states.Ensure({11}, resized, factory, safePoint);
    CHECK(states.Find({11}) != nullptr);
    CHECK(states.Find({10}) == nullptr);
    states.ReleaseAll(safePoint);
    CHECK(waits == 3);
    CHECK(states.Find({11}) == nullptr);
}

TEST_CASE("FogViewState: invalid allocation configurations fail before device access")
{
    CHECK_THROWS_AS(FogViewState::Config(0, 600, VolumetricSettings::Quality::High), std::invalid_argument);
    CHECK_THROWS_AS(FogViewState::Config(800, 0, VolumetricSettings::Quality::High), std::invalid_argument);
    CHECK_THROWS_AS(FogViewState::Config(800, 600, static_cast<VolumetricSettings::Quality>(3)), std::invalid_argument);
    CHECK(FogViewState::Config(800, 600, VolumetricSettings::Quality::Medium).signature == 1);
    const auto config = FogViewState::Config(800, 600, VolumetricSettings::Quality::High);
    CHECK_THROWS_AS(FogViewState::Create({}, config, {}), std::invalid_argument);
    CHECK_THROWS_AS(FogViewState::Create({1}, config, {}), std::runtime_error);
}

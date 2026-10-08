#include <doctest/doctest.h>
#include "luth/renderer/features/BloomViewState.h"
#include "luth/renderer/features/TaaViewState.h"

using namespace Luth;
namespace
{
    auto MakeBloom(const ViewStateConfig& config)
    {
        auto state = std::make_shared<BloomViewState>();
        state->width = config.width; state->height = config.height;
        return state;
    }
}
TEST_CASE("BloomViewState: six half-resolution extents preserve odd sizes and minimum one")
{
    const auto config = BloomViewState::Config(801, 601);
    const std::array<std::pair<u32, u32>, 6> expected{{{400, 300}, {200, 150}, {100, 75}, {50, 37}, {25, 18}, {12, 9}}};
    for (u32 mip = 0; mip < expected.size(); ++mip)
    {
        CHECK(BloomViewState::MipExtent(config, mip) == expected[mip]);
        const auto tiny = BloomViewState::MipExtent(BloomViewState::Config(1, 3), mip);
        CHECK(tiny.first == 1); CHECK(tiny.second == 1);
    }
    const auto last = BloomViewState::MipExtent(BloomViewState::Config(1920, 1080), 5);
    CHECK(last.first == 30); CHECK(last.second == 16);
}
TEST_CASE("BloomViewState: resizing replaces only that view at a safe point")
{
    BloomViewStateStore bloom; TaaViewStateStore taa; int waits = 0, allocations = 0;
    auto safe = [&] { ++waits; };
    auto factory = [&](const ViewStateConfig& config) { ++allocations; return MakeBloom(config); };
    auto config = BloomViewState::Config(800, 600);
    auto scene = bloom.Ensure({1}, config, factory, safe);
    auto game = bloom.Ensure({2}, BloomViewState::Config(1280, 720), factory, safe);
    auto history = taa.Ensure({1}, TaaViewState::Config(800, 600),
        [](const ViewStateConfig&) { return std::make_shared<TaaViewState>(); }, safe);
    history->history.Commit(5, 1);
    CHECK(scene != game); CHECK(allocations == 2); CHECK(waits == 0);
    CHECK(bloom.Ensure({1}, config, factory, safe) == scene); CHECK(allocations == 2); CHECK(waits == 0);
    auto replacement = bloom.Ensure({1}, BloomViewState::Config(1600, 900), factory, [&] {
        CHECK(*bloom.Find({1}) == scene); ++waits;
    });
    CHECK(replacement != scene); CHECK(allocations == 3); CHECK(waits == 1);
    CHECK(replacement->width == 1600); CHECK(replacement->height == 900);
    CHECK(*bloom.Find({2}) == game); CHECK(*taa.Find({1}) == history); CHECK(history->history.CanReuse(6, 1));
    CHECK(bloom.Ensure({1}, BloomViewState::Config(1600, 900), factory, safe) == replacement);
    CHECK(allocations == 3); CHECK(waits == 1);
}
TEST_CASE("BloomViewState: failed replacement release and reopen preserve borrowed ownership")
{
    BloomViewStateStore states; int waits = 0; auto safe = [&] { ++waits; };
    auto config = BloomViewState::Config(800, 600);
    auto prior = states.Ensure({1}, config, MakeBloom, safe);
    CHECK_THROWS_AS(states.Ensure({1}, BloomViewState::Config(1600, 900),
        [](const ViewStateConfig&) -> std::shared_ptr<BloomViewState> { throw std::runtime_error("allocation failed"); }, safe), std::runtime_error);
    CHECK(*states.Find({1}) == prior); CHECK(waits == 1);
    CHECK(states.Ensure({1}, config, MakeBloom, safe) == prior); CHECK(waits == 1);
    std::weak_ptr<BloomViewState> retained = prior;
    states.Release({1}, safe); CHECK(waits == 2); CHECK(states.Find({1}) == nullptr); CHECK_FALSE(retained.expired());
    prior.reset(); CHECK(retained.expired());
    states.Release({1}, safe); CHECK(waits == 2);
    auto reopened = states.Ensure({2}, config, MakeBloom, safe);
    CHECK(reopened->width == 800); CHECK(states.Find({1}) == nullptr);
    states.ReleaseAll(safe); CHECK(waits == 3); CHECK(states.Find({2}) == nullptr);
}
TEST_CASE("BloomViewState: invalid extent pyramid and native inputs fail before device access")
{
    CHECK_THROWS_AS(BloomViewState::Config(0, 600), std::invalid_argument);
    CHECK_THROWS_AS(BloomViewState::Config(800, 0), std::invalid_argument);
    CHECK_THROWS_AS(BloomViewState::MipExtent({800, 600, 5}, 0), std::invalid_argument);
    CHECK_THROWS_AS(BloomViewState::MipExtent(BloomViewState::Config(800, 600), 6), std::invalid_argument);
    CHECK_THROWS_AS(BloomViewState::Create({}, BloomViewState::Config(800, 600), {}), std::invalid_argument);
    CHECK_THROWS_AS(BloomViewState::Create({1}, BloomViewState::Config(800, 600), {}), std::runtime_error);
}

#include <doctest/doctest.h>
#include "luth/renderer/features/rt/TemporalSignalHistory.h"
#include "luth/renderer/features/rt/RestirGiBindings.h"
#include "luth/renderer/features/rt/GiDenoiserBindings.h"
#include "luth/renderer/shader/SlangCompiler.h"
#include <filesystem>

using namespace Luth;
using GiTemporalHistory = TemporalSignalHistory;

TEST_CASE("GiTemporalHistory: provisional work requires matching successful submission")
{
    GiTemporalHistory history;
    CHECK_FALSE(history.CanReuse(1, 7, true));
    history.Record(1, 7);
    CHECK_FALSE(history.CanReuse(2, 7, true));
    history.Finish(1, 7, false);
    CHECK_FALSE(history.CanReuse(2, 7, true));
    history.Record(2, 7);
    history.Finish(2, 7, true);
    CHECK(history.CanReuse(3, 7, true));
    CHECK_FALSE(history.CanReuse(3, 7, false));
    CHECK_FALSE(history.CanReuse(4, 7, true));
    CHECK_FALSE(history.CanReuse(3, 8, true));
    history.Record(3, 7);
    history.Finish(4, 7, true);
    CHECK_FALSE(history.committed.valid);
    history.Record(5, 7);
    history.Finish(5, 8, true);
    CHECK_FALSE(history.committed.valid);
}

TEST_CASE("GiTemporalHistory: absence bypass reload and failed graph discard history")
{
    GiTemporalHistory raw, diffuse, specular;
    for (auto* history : {&raw, &diffuse, &specular}) {
        history->Record(10, 3); history->Finish(10, 3, true);
        REQUIRE(history->CanReuse(11, 3, true));
        history->Begin();
    }
    // Specular is absent, diffuse filtering bypasses, raw still produces.
    raw.Record(11, 3);
    for (auto* history : {&raw, &diffuse, &specular}) history->Finish(11, 3, true);
    CHECK(raw.CanReuse(12, 3, true));
    CHECK_FALSE(diffuse.CanReuse(12, 3, true));
    CHECK_FALSE(specular.CanReuse(12, 3, true));
    raw.Record(12, 3); raw.Invalidate(); // Successful producer shader reload.
    raw.Finish(12, 3, true);
    CHECK_FALSE(raw.CanReuse(13, 3, true));
    diffuse.Record(12, 3); diffuse.Begin(); // Failed build did not record this graph.
    diffuse.Finish(12, 3, true);
    CHECK_FALSE(diffuse.committed.valid);
}

TEST_CASE("GiTemporalHistory: independent views preserve absolute frame continuity")
{
    FeatureViewStates<GiTemporalHistory> views;
    auto make = [](const ViewStateConfig&) { return GiTemporalHistory{}; };
    auto safe = [] {};
    auto& scene = views.Ensure({1}, {64, 64, 0, 1}, make, safe);
    auto& game = views.Ensure({2}, {32, 32, 0, 1}, make, safe);
    scene.Record(20, 1); scene.Finish(20, 1, true);
    game.Record(19, 1); game.Finish(19, 1, true);
    CHECK(scene.CanReuse(21, 1, true));
    CHECK_FALSE(game.CanReuse(21, 1, true));
    views.ForEach([](auto& history) { history.Invalidate(); });
    CHECK_FALSE(scene.committed.valid); CHECK_FALSE(game.committed.valid);
    scene.Record(21, 1); scene.Finish(21, 1, true);
    auto& resized = views.Ensure({1}, {128, 64, 0, 2}, make, safe);
    CHECK_FALSE(resized.committed.valid);
}

TEST_CASE("GiTemporalHistory: production reset guards compile with unchanged native ABI")
{
    REQUIRE(SlangCompiler::Available());
    for (const char* file : {"restir_gi_temporal.slang", "svgf_reproject.slang"}) {
        const auto code = SlangCompiler::CompileReflectStage(std::filesystem::absolute(
            std::filesystem::path("engine/assets/shaders") / file));
        CHECK_FALSE(code.spirv.empty());
        CHECK(code.stage == ShaderStage::Compute);
    }
}
TEST_CASE("GiTemporalHistory: raw and filtered owners gate independent histories and packed reset ABI")
{
    RestirGiViewState raw;
    GiDenoiserViewState filtered;
    raw.history.Record(40, 9); raw.history.Finish(40, 9, true);
    filtered.history.Record(40, 9); filtered.history.Finish(40, 9, true);
    CHECK(raw.history.CanReuse(41, 9, true));
    CHECK(filtered.history.CanReuse(41, 9, true));
    CHECK_FALSE(filtered.history.CanReuse(41, 9, false)); // Camera cut.
    raw.history.Begin(); filtered.history.Begin();
    raw.history.Record(41, 9);
    raw.history.Finish(41, 9, true); filtered.history.Finish(41, 9, true); // Raw-copy bypass.
    CHECK(raw.history.CanReuse(42, 9, true)); CHECK_FALSE(filtered.history.CanReuse(42, 9, true));
    raw.history.Invalidate(); // Raw producer reload must also prevent denoiser reuse.
    CHECK_FALSE(raw.history.CanReuse(42, 9, true));
    RestirGiBindings packet;
    packet.settings.temporalMCap = 0x12345u; packet.settings.maxReservoirAge = 0x23456u;
    CHECK(packet.TemporalCapAndAge() == 0x34560000u); // Reset leaves max-age bits intact.
    packet.historyValid = true;
    CHECK(packet.TemporalCapAndAge() == 0x34562345u);
    raw.history.Record((1ull << 32) + 7, 10); raw.history.Finish((1ull << 32) + 7, 10, true);
    CHECK(raw.history.CanReuse((1ull << 32) + 8, 10, true));
    CHECK_FALSE(raw.history.CanReuse(8, 10, true));
}


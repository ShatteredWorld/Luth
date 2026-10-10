#include <doctest/doctest.h>
#include "luth/renderer/features/rt/TemporalSignalHistory.h"
#include "luth/renderer/features/rt/ReflectionBindings.h"
#include "luth/renderer/features/rt/ReflectionDenoiserBindings.h"
#include "luth/renderer/shader/SlangCompiler.h"
#include <filesystem>
#include <fstream>

using namespace Luth;
using ReflectionTemporalHistory = TemporalSignalHistory;

TEST_CASE("ReflectionTemporalHistory: provisional work requires matching successful submission")
{
    ReflectionTemporalHistory history;
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

TEST_CASE("ReflectionTemporalHistory: absence bypass reload and failed graph discard history")
{
    ReflectionTemporalHistory raw, diffuse, specular;
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

TEST_CASE("ReflectionTemporalHistory: independent views preserve absolute frame continuity")
{
    FeatureViewStates<ReflectionTemporalHistory> views;
    auto make = [](const ViewStateConfig&) { return ReflectionTemporalHistory{}; };
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


TEST_CASE("ReflectionTemporalHistory: raw reload camera cuts and bypass invalidate filtered reuse")
{
    ReflectionViewState raw;
    ReflectionDenoiserViewState filtered;
    for (auto* history : {&raw.history, &filtered.history}) {
        history->Record(40, 9); history->Finish(40, 9, true);
        REQUIRE(history->CanReuse(41, 9, true));
    }
    CHECK_FALSE(filtered.history.CanReuse(41, 9, false));
    raw.history.Begin(); filtered.history.Begin();
    raw.history.Record(41, 9);
    raw.history.Finish(41, 9, true); filtered.history.Finish(41, 9, true);
    CHECK(raw.history.CanReuse(42, 9, true));
    CHECK_FALSE(filtered.history.CanReuse(42, 9, true)); // Copy-only bypass.
    raw.history.Invalidate();
    CHECK_FALSE(raw.history.CanReuse(42, 9, true));
    ReflectionDenoiserBindings packet;
    packet.settings.historyCap = 32;
    CHECK(packet.HistoryCap() == 0.0f);
    packet.historyValid = true;
    CHECK(packet.HistoryCap() == 32.0f);
    filtered.history.Record((1ull << 32) + 7, 10);
    filtered.history.Finish((1ull << 32) + 7, 10, true);
    CHECK(filtered.history.CanReuse((1ull << 32) + 8, 10, true));
    CHECK_FALSE(filtered.history.CanReuse(8, 10, true));
}

TEST_CASE("ReflectionTemporalHistory: production reset guard precedes all history loads and compiles")
{
    const auto path = std::filesystem::absolute("engine/assets/shaders/svgf_spec_reproject.slang");
    std::ifstream stream(path);
    REQUIRE(stream);
    const std::string source((std::istreambuf_iterator<char>(stream)), {});
    const auto guard = source.find("if (pc.historyCap > 0.0 && inBounds)");
    REQUIRE(guard != std::string::npos);
    for (const char* load : {"float4 g        = u_GeomPrev[tp]", "pColor += w * u_ColorHistPrev[tp]", "float3 m = u_MomentsPrev[tp]"}) {
        const auto position = source.find(load);
        REQUIRE(position != std::string::npos);
        CHECK(position > guard);
    }
    REQUIRE(SlangCompiler::Available());
    const auto code = SlangCompiler::CompileReflectStage(path);
    CHECK_FALSE(code.spirv.empty());
    CHECK(code.stage == ShaderStage::Compute);
}
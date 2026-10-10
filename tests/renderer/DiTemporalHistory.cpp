#include <doctest/doctest.h>
#include "luth/renderer/features/rt/DiTemporalHistory.h"
#include "luth/renderer/shader/SlangCompiler.h"
#include <filesystem>

using namespace Luth;

TEST_CASE("DiTemporalHistory: provisional work requires matching successful submission")
{
    DiTemporalHistory history;
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

TEST_CASE("DiTemporalHistory: absence bypass reload and failed graph discard history")
{
    DiTemporalHistory raw, diffuse, specular;
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

TEST_CASE("DiTemporalHistory: independent views preserve absolute frame continuity")
{
    FeatureViewStates<DiTemporalHistory> views;
    auto make = [](const ViewStateConfig&) { return DiTemporalHistory{}; };
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

TEST_CASE("DiTemporalHistory: production reset guards compile with unchanged native ABI")
{
    REQUIRE(SlangCompiler::Available());
    for (const char* file : {"restir_temporal.slang", "svgf_reproject.slang"}) {
        const auto code = SlangCompiler::CompileReflectStage(std::filesystem::absolute(
            std::filesystem::path("engine/assets/shaders") / file));
        CHECK_FALSE(code.spirv.empty());
        CHECK(code.stage == ShaderStage::Compute);
    }
}

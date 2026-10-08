#include <doctest/doctest.h>
#include "luth/renderer/debug/ProfilingQuerySlots.h"
using namespace Luth;

TEST_CASE("ProfilingQuerySlots: GPU-behind slots remain owned until both timelines complete")
{
    ProfilingQuerySlots slots;
    ProfileSubmissionProvenance record; record.view = {1}; record.resourceGeneration = 2;
    record.renderFrameIndex = 10; record.topologyGeneration = 3;
    record.passes = {{1, "Depth", ProfileSubmissionPhase::GraphicsA, false}};
    const SubmissionCompletionToken token{4, 12, 6, 11, 0, true};
    const auto first = slots.Acquire(record, true); REQUIRE(first);
    slots.Submit(*first, token);
    record.renderFrameIndex = 11; record.passes.clear();
    CHECK(slots.Get(*first).record.renderFrameIndex == 10);
    CHECK(slots.Get(*first).record.passes.size() == 1);
    CHECK(slots.Get(*first).stats);
    auto behind = [](const auto& t) { return SubmissionCompletionProgress{4, 12, 5}.Contains(t); };
    CHECK_FALSE(slots.CanRead(*first, behind));
    CHECK_FALSE(slots.Retire(*first, behind));
    for (u32 i = 1; i < ProfilingQuerySlots::Count; ++i)
    {
        const auto next = slots.Acquire(record, false); REQUIRE(next);
        slots.Submit(*next, token);
    }
    CHECK_FALSE(slots.Acquire(record, true));
    auto complete = [](const auto& t) { return SubmissionCompletionProgress{4, 12, 6}.Contains(t); };
    CHECK(slots.CanRead(*first, complete));
    CHECK(slots.Retire(*first, complete));
    const auto reused = slots.Acquire(record, false); REQUIRE(reused);
    CHECK(*reused == *first);
    CHECK_FALSE(slots.Get(*reused).stats);
    CHECK_FALSE(slots.Get(*reused).record.completion.valid);
}

TEST_CASE("ProfilingQuerySlots: failed submissions and stale timeline generations cannot recycle queries")
{
    ProfilingQueryEpoch epoch;
    CHECK_FALSE(epoch.Observe({}));
    CHECK_FALSE(epoch.Observe({9, 1, 0, 5, 1, true}));
    CHECK_FALSE(epoch.Observe({9, 2, 0, 6, 1, true}));
    CHECK_FALSE(epoch.Observe({10, 3, 0, 7, 1, false}));
    CHECK(epoch.Observe({10, 1, 0, 7, 1, true}));
    CHECK_FALSE(epoch.Observe({10, 2, 0, 8, 1, true}));
    ProfilingQuerySlots a, b;
    ProfileSubmissionProvenance record; record.view = {1};
    const auto first = a.Acquire(record, false); REQUIRE(first);
    CHECK_FALSE(a.CanRead(*first, [](const auto&) { return true; }));
    a.Submit(*first, {});
    CHECK(a.Get(*first).state == ProfilingQuerySlots::State::Quarantined);
    CHECK_FALSE(a.Retire(*first, [](const auto&) { return true; }));
    CHECK_THROWS_AS(a.Submit(*first, {}), std::logic_error);
    record.view = {2}; const auto other = b.Acquire(record, false); REQUIRE(other);
    b.Submit(*other, {9, 1, 0, 5, 1, true});
    CHECK_FALSE(b.CanRead(*other, [](const auto& t) { return SubmissionCompletionProgress{10, 100, 100}.Contains(t); }));
    CHECK(b.CanRead(*other, [](const auto& t) { return SubmissionCompletionProgress{9, 1, 0}.Contains(t); }));
    CHECK(a.Get(*first).record.view.value == 1);
    CHECK(b.Get(*other).record.view.value == 2);
}

TEST_CASE("ProfilingQuerySlots: frozen dense mappings reject mismatched views generations and topology")
{
    ProfileSubmissionProvenance current; current.view = {7}; current.resourceGeneration = 8;
    current.topologyGeneration = 9; current.renderFrameIndex = 20;
    current.passes = {{1, "Depth", ProfileSubmissionPhase::GraphicsA, false},
        {3, "Bloom", ProfileSubmissionPhase::GraphicsB, true}};
    ProfileQueryResults results; results.record = current; results.record.renderFrameIndex = 18;
    results.times = {1.5f, 0.5f}; results.stats.resize(2);
    results.stats[0].valid = true; results.stats[0].inputVertices = 42;
    RG::RenderGraphSnapshot snapshot; snapshot.passes.resize(4);
    snapshot.passes[0].culled = snapshot.passes[2].culled = true;
    snapshot.passes[1].name = "Depth"; snapshot.passes[3].name = "Bloom";
    CHECK(ApplyProfileResults(results, current, snapshot));
    CHECK(snapshot.profilingAvailable);
    CHECK(snapshot.profileRenderFrameIndex == 18);
    CHECK(snapshot.profileTopologyGeneration == 9);
    CHECK(snapshot.profileViewId == 7);
    CHECK(snapshot.passes[0].gpuTimeMs == -1);
    CHECK(snapshot.passes[1].gpuTimeMs == 1.5f);
    CHECK(snapshot.passes[3].gpuTimeMs == 0.5f);
    CHECK(snapshot.totalGpuTimeMs == 2.0f);
    CHECK(snapshot.totalStats.inputVertices == 42);
    auto mismatch = current; mismatch.view = {6}; CHECK_FALSE(ApplyProfileResults(results, mismatch, snapshot));
    mismatch = current; ++mismatch.resourceGeneration; CHECK_FALSE(ApplyProfileResults(results, mismatch, snapshot));
    mismatch = current; ++mismatch.topologyGeneration; CHECK_FALSE(ApplyProfileResults(results, mismatch, snapshot));
    mismatch = current; mismatch.passes.pop_back(); CHECK_FALSE(ApplyProfileResults(results, mismatch, snapshot));
    snapshot.passes[3].name = "Changed"; CHECK_FALSE(ApplyProfileResults(results, current, snapshot));
    CHECK(snapshot.passes[1].gpuTimeMs == 1.5f); // No partial overwrite on invalid mapping.
    snapshot.passes[3].name = "Bloom";
    results.times[1] = -1;
    CHECK(ApplyProfileResults(results, current, snapshot));
    CHECK_FALSE(snapshot.profilingAvailable);
    CHECK(snapshot.totalGpuTimeMs == -1);
    CHECK(snapshot.passes[1].gpuTimeMs == 1.5f);
    CHECK(snapshot.passes[3].gpuTimeMs == -1);
}

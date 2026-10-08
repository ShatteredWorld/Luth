#include <doctest/doctest.h>
#include "luth/renderer/debug/CaptureFinalization.h"
#include "luth/renderer/debug/CaptureRecordingSession.h"
#include "luth/renderer/rendergraph/RenderGraph.h"
#include "luth/memory/LinearAllocator.h"

using namespace Luth;
namespace
{
    struct Fixture
    {
        int target = 0;
        FrameDebugger debugger;
        RenderViewRegistry views;
        CaptureFinalizationInputs inputs;
        RG::RenderGraphSnapshot snapshot;
        Fixture()
        {
            debugger.state = DebuggerState::CaptureRequested;
            inputs.view.id = views.Register(&target);
            inputs.view.targets = reinterpret_cast<FrameTargets*>(&target);
            inputs.view.resourceGeneration = 1;
            inputs.view.width = 640; inputs.view.height = 480;
            inputs.source = CaptureSource::Game;
        }
        bool Finish() { return FinalizeViewCapture(debugger, inputs, snapshot, views); }
    };
}

TEST_CASE("CaptureFinalization: graph identity maps timings after metadata sorting")
{
    Fixture f;
    f.snapshot.passes.resize(5);
    for (u32 i = 0; i < 5; ++i) f.snapshot.passes[i].gpuTimeMs = float(i + 1);
    f.snapshot.passes[1].culled = true;
    f.snapshot.totalGpuTimeMs = 14;
    // Recording emits graphics metadata before compute; one live graph pass emits none.
    for (u32 graphIndex : {4u, 0u, 2u})
    {
        RG::CapturedPass pass; pass.graphPassIndex = graphIndex;
        pass.name = "Pass" + std::to_string(graphIndex);
        pass.firstDrawIndex = u32(f.debugger.capturedFrame.drawCalls.size());
        pass.drawCallCount = 1;
        RG::CapturedDrawCall draw; draw.passIndex = u32(f.debugger.capturedFrame.passes.size());
        draw.passName = pass.name;
        f.debugger.capturedFrame.drawCalls.push_back(draw);
        f.debugger.capturedFrame.passes.push_back(pass);
    }
    REQUIRE(f.Finish());
    const auto& frame = f.debugger.capturedFrame;
    REQUIRE(frame.passes.size() == 3);
    for (u32 i = 0; i < 3; ++i)
    {
        CHECK(frame.passes[i].graphPassIndex == i * 2);
        CHECK(frame.passes[i].gpuTimeMs == float(i * 2 + 1));
        CHECK(frame.passes[i].firstDrawIndex == i);
        CHECK(frame.drawCalls[i].passIndex == i);
        CHECK(frame.drawCalls[i].passName == frame.passes[i].name);
    }
    CHECK(frame.totalGpuTimeMs == 14);
    REQUIRE(frame.rootEvent.children.size() == 3);
    CHECK(frame.rootEvent.children[0].gpuTimeMs == 1);
    CHECK(frame.rootEvent.children[2].gpuTimeMs == 5);
    CHECK(frame.valid);
    CHECK(f.debugger.state == DebuggerState::Frozen);
    CHECK(f.debugger.capturedSource == CaptureSource::Game);
}

TEST_CASE("CaptureFinalization: copies frozen native inputs and rejects stale view generations")
{
    Fixture f;
    f.inputs.globalUboBytes = {1, 2, 3};
    f.inputs.selectionHandles = {entt::entity{42}};
    f.inputs.iblIntensity = 2; f.inputs.skyboxIntensity = 3;
    f.inputs.cascades.splitsViewZ = Vec4(9);
    f.inputs.shadowParams.shadowBias = Vec4(0.1f);
    f.inputs.replayBindings.sets[0] = reinterpret_cast<VkDescriptorSet>(uintptr_t(9));
    REQUIRE(f.Finish());
    f.inputs.globalUboBytes[0] = 99;
    f.inputs.selectionHandles.clear();
    const auto& frame = f.debugger.capturedFrame;
    CHECK(frame.capturedGlobalUboBytes[0] == 1);
    CHECK(frame.capturedSelectionHandles.size() == 1);
    CHECK(frame.capturedIblIntensity == 2);
    CHECK(frame.capturedSkyboxIntensity == 3);
    CHECK(frame.cascadeSplitsViewZ.x == 9);
    CHECK(frame.shadowBias.x == doctest::Approx(0.1f));
    CHECK(frame.replayBindings.sets[0] == f.inputs.replayBindings.sets[0]);
    f.debugger.state = DebuggerState::CaptureRequested;
    f.views.Invalidate(f.inputs.view.id);
    CHECK_FALSE(f.Finish());
    CHECK(frame.capturedGlobalUboBytes[0] == 1);
    CHECK(f.debugger.state == DebuggerState::CaptureRequested);
    f.views.Release(f.inputs.view.id);
    CHECK_FALSE(f.Finish());
}

TEST_CASE("CaptureFinalization: state gates and unmatched samples leave no positional timing")
{
    Fixture f;
    for (auto state : {DebuggerState::Inactive, DebuggerState::Frozen})
    {
        f.debugger.state = state;
        CHECK_FALSE(f.Finish());
        CHECK_FALSE(f.debugger.capturedFrame.valid);
    }
    f.debugger.state = DebuggerState::CaptureRequested;
    Memory::LinearAllocator scratch(65536); RG::RenderGraph graph(scratch);
    {
        CaptureRecordingSession recording(f.debugger, graph, f.inputs.view.id, 1, true);
        CHECK_FALSE(f.Finish());
    }
    f.inputs.view.width = 0;
    CHECK_FALSE(f.Finish());
    f.inputs.view.width = 640;
    f.snapshot.passes.resize(1); f.snapshot.passes[0].culled = true;
    for (u32 index : {0u, 9u})
    {
        RG::CapturedPass pass; pass.graphPassIndex = index; pass.gpuTimeMs = 77;
        f.debugger.capturedFrame.passes.push_back(pass);
    }
    REQUIRE(f.Finish());
    CHECK(f.debugger.capturedFrame.passes[0].gpuTimeMs == 0);
    CHECK(f.debugger.capturedFrame.passes[1].gpuTimeMs == 0);
}

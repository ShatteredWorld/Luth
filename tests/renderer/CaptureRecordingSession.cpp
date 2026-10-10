#include <doctest/doctest.h>
#include "luth/renderer/debug/CaptureRecordingSession.h"
#include "luth/renderer/rendergraph/RenderGraph.h"
#include "luth/memory/LinearAllocator.h"

using namespace Luth;

namespace
{
    void Record(FrameDebugger& debugger, u32 index)
    {
        debugger.BeginCapturePass(index, "Compute", "SceneColor", false, {});
        debugger.CaptureComputeDispatch("Compute", "test", 2, 3, 4);
        debugger.EndCapturePass();
    }
}

TEST_CASE("CaptureRecordingSession: selected view exclusively records metadata")
{
    FrameDebugger debugger;
    debugger.state = DebuggerState::CaptureRequested;
    debugger.requestedSource = CaptureSource::Game;
    Memory::LinearAllocator scratch(1024 * 1024);
    RG::RenderGraph other(scratch), selected(scratch);
    Record(debugger, 0);
    CHECK(debugger.capturedFrame.passes.empty());
    {
        CaptureRecordingSession session(debugger, other, {11}, 7, false);
        CHECK_FALSE(session.IsCapturing());
        CHECK_FALSE(debugger.IsRecordingCapture());
        CHECK(debugger.state == DebuggerState::CaptureRequested);
        Record(debugger, 1);
        CHECK(debugger.capturedFrame.drawCalls.empty());
    }
    {
        CaptureRecordingSession session(debugger, selected, {12}, 9, true);
        CHECK(session.IsCapturing());
        CHECK(session.ViewId() == RenderViewId{12});
        CHECK(session.ResourceGeneration() == 9);
        CHECK(session.Source() == CaptureSource::Game);
        debugger.requestedSource = CaptureSource::Scene;
        CHECK(session.Source() == CaptureSource::Game);
        Record(debugger, 3);
        REQUIRE(debugger.capturedFrame.passes.size() == 1);
        CHECK(debugger.capturedFrame.passes[0].graphPassIndex == 3);
        CHECK(debugger.capturedFrame.passes[0].drawCallCount == 1);
        CHECK(debugger.capturedFrame.drawCalls[0].groupCountY == 3);
        CHECK(debugger.trackedRTs.size() == 16);
        CHECK(debugger.trackedRTs.contains("ShadowMap.C3"));
        CHECK(debugger.trackedRTs.contains("SlimMaterialID"));
    }
    CHECK_FALSE(debugger.IsRecordingCapture());
    CHECK(debugger.state == DebuggerState::CaptureRequested);
    Record(debugger, 4);
    CHECK(debugger.capturedFrame.passes.size() == 1);
}

TEST_CASE("CaptureRecordingSession: exception cleanup and invalid identity")
{
    FrameDebugger debugger;
    debugger.state = DebuggerState::CaptureRequested;
    Memory::LinearAllocator scratch(1024 * 1024);
    RG::RenderGraph graph(scratch), second(scratch);
    CHECK_THROWS_AS(CaptureRecordingSession(debugger, graph, {}, 1, true), std::invalid_argument);
    CHECK_THROWS_AS(CaptureRecordingSession(debugger, graph, {1}, 0, true), std::invalid_argument);
    CHECK_FALSE(debugger.IsRecordingCapture());
    try
    {
        CaptureRecordingSession session(debugger, graph, {1}, 2, true);
        CHECK_THROWS_AS(CaptureRecordingSession(debugger, second, {2}, 3, false), std::logic_error);
        CHECK(debugger.IsRecordingCapture());
        throw std::runtime_error("recording failed");
    }
    catch (const std::runtime_error&) {}
    CHECK_FALSE(debugger.IsRecordingCapture());
    CHECK(debugger.state == DebuggerState::CaptureRequested);
    CaptureRecordingSession retry(debugger, second, {2}, 3, true);
    CHECK(retry.IsCapturing());
}

TEST_CASE("CaptureRecordingSession: inactive and frozen views cannot append metadata")
{
    FrameDebugger debugger;
    Memory::LinearAllocator scratch(1024 * 1024);
    RG::RenderGraph graph(scratch);
    for (auto state : {DebuggerState::Inactive, DebuggerState::Frozen})
    {
        debugger.state = state;
        CaptureRecordingSession session(debugger, graph, {}, 0, true);
        CHECK_FALSE(session.IsCapturing());
        Record(debugger, 0);
        debugger.CaptureDrawCall("pass", "mesh", "entity", 1, 3, {}, {});
        debugger.CaptureIndirectDraw("pass", "mesh", "entity", 1, 3, 0, 16, {});
        CHECK(debugger.capturedFrame.passes.empty());
        CHECK(debugger.capturedFrame.drawCalls.empty());
        CHECK(debugger.trackedRTs.empty());
        CHECK(debugger.state == state);
    }
}

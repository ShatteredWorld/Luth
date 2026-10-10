#include "luthpch.h"
#include "luth/renderer/debug/CaptureRecordingSession.h"
#include "luth/renderer/rendergraph/RenderGraph.h"
#include "luth/renderer/lighting/LightTypes.h"
#include <stdexcept>

namespace Luth
{
    CaptureRecordingSession::CaptureRecordingSession(FrameDebugger& debugger, RG::RenderGraph& graph,
        RenderViewId view, u64 generation, bool requested)
        : m_Debugger(debugger), m_Graph(graph), m_View(view), m_Generation(generation),
          m_Source(debugger.requestedSource),
          m_Capturing(requested && debugger.state == DebuggerState::CaptureRequested)
    {
        if (debugger.m_RecordingSession)
            throw std::logic_error("Capture recording sessions cannot overlap");
        if (m_Capturing && (!view.value || !generation))
            throw std::invalid_argument("Capture recording requires a registered view generation");

        if (m_Capturing)
        {
            // Names retain the existing archive contract. Allocation and preview setup
            // happen before recording; this session only configures the graph's sink.
            for (const auto* name : {"SceneColor", "SceneDepth", "LDROutput", "EntityID",
                "BloomAFinal", "GTAOLinearDepth", "GTAORawAO", "GTAOFinal",
                "SlimNormal", "SlimRoughness", "SlimMotion", "SlimMaterialID"})
                debugger.RegisterTrackedRT(name);
            for (u32 cascade = 0; cascade < k_ShadowCascadeCount; ++cascade)
                debugger.RegisterTrackedRT("ShadowMap.C" + std::to_string(cascade));
            graph.SetArchiveSink(&debugger);
            graph.SetSerialize(true);
        }
        debugger.m_RecordingSession = this;
    }

    CaptureRecordingSession::~CaptureRecordingSession()
    {
        if (m_Capturing)
        {
            m_Graph.SetArchiveSink(nullptr);
            m_Graph.SetSerialize(false);
        }
        m_Debugger.m_RecordingSession = nullptr;
    }

    bool FrameDebugger::IsRecordingCapture() const
    {
        return m_RecordingSession && m_RecordingSession->IsCapturing()
            && state == DebuggerState::CaptureRequested;
    }
}

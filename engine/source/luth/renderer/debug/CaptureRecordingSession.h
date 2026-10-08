#pragma once

#include "luth/renderer/FrameDebugger.h"
#include "luth/renderer/features/RenderViewState.h"

namespace Luth
{
    // Scoped to one synchronous RecordGraph invocation. Worker jobs finish before this
    // object is destroyed; only a capturing graph records metadata, and does so serially.
    class CaptureRecordingSession final
    {
    public:
        CaptureRecordingSession(FrameDebugger&, RG::RenderGraph&, RenderViewId,
            u64 resourceGeneration, bool captureRequested);
        ~CaptureRecordingSession();
        CaptureRecordingSession(const CaptureRecordingSession&) = delete;
        CaptureRecordingSession& operator=(const CaptureRecordingSession&) = delete;

        bool IsCapturing() const { return m_Capturing; }
        RenderViewId ViewId() const { return m_View; }
        u64 ResourceGeneration() const { return m_Generation; }
        CaptureSource Source() const { return m_Source; }
        const RG::RenderGraph& Graph() const { return m_Graph; }

    private:
        FrameDebugger& m_Debugger;
        RG::RenderGraph& m_Graph;
        RenderViewId m_View;
        u64 m_Generation;
        CaptureSource m_Source;
        bool m_Capturing;
    };
}

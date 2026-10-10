#include "luthpch.h"
#include "luth/renderer/debug/CaptureFinalization.h"

namespace Luth
{
    bool FinalizeViewCapture(FrameDebugger& debugger, const CaptureFinalizationInputs& inputs,
        const RG::RenderGraphSnapshot& snapshot, const RenderViewRegistry& views)
    {
        if (debugger.state != DebuggerState::CaptureRequested || debugger.IsRecordingCapture()
            || !inputs.view.width || !inputs.view.height
            || !views.Matches(inputs.view.id, inputs.view.targets, inputs.view.resourceGeneration))
            return false;

        auto& frame = debugger.capturedFrame;
        frame.resources = snapshot.resources;
        frame.totalGpuTimeMs = snapshot.totalGpuTimeMs;
        frame.capturedView = inputs.view;
        frame.replayBindings = inputs.replayBindings;
        frame.capturedGlobalUboBytes = inputs.globalUboBytes;
        frame.capturedIrradiance = inputs.irradiance;
        frame.capturedPrefiltered = inputs.prefiltered;
        frame.capturedBRDF = inputs.brdf;
        frame.capturedGTAOFinal = inputs.gtaoFinal;
        frame.capturedIblIntensity = inputs.iblIntensity;
        frame.capturedSkyboxIntensity = inputs.skyboxIntensity;
        frame.capturedSelectionHandles = inputs.selectionHandles;
        frame.cascadeSplitsViewZ = inputs.cascades.splitsViewZ;
        frame.cascadeTexelSize = inputs.cascades.texelSize;
        frame.shadowBias = inputs.shadowParams.shadowBias;
        frame.shadowNormalBias = inputs.shadowParams.shadowNormalBias;
        for (u32 i = 0; i < k_ShadowCascadeCount; ++i)
            frame.lightSpaceMatrix[i] = inputs.cascades.lightSpaceMatrix[i];

        // Attach timings by graph index before FinalizeCapture sorts metadata and
        // builds the event tree. Missing metadata cannot shift another pass's sample.
        for (auto& pass : frame.passes)
        {
            pass.gpuTimeMs = 0.0f;
            if (pass.graphPassIndex < snapshot.passes.size())
            {
                const auto& sample = snapshot.passes[pass.graphPassIndex];
                if (!sample.culled) pass.gpuTimeMs = sample.gpuTimeMs;
            }
        }
        debugger.FinalizeCapture(inputs.viewProj);
        frame.valid = true;
        debugger.capturedSource = inputs.source;
        debugger.state = DebuggerState::Frozen;
        return true;
    }
}

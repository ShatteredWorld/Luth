#pragma once

#include "luth/renderer/FrameDebugger.h"
#include "luth/renderer/lighting/LightTypes.h"

namespace Luth
{
    // Explicit native contribution. Values are copied into the frozen capture;
    // textures retain ownership, native descriptor/buffer bindings remain borrowed.
    struct CaptureFinalizationInputs
    {
        RG::CapturedViewState view;
        CaptureSource source = CaptureSource::Scene;
        Mat4 viewProj{1.0f};
        CascadeData cascades;
        DirectionalLightShadowParams shadowParams;
        CapturedReplayBindings replayBindings;
        std::vector<u8> globalUboBytes;
        std::shared_ptr<Texture> irradiance, prefiltered, brdf, gtaoFinal;
        float iblIntensity = 1.0f, skyboxIntensity = 1.0f;
        std::vector<entt::entity> selectionHandles;
    };

    // Called after recording jobs finish. Rejected state/identity leaves metadata
    // untouched; this helper never queries active views or native domains.
    bool FinalizeViewCapture(FrameDebugger&, const CaptureFinalizationInputs&,
        const RG::RenderGraphSnapshot&, const RenderViewRegistry&);
}

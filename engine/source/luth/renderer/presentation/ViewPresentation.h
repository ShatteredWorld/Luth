#pragma once
#include "luth/renderer/rendergraph/RenderGraph.h"
struct ImDrawData;

namespace Luth
{
    struct FrameDebugger;
    struct ViewPresentationInputs
    {
        RG::TextureDesc backbuffer;
        void* image = nullptr;
        void* imageView = nullptr;
        ImDrawData* drawData = nullptr; // Frozen after ImGui::Render, borrowed through synchronous recording.
    };
    // Forward the producer's handle. Never imports another logical node for the same image.
    void AddViewOutputExport(RG::RenderGraph&, RG::ResourceHandle finalLdr);
    void AddViewImGuiPass(RG::RenderGraph&, const ViewPresentationInputs&, FrameDebugger&,
        RG::ResourceHandle sceneLdr = {});
}

#include "luthpch.h"
#include "luth/renderer/presentation/ViewPresentation.h"
#include "luth/renderer/FrameDebugger.h"
#include <backends/imgui_impl_vulkan.h>
#include <stdexcept>

namespace Luth
{
    void AddViewImGuiPass(RG::RenderGraph& graph, const ViewPresentationInputs& inputs,
        FrameDebugger& debugger, RG::ResourceHandle sceneLdr)
    {
        if (!inputs.image || !inputs.imageView || !inputs.backbuffer.width || !inputs.backbuffer.height)
            throw std::invalid_argument("Presentation requires an acquired backbuffer");
        struct Data { RG::ResourceHandle backbuffer, sceneTexture; ImDrawData* draws; };
        // Import once; finalState=Present retains the existing RG-generated present barrier.
        const auto backbuffer = graph.ImportResource(inputs.backbuffer, inputs.image, inputs.imageView,
            RG::ResourceState::Undefined, RG::ResourceState::Present);
        graph.AddPass<Data>("ImGuiPass",
            [backbuffer, sceneLdr, draws = inputs.drawData](Data& data, RG::RenderPassBuilder& builder) {
                data.backbuffer = builder.Write(backbuffer);
                data.draws = draws;
                if (sceneLdr.IsValid()) data.sceneTexture = builder.Read(sceneLdr);
            }, [&debugger](Data& data, RG::RenderPassContext& context) {
                debugger.BeginCapturePass(context.passIndex, "ImGuiPass", "Backbuffer", false,
                    {"imgui", 0, VK_CULL_MODE_NONE, VK_POLYGON_MODE_FILL, false, false, false, true});
                ImGui_ImplVulkan_RenderDrawData(data.draws, context.commandBuffer);
                debugger.EndCapturePass();
            });
    }
}

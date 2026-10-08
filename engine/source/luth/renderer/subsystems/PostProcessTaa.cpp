#include "luthpch.h"
#include "luth/renderer/subsystems/PostProcessSubsystem.h"
#include "luth/renderer/backend/vulkan/VulkanTexture.h"
#include "luth/renderer/FrameDebugger.h"

namespace Luth
{
    TaaBindings PostProcessSubsystem::PrepareTaaBindings(const std::shared_ptr<TaaViewState>& state,
        u64 frame, u64 generation, const Mat4& skyReproj, float alpha, bool enabled)
    {
        TaaBindings out;
        out.enabled = enabled; out.state = state;
        if (state)
        {
            state->recorded = false;
            state->ApplyShaderGeneration(m_TaaShaderGeneration);
        }
        if (!enabled || !state || !m_TaaResolvePipeline || !m_TaaResolvePipeline->GetHandle())
        {
            if (state) state->history.Invalidate();
            return out;
        }
        if (!state->historyA || !state->historyB) throw std::invalid_argument("TAA: missing prepared histories");
        WriteTaaResolvePerFrame(*state, frame);
        const bool parity = (frame & 1u) != 0;
        const auto previous = std::static_pointer_cast<VKTexture>(parity ? state->historyA : state->historyB);
        const auto current = std::static_pointer_cast<VKTexture>(parity ? state->historyB : state->historyA);
        out.pipeline = m_TaaResolvePipeline->GetHandle(); out.layout = m_TaaResolvePipeline->GetLayout();
        out.set = state->resolveSets[frame % MAX_FRAMES_IN_FLIGHT];
        for (size_t i = 0; i < out.sources.size(); ++i) out.sources[i] = {state->sources[i].get()};
        out.previous = {previous.get()}; out.current = {current.get()};
        out.previousImage = previous->GetImage(); out.previousView = previous->GetImageView();
        out.currentImage = current->GetImage(); out.currentView = current->GetImageView();
        out.width = current->GetWidth(); out.height = current->GetHeight();
        out.constants.skyReproj = skyReproj;
        out.constants.temporalAlpha = state->history.CanReuse(frame, generation) ? alpha : -1.0f;
        return out;
    }
    void PostProcessSubsystem::InvalidateTaaView(RenderViewId id)
    {
        if (auto* state = m_TaaStates.Find(id)) (*state)->history.Invalidate();
    }
    RG::ResourceHandle PostProcessSubsystem::AddTaaResolvePass(RG::RenderGraph& graph,
        RG::ResourceHandle color, RG::ResourceHandle motion, RG::ResourceHandle depth,
        const TaaBindings& packet, FrameDebugger* debugger)
    {
        RG::TextureDesc desc;
        desc.name = "TaaHistory"; desc.width = packet.width; desc.height = packet.height;
        desc.format = RG::TextureFormat::RGBA16_Float;
        const auto previous = graph.ImportResource(desc, (void*)packet.previousImage,
            (void*)packet.previousView, RG::ResourceState::ShaderResource);
        desc.name = "TaaCurrent";
        auto output = graph.ImportResource(desc, (void*)packet.currentImage,
            (void*)packet.currentView, RG::ResourceState::Undefined);
        struct Data { RG::ResourceHandle output; };
        graph.AddPass<Data>("TaaResolve",
            [&](Data& data, RG::RenderPassBuilder& builder) {
                builder.Read(color); builder.Read(motion); builder.Read(depth); builder.Read(previous);
                data.output = builder.Write(output, VK_ATTACHMENT_LOAD_OP_DONT_CARE, VK_ATTACHMENT_STORE_OP_STORE);
                output = data.output;
            },
            [packet, debugger](Data&, RG::RenderPassContext& ctx) {
                if (debugger) debugger->BeginCapturePass(ctx.passIndex, "TaaResolve", "TaaCurrent", false,
                    {"taa_resolve", 0, VK_CULL_MODE_NONE, VK_POLYGON_MODE_FILL, false, false, false, false});
                const auto cmd = ctx.commandBuffer;
                vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, packet.pipeline);
                vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, packet.layout, 0, 1, &packet.set, 0, nullptr);
                vkCmdPushConstants(cmd, packet.layout, VK_SHADER_STAGE_FRAGMENT_BIT, 0,
                    sizeof(packet.constants), &packet.constants);
                VkViewport viewport{}; viewport.width = (float)packet.width; viewport.height = (float)packet.height; viewport.maxDepth = 1.0f;
                vkCmdSetViewport(cmd, 0, 1, &viewport);
                const VkRect2D scissor{{0, 0}, {packet.width, packet.height}};
                vkCmdSetScissor(cmd, 0, 1, &scissor);
                vkCmdDraw(cmd, 3, 1, 0, 0);
                ObjectPushConstants dummy{};
                if (debugger) debugger->CaptureDrawCall("TaaResolve", "FullscreenTriangle", "TaaResolve", 0, 0, dummy,
                    {"taa_resolve", 0, VK_CULL_MODE_NONE, VK_POLYGON_MODE_FILL, false, false, false, false});
                if (debugger) debugger->EndCapturePass();
                packet.state->recorded = true;
            });
        return output;
    }
}

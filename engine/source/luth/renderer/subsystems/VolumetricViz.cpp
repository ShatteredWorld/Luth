#include "luthpch.h"
#include "luth/renderer/subsystems/VolumetricSubsystem.h"
#include "luth/renderer/backend/vulkan/VulkanTexture.h"
#include "luth/renderer/backend/vulkan/VulkanContext.h"
#include "luth/renderer/FrameDebugger.h"
namespace Luth
{
    void VolumetricSubsystem::WriteVizPerFrame(FogViewState& state, u64 frameAbs)
    {
        LH_PROFILE_FUNCTION();
        if (m_VizDescLayout == VK_NULL_HANDLE) return;
        if (!state.volInScatterHistA || !state.volInScatterHistB) return;

        const u32 slot    = frameAbs % MAX_FRAMES_IN_FLIGHT;
        const bool parity = (frameAbs & 1u) != 0u;
        if (state.volVizDescSet[slot] == VK_NULL_HANDLE) return;

        // Same parity rule as composite: sample the resolved atlas this frame.
        auto vkScat = std::static_pointer_cast<VKTexture>(
            parity ? state.volInScatterHistA : state.volInScatterHistB);

        VkDescriptorImageInfo scatInfo{};
        scatInfo.imageView   = vkScat->GetImageView();
        scatInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        scatInfo.sampler     = m_Sampler;

        VkWriteDescriptorSet write{ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
        write.dstSet          = state.volVizDescSet[slot];
        write.dstBinding      = 2;
        write.descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        write.descriptorCount = 1;
        write.pImageInfo      = &scatInfo;
        vkUpdateDescriptorSets(VulkanContext::Get().GetDevice(), 1, &write, 0, nullptr);
    }

    FogVizBindings VolumetricSubsystem::PrepareVizBindings(std::shared_ptr<FogViewState> state, u64 frame,
        VkDescriptorSet global, u32 mode, float scale, float opacity, bool enabled)
    {
        FogVizBindings out; out.enabled = enabled;
        if (!enabled || !m_VizPipeline) return out;
        if (!state || !global || !state->volVizDescSet[frame % MAX_FRAMES_IN_FLIGHT] ||
            !state->depthSource || !state->volDensity || !state->volInScatterHistA || !state->volInScatterHistB)
            throw std::invalid_argument("FogViz: incomplete native sources");
        out.pipeline = m_VizPipeline->GetHandle(); out.layout = m_VizPipeline->GetLayout();
        out.sets = {global, state->volVizDescSet[frame % MAX_FRAMES_IN_FLIGHT]};
        out.state = std::move(state); out.renderFrameIndex = frame; out.parameters = {mode, scale, opacity};
        WriteVizPerFrame(*out.state, frame); return out;
    }
    RG::ResourceHandle VolumetricSubsystem::AddVizPass(RG::RenderGraph& graph, RG::ResourceHandle input,
        RG::ResourceHandle density, RG::ResourceHandle resolved, RG::ResourceHandle depth,
        const FogVizBindings& packet, FrameDebugger* debugger)
    {
        struct Data { RG::ResourceHandle output, depth, density, resolved; };
        RG::ResourceHandle output;
        graph.AddPass<Data>("VolumetricVizPass", [&](Data& data, RG::RenderPassBuilder& builder) {
            builder.SetDebugMetadata(RG::RenderPassMetadata::Graphics("volumetric_viz", false, false, true, VK_CULL_MODE_NONE, 1));
            VkClearValue clear{}; clear.color = {{0, 0, 0, 1}};
            output = data.output = builder.Write(input, VK_ATTACHMENT_LOAD_OP_LOAD, VK_ATTACHMENT_STORE_OP_STORE, clear);
            data.depth = builder.Read(depth); data.density = builder.Read(density); data.resolved = builder.Read(resolved);
        }, [packet, debugger](Data& data, RG::RenderPassContext& ctx) {
            if (debugger) debugger->BeginCapturePass(ctx.passIndex, "VolumetricVizPass", "LDROutput", false,
                {"volumetric_viz", 0, VK_CULL_MODE_NONE, VK_POLYGON_MODE_FILL, false, false, false, false});
            const auto cmd = ctx.commandBuffer;
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, packet.pipeline);
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, packet.layout, 0, 2, packet.sets.data(), 0, nullptr);
            vkCmdPushConstants(cmd, packet.layout, VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(packet.parameters), &packet.parameters);
            const auto* res = static_cast<const RG::RenderGraph::ResourceNode*>(ctx.GetResource(data.output));
            VkViewport vp{}; vp.width = (float)res->desc.width; vp.height = (float)res->desc.height; vp.maxDepth = 1;
            vkCmdSetViewport(cmd, 0, 1, &vp); VkRect2D sc{}; sc.extent = {res->desc.width, res->desc.height};
            vkCmdSetScissor(cmd, 0, 1, &sc); vkCmdDraw(cmd, 3, 1, 0, 0);
            if (debugger)
            {
                ObjectPushConstants dummy{};
                debugger->CaptureDrawCall("VolumetricVizPass", "FullscreenTriangle", "VolumetricViz", 0, 0, dummy,
                    {"volumetric_viz", 0, VK_CULL_MODE_NONE, VK_POLYGON_MODE_FILL, false, false, false, false});
                debugger->EndCapturePass();
            }
        });
        return output;
    }
}

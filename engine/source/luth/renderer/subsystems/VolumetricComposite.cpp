#include "luthpch.h"
#include "luth/renderer/subsystems/VolumetricSubsystem.h"
#include "luth/renderer/CameraParams.h"
#include "luth/renderer/FrameDebugger.h"

namespace Luth
{
    FogCompositeBindings VolumetricSubsystem::PrepareCompositeBindings(FogViewState& state, u32 frameAbs,
        const CameraParams& camera, VkDescriptorSet global, bool enabled)
    {
        FogCompositeBindings out;
        out.enabled = enabled;
        if (!enabled || !m_CompositePipeline) return out;
        out.pipeline = m_CompositePipeline->GetHandle();
        out.layout = m_CompositePipeline->GetLayout();
        out.sets = {global, state.volCompositeDescSet[frameAbs % MAX_FRAMES_IN_FLIGHT]};
        out.invView = Math::Inverse(camera.view);
        out.depth = {state.depthSource.get()};
        out.resolved = {(frameAbs & 1u) ? state.volInScatterHistA.get() : state.volInScatterHistB.get()};
        if (out.sets[0] && out.sets[1] && out.depth.texture && out.resolved.texture)
            WriteCompositePerFrame(state, frameAbs);
        return out;
    }
    RG::ResourceHandle VolumetricSubsystem::AddCompositePass(RG::RenderGraph& rg,
                                                              RG::ResourceHandle sceneColor,
                                                              RG::ResourceHandle sceneDepth,
                                                              RG::ResourceHandle resolvedInScatter, u32 width, u32 height,
        const FogCompositeBindings& packet, FrameDebugger* debugger)
    {
        LH_PROFILE_FUNCTION();
        if (!packet.pipeline) return sceneColor;

        struct CompositeData {
            RG::ResourceHandle color;
            RG::ResourceHandle depth;
            RG::ResourceHandle inScatter;
        };
        RG::ResourceHandle outputHandle;

        rg.AddPass<CompositeData>("VolumetricComposite",
            [&, sceneColor, sceneDepth, resolvedInScatter](CompositeData& data, RG::RenderPassBuilder& builder)
            {
                builder.SetDebugMetadata(RG::RenderPassMetadata::Graphics("volumetric_composite", false, false, true, VK_CULL_MODE_NONE));
                data.color = builder.Write(sceneColor,
                    VK_ATTACHMENT_LOAD_OP_LOAD, VK_ATTACHMENT_STORE_OP_STORE);
                data.depth = builder.Read(sceneDepth);
                // Sampler-binding 1 of the composite descriptor; declaring the read makes RG emit
                // the GENERAL -> SHADER_READ_ONLY transition after resolve's storage write.
                if (resolvedInScatter.IsValid())
                    data.inScatter = builder.Read(resolvedInScatter);
                outputHandle = data.color;
            },
            [packet, width, height, debugger](CompositeData& /*data*/, RG::RenderPassContext& ctx)
            {
                if (debugger) debugger->BeginCapturePass(ctx.passIndex, "VolumetricComposite",
                    "SceneColor", false,
                    { "volumetric_composite", 0, VK_CULL_MODE_NONE, VK_POLYGON_MODE_FILL, false, false, false, false });

                VkCommandBuffer cmd = ctx.commandBuffer;
                vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, packet.pipeline);

                vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                    packet.layout, 0, 2, packet.sets.data(), 0, nullptr);


                vkCmdPushConstants(cmd, packet.layout,
                    VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(Mat4), &packet.invView);

                const u32 w = width, h = height;
                VkViewport vp{}; vp.width = (float)w; vp.height = (float)h; vp.maxDepth = 1.0f;
                vkCmdSetViewport(cmd, 0, 1, &vp);
                VkRect2D sc{}; sc.extent = { w, h };
                vkCmdSetScissor(cmd, 0, 1, &sc);
                vkCmdDraw(cmd, 3, 1, 0, 0);

                ObjectPushConstants dummyPC{};
                if (debugger) debugger->CaptureDrawCall("VolumetricComposite", "FullscreenTriangle",
                    "VolumetricComposite", 0, 0, dummyPC,
                    { "volumetric_composite", 0, VK_CULL_MODE_NONE, VK_POLYGON_MODE_FILL, false, false, false, false });
                if (debugger) debugger->EndCapturePass();
            });
        return outputHandle;
    }

}

#include "luthpch.h"
#include "luth/renderer/subsystems/PostProcessSubsystem.h"
#include "luth/renderer/backend/vulkan/VulkanTexture.h"
#include "luth/renderer/backend/vulkan/VulkanContext.h"
#include "luth/renderer/FrameDebugger.h"
#include "luth/jobs/JobSystem.h"
#include "luth/memory/GPUTaggedPageAllocator.h"

namespace Luth
{
    PostProcessUBO MakeCompositeUniforms(const PostProcessSettings& s, bool dataDebug, bool hasBloom, float time)
    {
        PostProcessUBO ubo{};
        ubo.bloomThreshold = s.bloomThreshold; ubo.bloomStrength = hasBloom ? s.bloomStrength : 0.0f;
        ubo.exposure = s.exposure; ubo.contrast = s.contrast; ubo.saturation = s.saturation;
        ubo.tonemapOp = dataDebug ? -1 : static_cast<int>(s.tonemapOp);
        ubo.vignetteAmount = s.vignetteAmount; ubo.vignetteHardness = s.vignetteHardness;
        ubo.grainAmount = s.grainAmount; ubo.sharpness = s.sharpness; ubo.chromaticAberration = s.chromaticAberration;
        ubo.time = time; ubo.shadowBalance = s.shadowBalance; ubo.midtoneBalance = s.midtoneBalance;
        ubo.highlightBalance = s.highlightBalance;
        return ubo;
    }
    CompositeBindings PostProcessSubsystem::PrepareCompositeBindings(const std::shared_ptr<CompositeViewState>& state,
        TextureBindingRef source, TextureBindingRef bloom, const std::shared_ptr<Texture>& output,
        u64 frame, const PostProcessUBO& parameters)
    {
        CompositeBindings packet;
        packet.state = state; packet.source = source; packet.bloom = bloom; packet.parameters = parameters;
        packet.outputOwner = output;
        if (!state || !m_PostProcessPipeline || !m_PostProcessPipeline->GetHandle()) return packet;
        if (!source.texture || !output || !state->sources[1])
            throw std::invalid_argument("Composite: missing prepared images");
        auto* job = JobSystem::GetCurrentJobContext();
        if (!job) throw std::runtime_error("Composite: uniform upload requires a job context");
        packet.set = state->sets[frame % MAX_FRAMES_IN_FLIGHT];
        if (!packet.set) throw std::runtime_error("Composite: descriptor set unavailable");
        auto& heap = Memory::GPUTaggedPageAllocator::Get();
        job->GpuCache.CurrentTag = static_cast<u32>(frame);
        auto region = heap.Allocate(job->GpuCache, sizeof(PostProcessUBO), VulkanContext::Get().GetMinUniformBufferAlignment());
        if (!region.buffer) throw std::runtime_error("Composite: uniform allocation failed");
        memcpy(region.mappedPtr, &parameters, sizeof(parameters)); heap.FlushRegion(region);
        packet.uniform = {region.buffer, region.offset, region.size};
        packet.pipeline = m_PostProcessPipeline->GetHandle(); packet.layout = m_PostProcessPipeline->GetLayout();
        const auto nativeOutput = std::static_pointer_cast<VKTexture>(output);
        packet.output = {output.get()}; packet.outputImage = nativeOutput->GetImage(); packet.outputView = nativeOutput->GetImageView();
        packet.width = output->GetWidth(); packet.height = output->GetHeight();
        const auto* bloomSource = bloom.texture ? bloom.texture : state->sources[1].get();
        const std::array<VkDescriptorImageInfo, 2> images{{
            {m_Sampler, static_cast<const VKTexture*>(source.texture)->GetImageView(), VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL},
            {m_Sampler, static_cast<const VKTexture*>(bloomSource)->GetImageView(), VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL}}};
        std::array<VkWriteDescriptorSet, 3> writes{};
        for (u32 i = 0; i < writes.size(); ++i)
        {
            writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; writes[i].dstSet = packet.set;
            writes[i].dstBinding = i; writes[i].descriptorCount = 1;
            writes[i].descriptorType = i == 2 ? VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER : VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            if (i == 2) writes[i].pBufferInfo = &packet.uniform; else writes[i].pImageInfo = &images[i];
        }
        vkUpdateDescriptorSets(VulkanContext::Get().GetDevice(), (u32)writes.size(), writes.data(), 0, nullptr);
        return packet;
    }
    RG::ResourceHandle PostProcessSubsystem::AddCompositePass(RG::RenderGraph& graph, RG::ResourceHandle hdr,
        RG::ResourceHandle bloom, const CompositeBindings& packet, FrameDebugger* debugger)
    {
        RG::TextureDesc desc; desc.name = "LDROutput"; desc.width = packet.width; desc.height = packet.height;
        desc.format = RG::TextureFormat::RGBA8_Unorm;
        auto output = graph.ImportResource(desc, (void*)packet.outputImage, (void*)packet.outputView, RG::ResourceState::ShaderResource);
        struct Data {};
        graph.AddPass<Data>("PostProcess",
            [&](Data&, RG::RenderPassBuilder& builder) {
                output = builder.Write(output); builder.Read(hdr); if (bloom.IsValid()) builder.Read(bloom);
            },
            [packet, debugger](Data&, RG::RenderPassContext& ctx) {
                if (debugger) debugger->BeginCapturePass(ctx.passIndex, "PostProcess", "LDROutput", false,
                    {"postprocess", 0, VK_CULL_MODE_NONE, VK_POLYGON_MODE_FILL, false, false, false, false});
                const auto cmd = ctx.commandBuffer;
                vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, packet.pipeline);
                vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, packet.layout, 0, 1, &packet.set, 0, nullptr);
                VkViewport viewport{}; viewport.width = (float)packet.width; viewport.height = (float)packet.height; viewport.maxDepth = 1.0f;
                vkCmdSetViewport(cmd, 0, 1, &viewport);
                const VkRect2D scissor{{0, 0}, {packet.width, packet.height}}; vkCmdSetScissor(cmd, 0, 1, &scissor);
                vkCmdDraw(cmd, 3, 1, 0, 0);
                ObjectPushConstants dummy{};
                if (debugger) debugger->CaptureDrawCall("PostProcess", "FullscreenTriangle", "PostProcess", 0, 0, dummy,
                    {"postprocess", 0, VK_CULL_MODE_NONE, VK_POLYGON_MODE_FILL, false, false, false, false});
                if (debugger) debugger->EndCapturePass();
            });
        return output;
    }
}

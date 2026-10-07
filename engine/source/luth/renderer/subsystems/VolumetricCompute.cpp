#include "luthpch.h"
#include "luth/renderer/subsystems/VolumetricSubsystem.h"
#include "luth/renderer/subsystems/RtSubsystem.h"
#include "luth/renderer/CameraParams.h"
#include "luth/renderer/FrameDebugger.h"
#include "luth/renderer/material/MaterialSystem.h"
#include "luth/renderer/backend/vulkan/VulkanContext.h"
#include "luth/renderer/backend/vulkan/VulkanTexture.h"

namespace Luth
{
    FogComputeBindings VolumetricSubsystem::PrepareComputeBindings(FogViewState& state, u32 frameAbs,
        const CameraParams& camera, VkDescriptorSet global, bool enabled, bool rtShadows,
        const RtSubsystem* rayScene, const Memory::GPUSubRegion& volumes,
        const Memory::GPUSubRegion& lights, const Memory::GPUSubRegion& grid,
        const Memory::GPUSubRegion& indices)
    {
        FogComputeBindings out;
        out.enabled = enabled;
        if (!enabled) return out;
        const u32 slot = frameAbs % MAX_FRAMES_IN_FLIGHT;
        const bool parity = (frameAbs & 1u) != 0;
        const VKComputePipeline* pipelines[] = {m_InjectDensityPipeline.get(), m_InjectScatterPipeline.get(),
            m_IntegratePipeline.get(), m_ResolvePipeline.get()};
        for (u32 i = 0; i < 4; ++i)
            if (pipelines[i]) { out.pipelines[i] = pipelines[i]->GetHandle(); out.layouts[i] = pipelines[i]->GetLayout(); }
        out.sets = {state.volInjectDensityDescSet[slot], state.volInjectScatterDescSet[slot],
            state.volIntegrateDescSet[slot], state.volResolveDescSet[slot]};
        out.global = global; out.material = MaterialSystem::GetDescriptorSet(slot);
        out.bindless = VulkanContext::Get().GetBindlessSet().GetSet();
        auto binding = [](const std::shared_ptr<Texture>& image) {
            if (!image) return FogImageBinding{};
            const auto* native = static_cast<const VKTexture*>(image.get());
            return FogImageBinding{native->GetImage(), native->GetImageView(), {image.get()}};
        };
        out.density = binding(state.volDensity); out.scratch = binding(state.volInScatter);
        out.previous = binding(parity ? state.volInScatterHistB : state.volInScatterHistA);
        out.current = binding(parity ? state.volInScatterHistA : state.volInScatterHistB);
        out.volumes = volumes; out.lights = lights; out.grid = grid; out.indices = indices;
        out.inject.invView = Math::Inverse(camera.view);
        out.inject.volDimX = out.integrate.volDimX = out.resolve.volDimX = state.volDimX;
        out.inject.volDimY = out.integrate.volDimY = out.resolve.volDimY = state.volDimY;
        out.inject.volDimZ = out.integrate.volDimZ = out.resolve.volDimZ = state.volDimZ;
        out.integrate.nearFarPad = Vec4(camera.nearZ, camera.farZ, 0, 0);
        out.resolve.invView = out.inject.invView;
        out.currentHistoryA = parity; out.rayScene = rayScene; out.rtShadows = rtShadows;
        out.ready = global && out.material && out.bindless && volumes.buffer && lights.buffer && grid.buffer && indices.buffer &&
            std::all_of(out.pipelines.begin(), out.pipelines.end(), [](auto p) { return p != VK_NULL_HANDLE; }) &&
            std::all_of(out.sets.begin(), out.sets.end(), [](auto p) { return p != VK_NULL_HANDLE; });
        if (out.ready)
        {
            WriteInjectDensityPerFrame(state, frameAbs, volumes);
            WriteInjectScatterPerFrame(state, frameAbs, lights, grid, indices);
            WriteResolvePerFrame(state, frameAbs);
        }
        return out;
    }

    std::array<GraphTextureRef, 3> VolumetricSubsystem::AddComputePasses(RG::RenderGraph& graph,
        const FogComputeBindings& packet, RG::BufferHandle volumes, RG::BufferHandle lights,
        RG::BufferHandle grid, RG::BufferHandle indices, const ShadowCascadeRefs* shadows, FrameDebugger* debugger)
    {
        auto import = [&](const char* name, const FogImageBinding& image) {
            RG::TextureDesc desc;
            desc.name = name; desc.width = packet.inject.volDimX; desc.height = packet.inject.volDimY;
            desc.format = RG::TextureFormat::RGBA16_Float;
            return graph.ImportResource(desc, (void*)image.image, (void*)image.view, RG::ResourceState::Undefined);
        };
        auto density = import("VolDensity", packet.density);
        auto scatter = import("VolInScatter", packet.scratch);
        auto resolved = import(packet.currentHistoryA ? "VolInScatterHistA[curr]" : "VolInScatterHistB[curr]", packet.current);
        const char* names[] = {"VolumetricInjectDensity", "VolumetricInjectScatter", "VolumetricIntegrate", "VolumetricResolve"};
        const char* shaders[] = {"volumetric_inject_density", "volumetric_inject_scatter", "volumetric_integrate", "volumetric_resolve"};
        const char* targets[] = {"VolDensity", "VolInScatter", "VolInScatter", "VolInScatterHistA"};
        struct Data { RG::ResourceHandle output; };
        for (u32 pass = 0; pass < 4; ++pass)
        {
            graph.AddComputePass<Data>(names[pass], RG::QueueFamily::AsyncCompute,
                [&](Data& data, RG::RenderPassBuilder& builder) {
                    if (pass == 0)
                    {
                        builder.ReadBuffer(volumes);
                        density = data.output = builder.WriteStorageImage(density);
                    }
                    else if (pass == 1)
                    {
                        builder.ReadStorageImage(density);
                        builder.ReadBuffer(lights); builder.ReadBuffer(grid); builder.ReadBuffer(indices);
                        if (shadows) for (const auto& cascade : shadows->cascades) builder.ReadStorageImage(cascade.handle);
                        scatter = data.output = builder.WriteStorageImage(scatter);
                    }
                    else if (pass == 2)
                    {
                        builder.ReadStorageImage(density);
                        scatter = data.output = builder.WriteStorageImage(scatter);
                    }
                    else
                    {
                        builder.ReadStorageImage(scatter);
                        // Preserve legacy temporal descriptor/synchronization behavior. The previous
                        // atlas is a native history binding, not another import of the current image.
                        resolved = data.output = builder.WriteStorageImage(resolved);
                    }
                },
                [packet, debugger, pass, name = names[pass], shader = shaders[pass], target = targets[pass]]
                (Data&, RG::RenderPassContext& ctx) {
                    const auto cmd = ctx.commandBuffer;
                    if (debugger) debugger->BeginCapturePass(ctx.passIndex, name, target, false,
                        {shader, 0, 0, VK_POLYGON_MODE_FILL, false, false, false, false});
                    if (pass == 1 && packet.rtShadows)
                    {
                        VkMemoryBarrier2 barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
                        barrier.srcStageMask = VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR;
                        barrier.srcAccessMask = VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
                        barrier.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
                        barrier.dstAccessMask = VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR;
                        VkDependencyInfo dependency{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
                        dependency.memoryBarrierCount = 1; dependency.pMemoryBarriers = &barrier;
                        vkCmdPipelineBarrier2(cmd, &dependency);
                    }
                    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, packet.pipelines[pass]);
                    if (pass == 2)
                        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, packet.layouts[pass], 0, 1, &packet.sets[pass], 0, nullptr);
                    else
                    {
                        const VkDescriptorSet sets[] = {packet.global, packet.sets[pass]};
                        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, packet.layouts[pass], 0, 2, sets, 0, nullptr);
                    }
                    if (pass == 1)
                    {
                        const VkDescriptorSet sets[] = {packet.material, packet.bindless};
                        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, packet.layouts[pass], 3, 2, sets, 0, nullptr);
                    }
                    if (pass < 2)
                    {
                        auto constants = packet.inject;
                        if (pass == 1 && packet.rayScene) constants.geomTableBDA = packet.rayScene->GetGeometryTableBDA();
                        vkCmdPushConstants(cmd, packet.layouts[pass], VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(constants), &constants);
                    }
                    else if (pass == 2)
                        vkCmdPushConstants(cmd, packet.layouts[pass], VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(packet.integrate), &packet.integrate);
                    else
                        vkCmdPushConstants(cmd, packet.layouts[pass], VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(packet.resolve), &packet.resolve);
                    const u32 x = (packet.inject.volDimX + 7) / 8, y = (packet.inject.volDimY + 7) / 8;
                    const u32 z = pass == 2 ? 1 : (packet.inject.volDimZ + 3) / 4;
                    vkCmdDispatch(cmd, x, y, z);
                    if (debugger) { debugger->CaptureComputeDispatch(name, shader, x, y, z); debugger->EndCapturePass(); }
                });
        }
        return {{{density, packet.density.binding}, {scatter, packet.scratch.binding}, {resolved, packet.current.binding}}};
    }
}

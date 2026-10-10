#include "luthpch.h"
#include "luth/renderer/features/ClusteredLightingFeature.h"
#include "luth/renderer/subsystems/LightingSubsystem.h"
#include "luth/memory/LinearAllocator.h"

namespace Luth
{
    FeatureInfo ClusteredLightingFeature::Describe() const
    {
        FeatureInfo info;
        info.name = "ClusteredLighting";
        info.phase = FeaturePhase::Async;
        info.activation = FeatureActivation::Conditional;
        info.resources.reads = {{ClusterResources::Bindings},
            {ClusterResources::UploadedLights, ResourceReadRequirement::Optional}};
        info.resources.writes = {{RenderResources::LightData, ResourceOutputPresence::Optional,
                ResourceKeyRef{ClusterResources::UploadedLights}},
            {RenderResources::ClusterGrid, ResourceOutputPresence::Optional},
            {RenderResources::LightIndices, ResourceOutputPresence::Optional}};
        return info;
    }
    FeatureFrameDecision ClusteredLightingFeature::Evaluate(const FeaturePrepareContext& ctx) const
    {
        for (const auto bindings : {ctx.frame.resources, ctx.view.resources})
            for (const auto& binding : bindings)
                if (const auto* packet = binding.TryGet(ClusterResources::Bindings))
                {
                    if (!packet->native) throw std::invalid_argument("ClusteredLighting: missing native packet");
                    return {packet->native->ready};
                }
        throw std::invalid_argument("ClusteredLighting: missing frozen bindings");
    }
    void ClusteredLightingFeature::Build(RG::RenderGraph& graph, RenderFeatureContext& ctx)
    {
        const auto& packet = *ctx.resources.Get(ClusterResources::Bindings).native;
        const auto* lights = ctx.resources.TryGet(ClusterResources::UploadedLights);
        if (!lights || !packet.lights.buffer || !lights->handle.IsValid() || lights->handle.index > graph.GetBuffers().size() ||
            !lights->binding.slice || lights->binding.slice->buffer != packet.lights.buffer ||
            lights->binding.offset != packet.lights.offset || lights->binding.size != packet.lights.size ||
            lights->binding.offset != lights->binding.slice->offset || lights->binding.size > lights->binding.slice->size ||
            !packet.build || !packet.assign || !packet.buildLayout || !packet.assignLayout ||
            !packet.buildSet || !packet.assignSet || !ctx.view.width || !ctx.view.height ||
            packet.buildConstants.viewportSize != Vec2(float(ctx.view.width), float(ctx.view.height)) ||
            packet.lights.size < sizeof(LightSSBOHeader) + u64(packet.assignConstants.pointLightCount) * sizeof(PointLightData)
                + u64(packet.assignConstants.spotLightCount) * sizeof(SpotLightData))
            throw std::invalid_argument("ClusteredLighting: invalid prepared bindings, view or uploaded light slice");
        const auto valid = [](const Memory::GPUSubRegion& slice, u64 minimum) {
            return slice.buffer && slice.size >= minimum;
        };
        if (!valid(packet.aabb, u64(k_ClusterCount) * 32) ||
            !valid(packet.grid, u64(k_ClusterCount) * sizeof(GPUCluster)) ||
            !valid(packet.indices, u64(k_ClusterCount) * k_MaxLightsPerCluster * sizeof(u32)) ||
            !valid(packet.counter, 16))
            throw std::invalid_argument("ClusteredLighting: incomplete native output slices");
        const auto import = [&](const char* name, const Memory::GPUSubRegion& source) {
            // Keep slice identity even when several allocations share one VkBuffer.
            const auto* slice = ctx.scratch.New<Memory::GPUSubRegion>(source);
            return m_Native.ImportLightingBuffer(graph, name, *slice);
        };
        auto aabb = import("ClusterAABB", packet.aabb);
        auto grid = import("ClusterGrid", packet.grid);
        auto indices = import("LightIndex", packet.indices);
        const auto counter = import("LightCounter", packet.counter);
        const auto built = m_Native.AddClusterBuildPass(graph, aabb.handle, grid.handle, packet, m_Debugger);
        const auto assigned = m_Native.AddLightAssignPass(graph, lights->handle, built[0], built[1],
            indices.handle, counter.handle, packet, m_Debugger);
        grid.handle = assigned[0]; indices.handle = assigned[1];
        ctx.resources.Publish(RenderResources::LightData, *lights);
        ctx.resources.Publish(RenderResources::ClusterGrid, grid);
        ctx.resources.Publish(RenderResources::LightIndices, indices);
    }
}

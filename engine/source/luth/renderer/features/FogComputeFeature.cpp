#include "luthpch.h"
#include "luth/renderer/features/FogComputeFeature.h"
#include "luth/renderer/subsystems/VolumetricSubsystem.h"
#include "luth/renderer/lighting/FogVolumeGatherer.h"

namespace Luth
{
    FeatureInfo FogComputeFeature::Describe() const
    {
        FeatureInfo info;
        info.name = "FogCompute"; info.phase = FeaturePhase::Async;
        info.activation = FeatureActivation::Conditional;
        info.resources.reads = {{FogResources::Bindings},
            {FogResources::Volumes, ResourceReadRequirement::Optional},
            {RenderResources::LightData, ResourceReadRequirement::Optional},
            {RenderResources::ClusterGrid, ResourceReadRequirement::Optional},
            {RenderResources::LightIndices, ResourceReadRequirement::Optional},
            {RenderResources::ShadowCascades, ResourceReadRequirement::Optional}};
        info.resources.writes = {{RenderResources::FogDensity, ResourceOutputPresence::Optional},
            {FogResources::IntegratedScatter, ResourceOutputPresence::Optional},
            {RenderResources::ResolvedFog, ResourceOutputPresence::Optional}};
        return info;
    }
    FeatureFrameDecision FogComputeFeature::Evaluate(const FeaturePrepareContext& ctx) const
    {
        for (auto resources : {ctx.frame.resources, ctx.view.resources})
            for (const auto& resource : resources)
                if (const auto* ref = resource.TryGet(FogResources::Bindings))
                {
                    if (!ref->native) throw std::invalid_argument("FogCompute: missing native packet");
                    return {ref->native->enabled && ref->native->ready};
                }
        throw std::invalid_argument("FogCompute: missing frozen bindings");
    }
    void FogComputeFeature::Build(RG::RenderGraph& graph, RenderFeatureContext& ctx)
    {
        const auto& packet = *ctx.resources.Get(FogResources::Bindings).native;
        const auto* volumes = ctx.resources.TryGet(FogResources::Volumes);
        const auto* lights = ctx.resources.TryGet(RenderResources::LightData);
        const auto* grid = ctx.resources.TryGet(RenderResources::ClusterGrid);
        const auto* indices = ctx.resources.TryGet(RenderResources::LightIndices);
        auto valid = [&](const GraphBufferRef* value, const Memory::GPUSubRegion& expected, u64 minimum) {
            return value && value->handle.IsValid() && value->handle.index <= graph.GetBuffers().size() &&
                value->binding.slice && expected.buffer && expected.size >= minimum &&
                value->binding.slice->buffer == expected.buffer && value->binding.offset == expected.offset &&
                value->binding.size == expected.size && value->binding.slice->offset == expected.offset &&
                value->binding.slice->size >= expected.size;
        };
        if (!valid(volumes, packet.volumes, sizeof(FogVolumeSSBOHeader)) ||
            !valid(lights, packet.lights, sizeof(LightSSBOHeader)) ||
            !valid(grid, packet.grid, u64(k_ClusterCount) * sizeof(GPUCluster)) ||
            !valid(indices, packet.indices, u64(k_ClusterCount) * k_MaxLightsPerCluster * sizeof(u32)) ||
            !ctx.view.camera || !ctx.view.width || !ctx.view.height ||
            !packet.global || !packet.material || !packet.bindless ||
            std::any_of(packet.pipelines.begin(), packet.pipelines.end(), [](auto value) { return !value; }) ||
            std::any_of(packet.layouts.begin(), packet.layouts.end(), [](auto value) { return !value; }) ||
            std::any_of(packet.sets.begin(), packet.sets.end(), [](auto value) { return !value; }) ||
            !packet.inject.volDimX || !packet.inject.volDimY || !packet.inject.volDimZ ||
            (packet.rtShadows && !packet.rayScene))
            throw std::invalid_argument("FogCompute: incomplete prepared inputs or buffer slices");
        std::array<VkImage, 4> images{};
        u32 imageIndex = 0;
        for (const auto* image : {&packet.density, &packet.scratch, &packet.previous, &packet.current})
        {
            if (!image->image || !image->view || !image->binding.texture || image->binding.baseMip ||
                image->binding.mipCount != 1 || image->binding.baseLayer || image->binding.layerCount != 1)
                throw std::invalid_argument("FogCompute: incomplete native atlas bindings");
            images[imageIndex++] = image->image;
        }
        for (u32 i = 0; i < images.size(); ++i)
            for (u32 j = i + 1; j < images.size(); ++j)
                if (images[i] == images[j]) throw std::invalid_argument("FogCompute: aliased physical atlases");
        if (packet.integrate.volDimX != packet.inject.volDimX || packet.integrate.volDimY != packet.inject.volDimY ||
            packet.integrate.volDimZ != packet.inject.volDimZ || packet.resolve.volDimX != packet.inject.volDimX ||
            packet.resolve.volDimY != packet.inject.volDimY || packet.resolve.volDimZ != packet.inject.volDimZ)
            throw std::invalid_argument("FogCompute: inconsistent prepared atlas dimensions");
        const auto* shadows = ctx.resources.TryGet(RenderResources::ShadowCascades);
        if (shadows)
            for (const auto& cascade : shadows->cascades)
                if (!cascade.handle.IsValid() || cascade.handle.index > graph.GetResources().size() ||
                    !cascade.binding.texture || cascade.binding.layerCount != 1 || cascade.binding.mipCount != 1)
                    throw std::invalid_argument("FogCompute: invalid cascade producer");
        const auto output = m_Native.AddComputePasses(graph, packet, volumes->handle, lights->handle,
            grid->handle, indices->handle, shadows, m_Debugger);
        ctx.resources.Publish(RenderResources::FogDensity, output[0]);
        ctx.resources.Publish(FogResources::IntegratedScatter, output[1]);
        ctx.resources.Publish(RenderResources::ResolvedFog, output[2]);
    }
}

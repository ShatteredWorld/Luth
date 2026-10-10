#include "luthpch.h"
#include "luth/renderer/features/rt/RestirGiFeature.h"
#include "luth/renderer/subsystems/RtRestirGiSubsystem.h"
namespace Luth
{
    FeatureInfo RestirGiFeature::Describe() const
    {
        FeatureInfo info; info.name = "RestirGI"; info.phase = FeaturePhase::Async;
        info.activation = FeatureActivation::Conditional;
        info.resources.reads = {{RtSceneResources::Parameters}, {RestirGiResources::Bindings},
            {RenderResources::SurfaceDepth}, {RenderResources::Normal}, {RenderResources::MotionVectors},
            {RenderResources::LightData}, {RtSceneResources::Scene, ResourceReadRequirement::Optional}};
        info.resources.writes = {{RestirGiResources::Diffuse, ResourceOutputPresence::Optional},
            {GiReservoirVizResources::SpatialReservoir, ResourceOutputPresence::Optional}};
        info.capabilities.consumes = {{&RtSceneResources::RayScene}};
        info.capabilities.deviceRequirements = {&RtSceneResources::AccelerationStructures, &RtSceneResources::RayQueries};
        return info;
    }
    FeatureFrameDecision RestirGiFeature::Evaluate(const FeaturePrepareContext& ctx) const
    {
        if (!RestirGiRequested(ctx)) return {false};
        for (const auto bindings : {ctx.view.resources, ctx.frame.resources})
            for (const auto& binding : bindings)
                if (const auto* ref = binding.TryGet(RestirGiResources::Bindings))
                    return {ref->native && std::all_of(ref->native->pipelines.begin(), ref->native->pipelines.end(), [](auto p) { return p != VK_NULL_HANDLE; })};
        throw std::invalid_argument("RestirGI: missing frozen native bindings");
    }
    void RestirGiFeature::Build(RG::RenderGraph& graph, RenderFeatureContext& ctx)
    {
        const auto* native = ctx.resources.Get(RestirGiResources::Bindings).native;
        const auto* scene = ctx.resources.TryGet(RtSceneResources::Scene);
        const auto& lights = ctx.resources.Get(RenderResources::LightData);
        if (!native || !lights.handle.IsValid() || lights.handle.index > graph.GetBuffers().size() || !lights.binding.slice ||
            !native->lights.buffer || !native->lights.size || lights.binding.slice->buffer != native->lights.buffer ||
            lights.binding.offset != native->lights.offset || lights.binding.size != native->lights.size ||
            lights.binding.slice->offset != lights.binding.offset || lights.binding.slice->size < lights.binding.size ||
            graph.GetBuffers()[lights.handle.index - 1].buffer != native->lights.buffer)
            throw std::invalid_argument("RestirGI: light producer does not match prepared descriptor slice");
        if (!native || !scene || !scene->native || scene->native->frameIndex != ctx.frame.renderFrameIndex ||
            (scene->native->tlas.result.instanceCount && !scene->native->HasSceneData()) ||
            native->frameIndex != ctx.frame.renderFrameIndex || native->view != ctx.view.id ||
            native->generation != ctx.view.resourceGeneration || !native->settings.enabled ||
            native->fullWidth != ctx.view.width || native->fullHeight != ctx.view.height ||
            !native->fullWidth || !native->fullHeight || !native->tlas ||
            native->tlas != scene->native->GetTlas() || native->geometryTable != scene->native->GetGeometryTableBDA() ||
            std::any_of(native->layouts.begin(), native->layouts.end(), [](auto p) { return !p; }) ||
            std::any_of(native->sets.begin(), native->sets.end(), [](auto p) { return !p; }))
            throw std::invalid_argument("RestirGI: incomplete or stale scene/view bindings");
        const auto extent = RestirGiViewState::WorkingExtent(RestirGiViewState::Config(native->fullWidth,
            native->fullHeight, native->settings.halfResolution, 1));
        const u64 bytes = static_cast<u64>(extent[0]) * extent[1] * 64u;
        if (native->width != extent[0] || native->height != extent[1] || !native->scratch.buffer || !native->spatial.buffer ||
            native->scratch.buffer == native->spatial.buffer || native->scratch.offset || native->spatial.offset ||
            native->scratch.size != bytes || native->spatial.size != bytes ||
            !native->images[0] || !native->imageViews[0] || !native->outputs[0].texture)
            throw std::invalid_argument("RestirGI: incompatible output or dedicated reservoir allocation");
        const std::array inputs{ctx.resources.Get(RenderResources::SurfaceDepth), ctx.resources.Get(RenderResources::Normal),
            ctx.resources.Get(RenderResources::MotionVectors)};
        const std::array formats{RG::TextureFormat::D32_Float, RG::TextureFormat::RG16_Float,
            RG::TextureFormat::RG16_Float};
        for (u32 i = 0; i < inputs.size(); ++i) {
            const auto& input = inputs[i];
            if (!input.handle.IsValid() || input.handle.index > graph.GetResources().size() ||
                !native->sources[i] || input.binding.texture != native->sources[i] || input.binding.baseMip ||
                input.binding.mipCount != 1 || input.binding.baseLayer || input.binding.layerCount != 1)
                throw std::invalid_argument("RestirGI: sampled producer does not match prepared descriptors");
            const auto& node = graph.GetResources()[input.handle.index - 1];
            if (!native->sourceImages[i] || !native->sourceViews[i] || node.image != native->sourceImages[i] ||
                node.view != native->sourceViews[i] || node.desc.width != native->fullWidth || node.desc.height != native->fullHeight || node.desc.format != formats[i] ||
                node.image == native->images[0] ||
                input.binding.texture == native->outputs[0].texture)
                throw std::invalid_argument("RestirGI: incompatible input shape or output alias");
            for (u32 j = 0; j < i; ++j)
                if (input.handle.index == inputs[j].handle.index || node.image == graph.GetResources()[inputs[j].handle.index - 1].image)
                    throw std::invalid_argument("RestirGI: aliased sampled inputs");
        }
        for (const auto& output : native->outputs)
            if (output.baseMip || output.mipCount != 1 || output.baseLayer || output.layerCount != 1)
                throw std::invalid_argument("RestirGI: incompatible output subresource");
        const auto result = RtRestirGiSubsystem::AddPasses(graph, inputs[0].handle, inputs[1].handle,
            inputs[2].handle, *native, lights.handle);
        ctx.resources.Publish(RestirGiResources::Diffuse, GraphTextureRef{result.irradiance, native->outputs[0]});
        ctx.resources.Publish(GiReservoirVizResources::SpatialReservoir, GraphBufferRef{result.spatial,
            {native->retained ? &native->retained->restirGiSpatial : &native->spatial, native->spatial.offset, native->spatial.size}});
    }
}

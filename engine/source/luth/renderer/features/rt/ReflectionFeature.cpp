#include "luthpch.h"
#include "luth/renderer/features/rt/ReflectionFeature.h"
#include "luth/renderer/subsystems/ReflectionsSubsystem.h"
namespace Luth
{
    FeatureInfo ReflectionFeature::Describe() const
    {
        FeatureInfo info; info.name = "Reflection"; info.phase = FeaturePhase::Async;
        info.activation = FeatureActivation::Conditional;
        info.resources.reads = {{RtSceneResources::Parameters}, {ReflectionResources::Bindings},
            {RenderResources::SurfaceDepth}, {RenderResources::Normal}, {RenderResources::Roughness},
            {RenderResources::LightData}, {RtSceneResources::Scene, ResourceReadRequirement::Optional}};
        info.resources.writes = {{ReflectionResources::Radiance, ResourceOutputPresence::Optional}};
        info.capabilities.consumes = {{&RtSceneResources::RayScene}};
        info.capabilities.deviceRequirements = {&RtSceneResources::AccelerationStructures, &RtSceneResources::RayQueries};
        return info;
    }
    FeatureFrameDecision ReflectionFeature::Evaluate(const FeaturePrepareContext& ctx) const
    {
        if (!ReflectionRequested(ctx)) return {false};
        for (const auto bindings : {ctx.view.resources, ctx.frame.resources})
            for (const auto& binding : bindings)
                if (const auto* ref = binding.TryGet(ReflectionResources::Bindings))
                    return {ref->native && ref->native->pipeline != VK_NULL_HANDLE};
        throw std::invalid_argument("Reflection: missing frozen native bindings");
    }
    void ReflectionFeature::Build(RG::RenderGraph& graph, RenderFeatureContext& ctx)
    {
        const auto* native = ctx.resources.Get(ReflectionResources::Bindings).native;
        const auto* scene = ctx.resources.TryGet(RtSceneResources::Scene);
        const auto& lights = ctx.resources.Get(RenderResources::LightData);
        if (!native || !lights.handle.IsValid() || lights.handle.index > graph.GetBuffers().size() || !lights.binding.slice ||
            !native->lights.buffer || !native->lights.size || lights.binding.slice->buffer != native->lights.buffer ||
            lights.binding.offset != native->lights.offset || lights.binding.size != native->lights.size ||
            lights.binding.slice->offset != lights.binding.offset || lights.binding.slice->size < lights.binding.size ||
            graph.GetBuffers()[lights.handle.index - 1].buffer != native->lights.buffer)
            throw std::invalid_argument("Reflection: light producer does not match prepared descriptor slice");
        if (!native || !scene || !scene->native || scene->native->frameIndex != ctx.frame.renderFrameIndex ||
            (scene->native->tlas.result.instanceCount && !scene->native->HasSceneData()) ||
            native->frameIndex != ctx.frame.renderFrameIndex || native->view != ctx.view.id ||
            native->generation != ctx.view.resourceGeneration || !native->settings.enabled ||
            native->fullWidth != ctx.view.width || native->fullHeight != ctx.view.height ||
            !native->fullWidth || !native->fullHeight || !native->tlas ||
            native->tlas != scene->native->GetTlas() || native->geometryTable != scene->native->GetGeometryTableBDA() ||
            !native->layout ||
            std::any_of(native->sets.begin(), native->sets.end(), [](auto p) { return !p; }))
            throw std::invalid_argument("Reflection: incomplete or stale scene/view bindings");
        const auto extent = ReflectionViewState::WorkingExtent(ReflectionViewState::Config(native->fullWidth,
            native->fullHeight, native->settings.halfResolution, 1));
        if (native->width != extent[0] || native->height != extent[1] ||
            !native->image || !native->imageView || !native->output.texture)
            throw std::invalid_argument("Reflection: incompatible output extent or binding");
        const std::array parameters{native->settings.roughnessFadeEnd, native->settings.maxRayDistance,
            native->settings.fireflyClamp, native->settings.minLobeAlpha, native->settings.neeClamp};
        if (std::any_of(parameters.begin(), parameters.end(), [](auto value) { return !std::isfinite(value); }))
            throw std::invalid_argument("Reflection: non-finite trace parameter");
        for (const auto& node : graph.GetResources())
            if (node.image == native->image)
                throw std::invalid_argument("Reflection: output image already imported");
        const std::array inputs{ctx.resources.Get(RenderResources::SurfaceDepth), ctx.resources.Get(RenderResources::Normal),
            ctx.resources.Get(RenderResources::Roughness)};
        const std::array formats{RG::TextureFormat::D32_Float, RG::TextureFormat::RG16_Float,
            RG::TextureFormat::R8_Unorm};
        for (u32 i = 0; i < inputs.size(); ++i) {
            const auto& input = inputs[i];
            if (!input.handle.IsValid() || input.handle.index > graph.GetResources().size() ||
                !native->sources[i] || input.binding.texture != native->sources[i] || input.binding.baseMip ||
                input.binding.mipCount != 1 || input.binding.baseLayer || input.binding.layerCount != 1)
                throw std::invalid_argument("Reflection: sampled producer does not match prepared descriptors");
            const auto& node = graph.GetResources()[input.handle.index - 1];
            if (!native->sourceImages[i] || !native->sourceViews[i] || node.image != native->sourceImages[i] ||
                node.view != native->sourceViews[i] || node.desc.width != native->fullWidth || node.desc.height != native->fullHeight || node.desc.format != formats[i] ||
                node.image == native->image ||
                input.binding.texture == native->output.texture)
                throw std::invalid_argument("Reflection: incompatible input shape or output alias");
            for (u32 j = 0; j < i; ++j)
                if (input.handle.index == inputs[j].handle.index || node.image == graph.GetResources()[inputs[j].handle.index - 1].image)
                    throw std::invalid_argument("Reflection: aliased sampled inputs");
        }
        const auto& output = native->output;
        if (output.baseMip || output.mipCount != 1 || output.baseLayer || output.layerCount != 1)
            throw std::invalid_argument("Reflection: incompatible output subresource");
        const auto result = ReflectionsSubsystem::AddPasses(graph, inputs[0].handle, inputs[1].handle,
            inputs[2].handle, *native, lights.handle);
        ctx.resources.Publish(ReflectionResources::Radiance, GraphTextureRef{result, output});
    }
}

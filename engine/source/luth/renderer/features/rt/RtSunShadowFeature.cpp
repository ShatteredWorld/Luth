#include "luthpch.h"
#include "luth/renderer/features/rt/RtSunShadowFeature.h"
#include "luth/renderer/features/rt/RtSunShadowBindings.h"
#include "luth/renderer/rendergraph/RenderGraph.h"

namespace Luth
{
    FeatureInfo RtSunShadowFeature::Describe() const
    {
        FeatureInfo info;
        info.name = "RtSunShadows";
        info.phase = FeaturePhase::Async;
        info.activation = FeatureActivation::Conditional;
        info.resources.reads = {{RtSceneResources::Parameters}, {RtSunShadowResources::Bindings},
            {RenderResources::SurfaceDepth}, {RenderResources::Normal},
            {RtSceneResources::Scene, ResourceReadRequirement::Optional}};
        info.resources.writes = {{RtSunShadowResources::Mask, ResourceOutputPresence::Optional}};
        info.capabilities.consumes = {{&RtSceneResources::RayScene}};
        info.capabilities.deviceRequirements = {&RtSceneResources::AccelerationStructures, &RtSceneResources::RayQueries};
        return info;
    }
    FeatureFrameDecision RtSunShadowFeature::Evaluate(const FeaturePrepareContext& ctx) const
    {
        if (!RtSunShadowRequested(ctx)) return {false};
        for (const auto bindings : {ctx.view.resources, ctx.frame.resources})
            for (const auto& binding : bindings)
                if (const auto* ref = binding.TryGet(RtSunShadowResources::Bindings))
                    return {ref->native && ref->native->pipeline};
        throw std::invalid_argument("RtSunShadows: missing frozen native bindings");
    }
    void RtSunShadowFeature::Build(RG::RenderGraph& graph, RenderFeatureContext& ctx)
    {
        const auto* ref = ctx.resources.Get(RtSunShadowResources::Bindings).native;
        const auto* scene = ctx.resources.TryGet(RtSceneResources::Scene);
        const auto& depth = ctx.resources.Get(RenderResources::SurfaceDepth);
        const auto& normal = ctx.resources.Get(RenderResources::Normal);
        if (!ref || !scene || !scene->native || scene->native->frameIndex != ctx.frame.renderFrameIndex ||
            (scene->native->tlas.result.instanceCount && !scene->native->HasSceneData()) ||
            ref->frameIndex != ctx.frame.renderFrameIndex || ref->view != ctx.view.id ||
            ref->generation != ctx.view.resourceGeneration || !ref->width || !ref->height ||
            ref->width != ctx.view.width || ref->height != ctx.view.height ||
            !ref->pipeline || !ref->layout || !ref->image || !ref->imageView || !ref->mask.texture ||
            !ref->tlas || ref->tlas != scene->native->GetTlas() ||
            ref->geometryTable != scene->native->GetGeometryTableBDA() ||
            std::any_of(ref->sets.begin(), ref->sets.end(), [](auto set) { return !set; }))
            throw std::invalid_argument("RtSunShadows: incomplete or stale prepared scene/view bindings");
        for (const auto& [input, source] : {std::pair{depth, ref->depthSource}, std::pair{normal, ref->normalSource}})
            if (!input.handle.IsValid() || input.handle.index > graph.GetResources().size() || !source ||
                input.binding.texture != source || input.binding.baseMip || input.binding.mipCount != 1 ||
                input.binding.baseLayer || input.binding.layerCount != 1)
                throw std::invalid_argument("RtSunShadows: sampled producer does not match prepared descriptors");
        if (depth.handle.index == normal.handle.index || ref->mask.texture == ref->depthSource ||
            ref->mask.texture == ref->normalSource || ref->mask.baseMip || ref->mask.mipCount != 1 ||
            ref->mask.baseLayer || ref->mask.layerCount != 1)
            throw std::invalid_argument("RtSunShadows: incompatible physical image aliases");
        const auto& depthNode = graph.GetResources()[depth.handle.index - 1];
        const auto& normalNode = graph.GetResources()[normal.handle.index - 1];
        if (ref->image == depthNode.image || ref->image == normalNode.image || depthNode.image == normalNode.image ||
            depthNode.desc.width != ref->width || depthNode.desc.height != ref->height ||
            normalNode.desc.width != ref->width || normalNode.desc.height != ref->height ||
            depthNode.desc.format != RG::TextureFormat::D32_Float || normalNode.desc.format != RG::TextureFormat::RG16_Float)
            throw std::invalid_argument("RtSunShadows: incompatible sampled image shape or physical alias");
        ctx.resources.Publish(RtSunShadowResources::Mask,
            RtSubsystem::AddRtSunShadowsPass(graph, depth.handle, normal.handle, *ref));
    }
}

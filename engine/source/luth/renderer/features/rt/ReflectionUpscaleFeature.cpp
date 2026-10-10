#include "luthpch.h"
#include "luth/renderer/features/rt/ReflectionUpscaleFeature.h"
#include "luth/renderer/subsystems/ReflectionsSubsystem.h"
#include <cmath>

namespace Luth
{
    FeatureInfo ReflectionUpscaleFeature::Describe() const
    {
        const auto input = ReflectionDenoiserResources::Radiance;
        FeatureInfo info; info.name = "ReflectionUpscale";
        info.phase = FeaturePhase::Async;
        info.resources.reads = {{ReflectionUpscaleResources::Bindings},
            {input, ResourceReadRequirement::Optional}, {RenderResources::SurfaceDepth}, {RenderResources::Normal}};
        ResourceWrite output{ReflectionUpscaleResources::Radiance, ResourceOutputPresence::Optional};
        output.disabledPassthrough = input;
        info.resources.writes = {output};
        return info;
    }
    void ReflectionUpscaleFeature::Build(RG::RenderGraph& graph, RenderFeatureContext& ctx)
    {
        const auto outputKey = ReflectionUpscaleResources::Radiance;
        const auto* input = ctx.resources.TryGet(ReflectionDenoiserResources::Radiance);
        const auto* native = ctx.resources.Get(ReflectionUpscaleResources::Bindings).native;
        if (!input || !native) { ctx.resources.PublishAbsent(outputKey); return; }
        const auto single = [](const TextureBindingRef& b) {
            return b.texture && !b.baseMip && b.mipCount == 1 && !b.baseLayer && b.layerCount == 1;
        };
        if (native->view != ctx.view.id ||
            native->generation != ctx.view.resourceGeneration || native->frameIndex != ctx.frame.renderFrameIndex ||
            native->fullWidth != ctx.view.width || native->fullHeight != ctx.view.height ||
            !native->width || !native->height || !native->fullWidth || !native->fullHeight ||
            !single(native->output) || !native->outputImage || !native->outputView ||
            !input->handle.IsValid() || input->handle.index > graph.GetResources().size() || !single(input->binding))
            throw std::invalid_argument("Reflection upscale: stale view/frame or incomplete input/output");
        const bool full = native->width == native->fullWidth && native->height == native->fullHeight;
        if (!full && (native->width != std::max(native->fullWidth / 2, 1u) ||
            native->height != std::max(native->fullHeight / 2, 1u)))
            throw std::invalid_argument("Reflection upscale: incompatible working extent");
        const auto& signal = graph.GetResources()[input->handle.index - 1];
        if (signal.desc.format != RG::TextureFormat::RGBA16_Float ||
            signal.desc.width != native->width || signal.desc.height != native->height)
            throw std::invalid_argument("Reflection upscale: incompatible signal shape");
        if (full) {
            if (input->binding.texture != native->output.texture || signal.image != native->outputImage || signal.view != native->outputView)
                throw std::invalid_argument("Reflection upscale: full-resolution producer disagrees with lighting output");
            ctx.resources.Publish(outputKey, *input); return;
        }
        const std::array inputs{*input, ctx.resources.Get(RenderResources::SurfaceDepth), ctx.resources.Get(RenderResources::Normal)};
        const std::array formats{RG::TextureFormat::RGBA16_Float, RG::TextureFormat::D32_Float, RG::TextureFormat::RG16_Float};
        std::array<RG::ResourceHandle, 3> handles{};
        for (u32 i = 0; i < inputs.size(); ++i) {
            const auto& source = inputs[i];
            if (!source.handle.IsValid() || source.handle.index > graph.GetResources().size() || !single(source.binding) ||
                !single(native->sources[i]) || source.binding.texture != native->sources[i].texture)
                throw std::invalid_argument("Reflection upscale: producer disagrees with prepared descriptor");
            const auto& node = graph.GetResources()[source.handle.index - 1];
            if (!native->sourceImages[i] || !native->sourceViews[i] || node.image != native->sourceImages[i] ||
                node.view != native->sourceViews[i] || node.desc.format != formats[i] ||
                node.desc.width != (i ? native->fullWidth : native->width) ||
                node.desc.height != (i ? native->fullHeight : native->height) ||
                node.image == native->outputImage || source.binding.texture == native->output.texture)
                throw std::invalid_argument("Reflection upscale: incompatible source shape or output alias");
            for (u32 j = 0; j < i; ++j)
                if (node.image == native->sourceImages[j] || source.binding.texture == native->sources[j].texture)
                    throw std::invalid_argument("Reflection upscale: aliased sampled inputs");
            handles[i] = source.handle;
        }
        for (const auto& node : graph.GetResources())
            if (node.image == native->outputImage) throw std::invalid_argument("Reflection upscale: output already imported");
        if (!std::isfinite(native->phiDepth) || !std::isfinite(native->phiNormal))
            throw std::invalid_argument("Reflection upscale: nonfinite filtering parameters");
        // A missing native pipeline cannot satisfy a full-resolution half-input output contract.
        if (!native->Ready()) { ctx.resources.PublishAbsent(outputKey); return; }
        const auto result = ReflectionsSubsystem::AddUpscalePass(graph, handles, *native);
        ctx.resources.Publish(outputKey, GraphTextureRef{result, native->output});
    }
}

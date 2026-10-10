#include "luthpch.h"
#include "luth/renderer/features/rt/ReflectionDenoiserFeature.h"
#include "luth/renderer/subsystems/SvgfDenoiser.h"
#include <cmath>

namespace Luth
{
    FeatureInfo ReflectionDenoiserFeature::Describe() const
    {
        FeatureInfo info; info.name = "DenoiseReflection"; info.phase = FeaturePhase::Async;
        info.resources.reads = {{ReflectionDenoiserResources::Bindings},
            {ReflectionResources::Radiance, ResourceReadRequirement::Optional},
            {RenderResources::SurfaceDepth}, {RenderResources::Normal},
            {RenderResources::MaterialID}, {RenderResources::Roughness}};
        info.resources.writes = {{ReflectionDenoiserResources::Radiance, ResourceOutputPresence::Optional}};
        return info;
    }
    void ReflectionDenoiserFeature::Build(RG::RenderGraph& graph, RenderFeatureContext& ctx)
    {
        const auto outputKey = ReflectionDenoiserResources::Radiance;
        const auto* noisy = ctx.resources.TryGet(ReflectionResources::Radiance);
        const auto* native = ctx.resources.Get(ReflectionDenoiserResources::Bindings).native;
        if (!noisy || !native || !native->Ready()) {
            ctx.resources.PublishAbsent(outputKey);
            return;
        }
        const auto single = [](const TextureBindingRef& b) {
            return b.texture && !b.baseMip && b.mipCount == 1 && !b.baseLayer && b.layerCount == 1;
        };
        if (native->view != ctx.view.id || native->generation != ctx.view.resourceGeneration ||
            native->frameIndex != ctx.frame.renderFrameIndex || native->fullWidth != ctx.view.width ||
            native->fullHeight != ctx.view.height || !native->width || !native->height ||
            !native->fullWidth || !native->fullHeight || !single(native->output) || !native->outputImage || !native->outputView)
            throw std::invalid_argument("DenoiseReflection: stale view/frame or incomplete output");
        const bool full = native->width == native->fullWidth && native->height == native->fullHeight;
        if (!full && (native->width != std::max(native->fullWidth / 2, 1u) ||
            native->height != std::max(native->fullHeight / 2, 1u)))
            throw std::invalid_argument("DenoiseReflection: incompatible working extent");
        const std::array inputs{*noisy, ctx.resources.Get(RenderResources::SurfaceDepth), ctx.resources.Get(RenderResources::Normal),
            ctx.resources.Get(RenderResources::Roughness), ctx.resources.Get(RenderResources::MaterialID)};
        const std::array formats{RG::TextureFormat::RGBA16_Float, RG::TextureFormat::D32_Float,
            RG::TextureFormat::RG16_Float, RG::TextureFormat::R8_Unorm, RG::TextureFormat::R16_Uint};
        std::array<RG::ResourceHandle, 5> handles{};
        for (u32 i = 0; i < inputs.size(); ++i) {
            const auto& input = inputs[i];
            if (!input.handle.IsValid() || input.handle.index > graph.GetResources().size() || !single(input.binding) ||
                !single(native->sources[i]) || input.binding.texture != native->sources[i].texture)
                throw std::invalid_argument("DenoiseReflection: producer disagrees with prepared descriptor");
            const auto& node = graph.GetResources()[input.handle.index - 1];
            if (!native->sourceImages[i] || !native->sourceViews[i] || node.image != native->sourceImages[i] ||
                node.view != native->sourceViews[i] || node.desc.format != formats[i] ||
                node.desc.width != (i ? native->fullWidth : native->width) ||
                node.desc.height != (i ? native->fullHeight : native->height) || node.image == native->outputImage ||
                input.binding.texture == native->output.texture)
                throw std::invalid_argument("DenoiseReflection: incompatible input shape or output alias");
            for (u32 j = 0; j < i; ++j)
                if (node.image == native->sourceImages[j] || input.binding.texture == native->sources[j].texture)
                    throw std::invalid_argument("DenoiseReflection: aliased sampled inputs");
            handles[i] = input.handle;
        }
        if (native->ChainReady()) {
            if (!native->globalSet || !native->layouts[1] || !native->layouts[2] || !native->layouts[3] ||
                native->settings.atrousIterations > 31)
                throw std::invalid_argument("DenoiseReflection: incomplete native chain or unsupported iteration count");
            const auto& s = native->settings;
            for (const auto parameter : {s.alphaColor, s.alphaMoments, s.depthThreshold, s.normalThreshold,
                s.antiFireflySigma, s.phiRough, s.phiColor, s.phiNormal, s.phiDepth})
                if (!std::isfinite(parameter)) throw std::invalid_argument("DenoiseReflection: nonfinite settings");
            for (u32 i = 0; i < native->working.size(); ++i) {
                if (!single(native->working[i]) || !native->workingImages[i] || !native->workingViews[i] ||
                    native->workingImages[i] == native->outputImage || native->working[i].texture == native->output.texture)
                    throw std::invalid_argument("DenoiseReflection: incomplete or aliased working image");
                for (u32 j = 0; j < inputs.size(); ++j)
                    if (native->workingImages[i] == native->sourceImages[j] || native->working[i].texture == native->sources[j].texture)
                        throw std::invalid_argument("DenoiseReflection: working image aliases sampled input");
                for (u32 j = 0; j < i; ++j)
                    if (native->workingImages[i] == native->workingImages[j] || native->working[i].texture == native->working[j].texture)
                        throw std::invalid_argument("DenoiseReflection: working images alias");
            }
        } else if (!native->layouts[0]) throw std::invalid_argument("DenoiseReflection: incomplete native copy");
        for (const auto& node : graph.GetResources()) {
            if (node.image == native->outputImage) throw std::invalid_argument("DenoiseReflection: output already imported");
            if (native->ChainReady() && std::find(native->workingImages.begin(), native->workingImages.end(), node.image) != native->workingImages.end())
                throw std::invalid_argument("DenoiseReflection: working image already imported");
        }
        const auto output = SvgfDenoiser::AddReflectionPasses(graph, handles, *native);
        ctx.resources.Publish(outputKey, GraphTextureRef{output, native->output});
    }
}

#include "luthpch.h"
#include "luth/renderer/features/rt/DiDenoiserFeature.h"
#include "luth/renderer/subsystems/SvgfDenoiser.h"
#include <cmath>

namespace Luth
{
    FeatureInfo DiDenoiserFeature::Describe() const
    {
        FeatureInfo info; info.name = "DenoiseDI"; info.phase = FeaturePhase::Async;
        info.resources.reads = {{DiDenoiserResources::Bindings},
            {RestirDiResources::Diffuse, ResourceReadRequirement::Optional},
            {RenderResources::SurfaceDepth}, {RenderResources::Normal}, {RenderResources::MotionVectors},
            {RenderResources::MaterialID}, {RenderResources::Roughness}};
        info.resources.writes = {{DiDenoiserResources::Diffuse, ResourceOutputPresence::Optional}};
        return info;
    }
    void DiDenoiserFeature::Build(RG::RenderGraph& graph, RenderFeatureContext& ctx)
    {
        const auto* noisy = ctx.resources.TryGet(RestirDiResources::Diffuse);
        const auto* native = ctx.resources.Get(DiDenoiserResources::Bindings).native;
        if (!noisy || !native || !native->Ready()) {
            ctx.resources.PublishAbsent(DiDenoiserResources::Diffuse);
            return;
        }
        const auto single = [](const TextureBindingRef& b) {
            return b.texture && !b.baseMip && b.mipCount == 1 && !b.baseLayer && b.layerCount == 1;
        };
        if (native->view != ctx.view.id || native->generation != ctx.view.resourceGeneration ||
            native->frameIndex != ctx.frame.renderFrameIndex || native->fullWidth != ctx.view.width ||
            native->fullHeight != ctx.view.height || !native->width || !native->height ||
            !native->fullWidth || !native->fullHeight || !single(native->output) || !native->outputImage || !native->outputView)
            throw std::invalid_argument("DenoiseDI: stale view/frame or incomplete output");
        const bool full = native->width == native->fullWidth && native->height == native->fullHeight;
        if (!full && (native->width != std::max(native->fullWidth / 2, 1u) ||
            native->height != std::max(native->fullHeight / 2, 1u)))
            throw std::invalid_argument("DenoiseDI: incompatible working extent");
        const std::array inputs{*noisy, ctx.resources.Get(RenderResources::SurfaceDepth), ctx.resources.Get(RenderResources::Normal),
            ctx.resources.Get(RenderResources::MotionVectors), ctx.resources.Get(RenderResources::MaterialID), ctx.resources.Get(RenderResources::Roughness)};
        const std::array formats{RG::TextureFormat::RGBA16_Float, RG::TextureFormat::D32_Float,
            RG::TextureFormat::RG16_Float, RG::TextureFormat::RG16_Float, RG::TextureFormat::R16_Uint, RG::TextureFormat::R8_Unorm};
        std::array<RG::ResourceHandle, 6> handles{};
        for (u32 i = 0; i < inputs.size(); ++i) {
            const auto& input = inputs[i];
            if (!input.handle.IsValid() || input.handle.index > graph.GetResources().size() || !single(input.binding) ||
                !single(native->sources[i]) || input.binding.texture != native->sources[i].texture)
                throw std::invalid_argument("DenoiseDI: producer disagrees with prepared descriptor");
            const auto& node = graph.GetResources()[input.handle.index - 1];
            if (!native->sourceImages[i] || !native->sourceViews[i] || node.image != native->sourceImages[i] ||
                node.view != native->sourceViews[i] || node.desc.format != formats[i] ||
                node.desc.width != (i ? native->fullWidth : native->width) ||
                node.desc.height != (i ? native->fullHeight : native->height) || node.image == native->outputImage ||
                input.binding.texture == native->output.texture)
                throw std::invalid_argument("DenoiseDI: incompatible input shape or output alias");
            for (u32 j = 0; j < i; ++j)
                if (node.image == native->sourceImages[j] || input.binding.texture == native->sources[j].texture)
                    throw std::invalid_argument("DenoiseDI: aliased sampled inputs");
            handles[i] = input.handle;
        }
        if (native->ChainReady()) {
            if (!native->globalSet || !native->layouts[1] || !native->layouts[2] || !native->layouts[3] ||
                native->settings.atrousIterations > 31)
                throw std::invalid_argument("DenoiseDI: incomplete native chain or unsupported iteration count");
            const auto& s = native->settings;
            for (const auto parameter : {s.alphaColor, s.alphaMoments, s.depthThreshold, s.normalThreshold,
                s.antiFireflySigma, s.confidenceScale, s.phiColor, s.phiNormal, s.phiDepth})
                if (!std::isfinite(parameter)) throw std::invalid_argument("DenoiseDI: nonfinite settings");
            for (u32 i = 0; i < native->working.size(); ++i) {
                if (!single(native->working[i]) || !native->workingImages[i] || !native->workingViews[i] ||
                    native->workingImages[i] == native->outputImage || native->working[i].texture == native->output.texture)
                    throw std::invalid_argument("DenoiseDI: incomplete or aliased working image");
                for (u32 j = 0; j < inputs.size(); ++j)
                    if (native->workingImages[i] == native->sourceImages[j] || native->working[i].texture == native->sources[j].texture)
                        throw std::invalid_argument("DenoiseDI: working image aliases sampled input");
                for (u32 j = 0; j < i; ++j)
                    if (native->workingImages[i] == native->workingImages[j] || native->working[i].texture == native->working[j].texture)
                        throw std::invalid_argument("DenoiseDI: working images alias");
            }
        } else if (!native->layouts[0]) throw std::invalid_argument("DenoiseDI: incomplete native copy");
        for (const auto& node : graph.GetResources()) {
            if (node.image == native->outputImage) throw std::invalid_argument("DenoiseDI: output already imported");
            if (native->ChainReady() && std::find(native->workingImages.begin(), native->workingImages.end(), node.image) != native->workingImages.end())
                throw std::invalid_argument("DenoiseDI: working image already imported");
        }
        const auto output = SvgfDenoiser::AddDiPasses(graph, handles, *native);
        ctx.resources.Publish(DiDenoiserResources::Diffuse, GraphTextureRef{output, native->output});
    }
}

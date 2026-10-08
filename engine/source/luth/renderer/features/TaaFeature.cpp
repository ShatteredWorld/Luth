#include "luthpch.h"
#include "luth/renderer/features/TaaFeature.h"
#include "luth/renderer/subsystems/PostProcessSubsystem.h"
#include <cmath>

namespace Luth
{
    FeatureInfo TaaFeature::Describe() const
    {
        FeatureInfo info;
        info.name = "TAA"; info.phase = FeaturePhase::AfterAsync;
        info.resources.reads = {{RenderResources::TransparentHDR}, {RenderResources::MotionVectors},
            {RenderResources::LitDepth}, {TaaResources::Bindings}};
        ResourceWrite output{RenderResources::ResolvedHDR};
        output.disabledPassthrough = ResourceKeyRef{RenderResources::TransparentHDR};
        info.resources.writes = {output};
        return info;
    }
    void TaaFeature::Build(RG::RenderGraph& graph, RenderFeatureContext& ctx)
    {
        const auto* packet = ctx.resources.Get(TaaResources::Bindings).native;
        const std::array inputs{ctx.resources.Get(RenderResources::TransparentHDR),
            ctx.resources.Get(RenderResources::MotionVectors), ctx.resources.Get(RenderResources::LitDepth)};
        const auto single = [](const TextureBindingRef& binding) {
            return binding.texture && !binding.baseMip && binding.mipCount == 1 &&
                !binding.baseLayer && binding.layerCount == 1;
        };
        if (!packet || !ctx.view.width || !ctx.view.height)
            throw std::invalid_argument("TAA: missing native packet or view extent");
        if (packet->state) packet->state->recorded = false;
        const std::array formats{RG::TextureFormat::RGBA16_Float, RG::TextureFormat::RG16_Float, RG::TextureFormat::D32_Float};
        for (size_t i = 0; i < inputs.size(); ++i)
        {
            const auto& input = inputs[i];
            if (!input.handle.IsValid() || input.handle.index > graph.GetResources().size() || !single(input.binding))
                throw std::invalid_argument("TAA: missing stage input");
            const auto& desc = graph.GetResources()[input.handle.index - 1].desc;
            if (desc.format != formats[i] || desc.width != ctx.view.width || desc.height != ctx.view.height)
                throw std::invalid_argument("TAA: incompatible input format or extent");
        }
        GraphTextureRef output = inputs[0];
        if (!packet->enabled || !packet->pipeline)
        {
            if (packet->state) packet->state->history.Invalidate();
        }
        else
        {
            if (!packet->state || !packet->layout || !packet->set ||
                packet->width != ctx.view.width || packet->height != ctx.view.height ||
                !single(packet->previous) || !single(packet->current) ||
                !packet->previousImage || !packet->currentImage || !packet->previousView || !packet->currentView ||
                packet->previousImage == packet->currentImage || packet->previous.texture == packet->current.texture ||
                !std::isfinite(packet->constants.temporalAlpha) ||
                (packet->constants.temporalAlpha != -1.0f &&
                    (packet->constants.temporalAlpha < 0.0f || packet->constants.temporalAlpha > 1.0f)))
                throw std::invalid_argument("TAA: incomplete history or native bindings");
            for (size_t i = 0; i < inputs.size(); ++i)
                if (inputs[i].binding.texture != packet->sources[i].texture || !single(packet->sources[i]))
                    throw std::invalid_argument("TAA: stage input disagrees with sampled descriptor");
            for (u32 column = 0; column < 4; ++column)
                for (u32 row = 0; row < 4; ++row)
                    if (!std::isfinite(packet->constants.skyReproj[column][row]))
                        throw std::invalid_argument("TAA: invalid sky reprojection matrix");
            for (const auto& resource : graph.GetResources())
                if (resource.image == packet->previousImage || resource.image == packet->currentImage)
                    throw std::invalid_argument("TAA: history image already imported in this graph");
            output = {m_Native.AddTaaResolvePass(graph, inputs[0].handle, inputs[1].handle,
                inputs[2].handle, *packet, m_Debugger), packet->current};
        }
        ctx.resources.Publish(RenderResources::ResolvedHDR, output);
    }
}

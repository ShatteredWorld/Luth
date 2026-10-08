#include "luthpch.h"
#include "luth/renderer/features/BloomFeature.h"
#include "luth/renderer/subsystems/PostProcessSubsystem.h"
#include <cmath>

namespace Luth
{
    FeatureInfo BloomFeature::Describe() const
    {
        FeatureInfo info;
        info.name = "Bloom"; info.phase = FeaturePhase::AfterAsync;
        info.resources.reads = {{RenderResources::ResolvedHDR}, {BloomResources::Bindings}};
        info.resources.writes = {{RenderResources::BloomOutput, ResourceOutputPresence::Optional}};
        return info;
    }
    void BloomFeature::Build(RG::RenderGraph& graph, RenderFeatureContext& ctx)
    {
        const auto* packet = ctx.resources.Get(BloomResources::Bindings).native;
        const auto& input = ctx.resources.Get(RenderResources::ResolvedHDR);
        const auto single = [](const TextureBindingRef& binding) {
            return binding.texture && !binding.baseMip && binding.mipCount == 1 &&
                !binding.baseLayer && binding.layerCount == 1;
        };
        if (!packet || !ctx.view.width || !ctx.view.height || !input.handle.IsValid() ||
            input.handle.index > graph.GetResources().size() || !single(input.binding))
            throw std::invalid_argument("Bloom: missing native packet or HDR input");
        const auto& desc = graph.GetResources()[input.handle.index - 1].desc;
        if (desc.format != RG::TextureFormat::RGBA16_Float || desc.width != ctx.view.width || desc.height != ctx.view.height)
            throw std::invalid_argument("Bloom: incompatible HDR format or extent");
        if (!packet->enabled || !packet->downPipeline || !packet->upPipeline)
        {
            ctx.resources.PublishAbsent(RenderResources::BloomOutput);
            return;
        }
        if (!packet->state || !packet->downLayout || !packet->upLayout || !packet->prefilterSet ||
            packet->width != ctx.view.width || packet->height != ctx.view.height ||
            !single(packet->source) || packet->source.texture != input.binding.texture ||
            !std::isfinite(packet->threshold) || !std::isfinite(packet->radius))
            throw std::invalid_argument("Bloom: incomplete frozen bindings");
        for (u32 i = 0; i < BloomViewState::kMipCount; ++i)
        {
            if (!single(packet->mips[i]) || !packet->images[i] || !packet->views[i] ||
                packet->mips[i].texture == input.binding.texture ||
                (i < BloomViewState::kMipCount - 1 && (!packet->downSets[i] || !packet->upSets[i])))
                throw std::invalid_argument("Bloom: incomplete pyramid bindings");
            for (u32 j = 0; j < i; ++j)
                if (packet->images[i] == packet->images[j] || packet->mips[i].texture == packet->mips[j].texture)
                    throw std::invalid_argument("Bloom: aliased pyramid images");
            for (const auto& resource : graph.GetResources())
                if (resource.image == packet->images[i])
                    throw std::invalid_argument("Bloom: pyramid image already imported");
        }
        ctx.resources.Publish(RenderResources::BloomOutput,
            GraphTextureRef{m_Native.AddBloomPasses(graph, input.handle, *packet, m_Debugger), packet->mips[0]});
    }
}

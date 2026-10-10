#include "luthpch.h"
#include "luth/renderer/features/CompositeFeature.h"
#include "luth/renderer/subsystems/PostProcessSubsystem.h"

namespace Luth
{
    FeatureInfo CompositeFeature::Describe() const
    {
        FeatureInfo info;
        info.name = "TonemapComposite"; info.phase = FeaturePhase::AfterAsync;
        info.resources.reads = {{RenderResources::GridHDR},
            {RenderResources::BloomOutput, ResourceReadRequirement::Optional}, {CompositeResources::Bindings}};
        info.resources.writes = {{RenderResources::TonemappedLDR}};
        return info;
    }
    void CompositeFeature::Build(RG::RenderGraph& graph, RenderFeatureContext& ctx)
    {
        const auto* packet = ctx.resources.Get(CompositeResources::Bindings).native;
        const auto& input = ctx.resources.Get(RenderResources::GridHDR);
        const auto* bloom = ctx.resources.TryGet(RenderResources::BloomOutput);
        const auto single = [](const TextureBindingRef& b) {
            return b.texture && !b.baseMip && b.mipCount == 1 && !b.baseLayer && b.layerCount == 1;
        };
        const auto checkInput = [&](const GraphTextureRef& resource, u32 width, u32 height) {
            if (!resource.handle.IsValid() || resource.handle.index > graph.GetResources().size() || !single(resource.binding))
                throw std::invalid_argument("Composite: missing sampled input");
            const auto& desc = graph.GetResources()[resource.handle.index - 1].desc;
            if (desc.format != RG::TextureFormat::RGBA16_Float || desc.width != width || desc.height != height)
                throw std::invalid_argument("Composite: incompatible sampled format or extent");
        };
        if (!packet || !ctx.view.width || !ctx.view.height)
            throw std::invalid_argument("Composite: missing packet or view extent");
        checkInput(input, ctx.view.width, ctx.view.height);
        if (bloom) checkInput(*bloom, std::max(ctx.view.width / 2, 1u), std::max(ctx.view.height / 2, 1u));
        if (!packet->pipeline || !packet->layout || !packet->set || !packet->state ||
            !packet->uniform.buffer || packet->uniform.range != sizeof(PostProcessUBO) ||
            !single(packet->source) || packet->source.texture != input.binding.texture ||
            !single(packet->output) || !packet->outputOwner || packet->outputOwner.get() != packet->output.texture ||
            !packet->outputImage || !packet->outputView ||
            packet->width != ctx.view.width || packet->height != ctx.view.height ||
            packet->output.texture == input.binding.texture ||
            (bloom && (!single(packet->bloom) || packet->bloom.texture != bloom->binding.texture ||
                packet->output.texture == bloom->binding.texture)) ||
            (!bloom && (packet->bloom.texture || packet->parameters.bloomStrength != 0)))
            throw std::invalid_argument("Composite: incomplete or inconsistent frozen bindings");
        for (const auto& resource : graph.GetResources())
            if (resource.image == packet->outputImage)
                throw std::invalid_argument("Composite: LDR image already imported");
        ctx.resources.Publish(RenderResources::TonemappedLDR, GraphTextureRef{
            m_Native.AddCompositePass(graph, input.handle, bloom ? bloom->handle : RG::ResourceHandle{},
                *packet, m_Debugger), packet->output});
    }
}

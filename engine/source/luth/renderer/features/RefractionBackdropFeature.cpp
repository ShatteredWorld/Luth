#include "luthpch.h"
#include "luth/renderer/features/RefractionBackdropFeature.h"
#include "luth/renderer/subsystems/TransparencySubsystem.h"

namespace Luth
{
    FeatureInfo RefractionBackdropFeature::Describe() const
    {
        FeatureInfo info;
        info.name = "RefractionBackdrop"; info.phase = FeaturePhase::AfterAsync;
        info.activation = FeatureActivation::Conditional;
        info.resources.reads = {{RenderResources::FoggedHDR}, {RefractionResources::Bindings}};
        info.resources.writes = {{RenderResources::RefractionBackdrop, ResourceOutputPresence::Optional}};
        return info;
    }
    FeatureFrameDecision RefractionBackdropFeature::Evaluate(const FeaturePrepareContext& ctx) const
    {
        for (const auto resources : {ctx.frame.resources, ctx.view.resources})
            for (const auto& resource : resources)
                if (const auto* ref = resource.TryGet(RefractionResources::Bindings))
                {
                    if (!ref->native) throw std::invalid_argument("RefractionBackdrop: missing native packet");
                    return {ref->native->enabled && ref->native->binding.texture};
                }
        throw std::invalid_argument("RefractionBackdrop: missing frozen bindings");
    }
    void RefractionBackdropFeature::Build(RG::RenderGraph& graph, RenderFeatureContext& ctx)
    {
        const auto& source = ctx.resources.Get(RenderResources::FoggedHDR);
        const auto& packet = *ctx.resources.Get(RefractionResources::Bindings).native;
        const auto singleImage = [](TextureBindingRef binding) {
            return binding.texture && !binding.baseMip && binding.mipCount == 1 &&
                !binding.baseLayer && binding.layerCount == 1;
        };
        if (!source.handle.IsValid() || source.handle.index > graph.GetResources().size() ||
            !singleImage(source.binding) || !singleImage(packet.binding) || !packet.image || !packet.view ||
            !ctx.view.width || !ctx.view.height || packet.width != ctx.view.width || packet.height != ctx.view.height)
            throw std::invalid_argument("RefractionBackdrop: invalid stage, extent or native bindings");
        const auto& sourceNode = graph.GetResources()[source.handle.index - 1];
        if (sourceNode.desc.format != RG::TextureFormat::RGBA16_Float ||
            sourceNode.desc.width != packet.width || sourceNode.desc.height != packet.height ||
            source.binding.texture == packet.binding.texture || sourceNode.image == packet.image)
            throw std::invalid_argument("RefractionBackdrop: incompatible source or aliased copy destination");
        for (const auto& node : graph.GetResources())
            if (node.image == packet.image)
                throw std::invalid_argument("RefractionBackdrop: destination already imported in this graph");
        ctx.resources.Publish(RenderResources::RefractionBackdrop,
            m_Native.AddBackdropCopyPass(graph, source.handle, packet));
    }
}

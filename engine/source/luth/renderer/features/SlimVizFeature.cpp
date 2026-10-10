#include "luthpch.h"
#include "luth/renderer/features/SlimVizFeature.h"
#include "luth/renderer/subsystems/PostProcessSubsystem.h"
#include <cmath>
namespace Luth
{
    FeatureInfo SlimVizFeature::Describe() const
    {
        FeatureInfo info; info.name = "SlimVisualization"; info.phase = FeaturePhase::AfterAsync;
        info.resources.reads = {{RenderResources::TonemappedLDR}, {SlimVizResources::Bindings},
            {RenderResources::Normal, ResourceReadRequirement::Optional}, {RenderResources::Roughness, ResourceReadRequirement::Optional},
            {RenderResources::MotionVectors, ResourceReadRequirement::Optional}, {RenderResources::MaterialID, ResourceReadRequirement::Optional}};
        info.resources.writes = {{SlimVizResources::Output, ResourceOutputPresence::Required,
            ResourceKeyRef{RenderResources::TonemappedLDR}, true}};
        return info;
    }
    void SlimVizFeature::Build(RG::RenderGraph& graph, RenderFeatureContext& ctx)
    {
        const auto& input = ctx.resources.Get(RenderResources::TonemappedLDR);
        const auto* packet = ctx.resources.Get(SlimVizResources::Bindings).native;
        const auto valid = [&](const GraphTextureRef& ref, RG::TextureFormat format) {
            if (!ref.handle.IsValid() || ref.handle.index > graph.GetResources().size() || !ref.binding.texture ||
                ref.binding.baseMip || ref.binding.mipCount != 1 || ref.binding.baseLayer || ref.binding.layerCount != 1) return false;
            const auto& desc = graph.GetResources()[ref.handle.index - 1].desc;
            return desc.format == format && desc.width == ctx.view.width && desc.height == ctx.view.height;
        };
        if (!packet || !ctx.view.width || !ctx.view.height || !valid(input, RG::TextureFormat::RGBA8_Unorm))
            throw std::invalid_argument("SlimViz: missing packet or incompatible LDR stage");
        GraphTextureRef output = input;
        if (packet->enabled && packet->pipeline)
        {
            if (!packet->layout || !packet->set || !packet->state || packet->set != packet->state->set || packet->parameters.mode > 3 || !std::isfinite(packet->parameters.scale))
                throw std::invalid_argument("SlimViz: incomplete frozen native bindings or invalid mode/scale");
            const std::array refs{ctx.resources.TryGet(RenderResources::Normal), ctx.resources.TryGet(RenderResources::Roughness),
                ctx.resources.TryGet(RenderResources::MotionVectors), ctx.resources.TryGet(RenderResources::MaterialID)};
            const std::array formats{RG::TextureFormat::RG16_Float, RG::TextureFormat::R8_Unorm,
                RG::TextureFormat::RG16_Float, RG::TextureFormat::R16_Uint};
            std::array<RG::ResourceHandle, 4> handles;
            for (size_t i = 0; i < refs.size(); ++i)
            {
                if (!refs[i] || !valid(*refs[i], formats[i]) || packet->state->sources[i].get() != refs[i]->binding.texture ||
                    input.handle.index == refs[i]->handle.index || input.binding.texture == refs[i]->binding.texture)
                    throw std::invalid_argument("SlimViz: missing or incompatible G-buffer source");
                for (size_t j = 0; j < i; ++j)
                    if (refs[j]->handle.index == refs[i]->handle.index || refs[j]->binding.texture == refs[i]->binding.texture)
                        throw std::invalid_argument("SlimViz: aliased G-buffer sources");
                handles[i] = refs[i]->handle;
            }
            output.handle = m_Native.AddSlimVizPass(graph, input.handle, handles, *packet, m_Debugger);
        }
        ctx.resources.Publish(SlimVizResources::Output, output);
    }
}

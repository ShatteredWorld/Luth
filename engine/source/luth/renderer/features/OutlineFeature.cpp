#include "luthpch.h"
#include "luth/renderer/features/OutlineFeature.h"
#include "luth/renderer/subsystems/EditorOverlaysSubsystem.h"
namespace Luth
{
    FeatureInfo OutlineFeature::Describe() const
    {
        FeatureInfo info; info.name = "Outline"; info.phase = FeaturePhase::AfterAsync;
        info.resources.reads = {{RenderResources::VisualizedLDR}, {OutlineResources::Bindings},
            {RenderResources::SelectionMask, ResourceReadRequirement::Optional},
            {RenderResources::SelectionDepth, ResourceReadRequirement::Optional},
            {RenderResources::LitDepth, ResourceReadRequirement::Optional}};
        info.resources.writes = {{RenderResources::OutlinedLDR, ResourceOutputPresence::Required,
            ResourceKeyRef{RenderResources::VisualizedLDR}, true}};
        return info;
    }
    void OutlineFeature::Build(RG::RenderGraph& graph, RenderFeatureContext& ctx)
    {
        const auto& input = ctx.resources.Get(RenderResources::VisualizedLDR);
        const auto* packet = ctx.resources.Get(OutlineResources::Bindings).native;
        const auto valid = [&](const GraphTextureRef& ref, RG::TextureFormat format) {
            if (!ref.handle.IsValid() || ref.handle.index > graph.GetResources().size() || !ref.binding.texture ||
                ref.binding.baseMip || ref.binding.mipCount != 1 || ref.binding.baseLayer || ref.binding.layerCount != 1) return false;
            const auto& desc = graph.GetResources()[ref.handle.index - 1].desc;
            return desc.format == format && desc.width == ctx.view.width && desc.height == ctx.view.height;
        };
        if (!packet || !ctx.view.width || !ctx.view.height || !valid(input, RG::TextureFormat::RGBA8_Unorm))
            throw std::invalid_argument("Outline: missing packet or incompatible LDR stage");
        GraphTextureRef output = input;
        if (packet->enabled && packet->pipeline)
        {
            const auto* mask = ctx.resources.TryGet(RenderResources::SelectionMask);
            const auto* selectedDepth = ctx.resources.TryGet(RenderResources::SelectionDepth);
            if (bool(mask) != bool(selectedDepth))
                throw std::invalid_argument("Outline: selection mask and depth presence differ");
            if (mask)
            {
                const auto* depth = ctx.resources.TryGet(RenderResources::LitDepth);
                if (!depth || !valid(*mask, RG::TextureFormat::RGBA8_Unorm) ||
                    !valid(*selectedDepth, RG::TextureFormat::D32_Float) || !valid(*depth, RG::TextureFormat::D32_Float) ||
                    !packet->layout || !packet->set || !packet->state || packet->set != packet->state->outlineSet ||
                    packet->state->sources[0].get() != mask->binding.texture ||
                    packet->state->sources[1].get() != selectedDepth->binding.texture ||
                    packet->state->sources[2].get() != depth->binding.texture ||
                    input.binding.texture == mask->binding.texture || input.binding.texture == selectedDepth->binding.texture ||
                    input.binding.texture == depth->binding.texture ||
                    input.handle.index == mask->handle.index || input.handle.index == selectedDepth->handle.index ||
                    input.handle.index == depth->handle.index || mask->handle.index == selectedDepth->handle.index ||
                    mask->handle.index == depth->handle.index || selectedDepth->handle.index == depth->handle.index ||
                    packet->parameters.texelSizeX != 1.0f / ctx.view.width || packet->parameters.texelSizeY != 1.0f / ctx.view.height)
                    throw std::invalid_argument("Outline: incomplete or mismatched frozen bindings");
                output.handle = m_Native.AddOutlinePass(graph, input.handle, mask->handle, selectedDepth->handle,
                    depth->handle, *packet, m_Debugger);
            }
        }
        ctx.resources.Publish(RenderResources::OutlinedLDR, output);
    }
}

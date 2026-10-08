#include "luthpch.h"
#include "luth/renderer/features/DebugDrawFeature.h"
#include "luth/renderer/subsystems/DebugDrawSubsystem.h"
#include <cmath>
namespace Luth
{
    FeatureInfo DebugDrawFeature::Describe() const
    {
        FeatureInfo info; info.name = "DebugShapes"; info.phase = FeaturePhase::AfterAsync;
        info.resources.reads = {{RenderResources::OutlinedLDR}, {DebugDrawResources::Bindings}};
        info.resources.writes = {{RenderResources::FinalViewLDR, ResourceOutputPresence::Required,
            ResourceKeyRef{RenderResources::OutlinedLDR}, true}};
        return info;
    }
    void DebugDrawFeature::Build(RG::RenderGraph& graph, RenderFeatureContext& ctx)
    {
        const auto& input = ctx.resources.Get(RenderResources::OutlinedLDR);
        const auto* packet = ctx.resources.Get(DebugDrawResources::Bindings).native;
        if (!packet || !ctx.view.width || !ctx.view.height || !input.handle.IsValid() ||
            input.handle.index > graph.GetResources().size() || !input.binding.texture || input.binding.baseMip ||
            input.binding.mipCount != 1 || input.binding.baseLayer || input.binding.layerCount != 1)
            throw std::invalid_argument("DebugDraw: missing packet or incompatible LDR input");
        const auto& desc = graph.GetResources()[input.handle.index - 1].desc;
        if (desc.format != RG::TextureFormat::RGBA8_Unorm || desc.width != ctx.view.width || desc.height != ctx.view.height)
            throw std::invalid_argument("DebugDraw: incompatible LDR format or extent");
        GraphTextureRef output = input;
        if (packet->enabled && packet->pipeline && packet->vertexCount)
        {
            if (!packet->layout || !packet->vertices.buffer || packet->vertexCount % 2 ||
                packet->renderFrameIndex != ctx.frame.renderFrameIndex ||
                packet->vertices.size < u64(packet->vertexCount) * sizeof(DebugVertex) ||
                packet->vertices.offset % alignof(DebugVertex))
                throw std::invalid_argument("DebugDraw: invalid frozen vertex bindings");
            for (u32 col = 0; col < 4; ++col) for (u32 row = 0; row < 4; ++row)
                if (!std::isfinite(packet->viewProj[col][row])) throw std::invalid_argument("DebugDraw: non-finite view matrix");
            output.handle = m_Native.AddDebugDrawPass(graph, input.handle, *packet, m_Debugger);
        }
        ctx.resources.Publish(RenderResources::FinalViewLDR, output);
    }
}

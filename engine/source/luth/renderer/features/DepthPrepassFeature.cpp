#include "luthpch.h"
#include "luth/renderer/features/DepthPrepassFeature.h"
#include "luth/renderer/subsystems/GeometrySubsystem.h"

namespace Luth
{
    FeatureInfo DepthPrepassFeature::Describe() const
    {
        FeatureInfo info;
        info.name = "DepthPrepass";
        info.phase = FeaturePhase::BeforeAsync;
        info.resources.reads = {{RenderResources::CameraVisibleDraws}, {DepthPrepassResources::Target},
            {DepthPrepassResources::Bindings}};
        info.resources.writes = {{RenderResources::PrepassDepth, ResourceOutputPresence::Required,
            ResourceKeyRef{DepthPrepassResources::Target}, true}};
        info.capabilities.consumes = {{&DeformationResources::DeformedGeometry}};
        return info;
    }
    void DepthPrepassFeature::Prepare(FeaturePrepareContext& ctx)
    {
        if (!ctx.frame.draws || !ctx.frame.snapshot || !ctx.view.width || !ctx.view.height)
            throw std::invalid_argument("DepthPrepass: missing immutable frame/view inputs");
    }
    void DepthPrepassFeature::Build(RG::RenderGraph& graph, RenderFeatureContext& ctx)
    {
        const auto& target = ctx.resources.Get(DepthPrepassResources::Target);
        const auto& visible = ctx.resources.Get(RenderResources::CameraVisibleDraws);
        const auto& bindings = ctx.resources.Get(DepthPrepassResources::Bindings);
        if (!bindings.native || !target.handle.IsValid() || !target.binding.texture ||
            target.binding.baseMip || target.binding.mipCount != 1 ||
            target.binding.baseLayer || target.binding.layerCount != 1 ||
            !visible.indirect.handle.IsValid() || !visible.indirect.binding.slice ||
            visible.indirect.binding.offset != visible.indirect.binding.slice->offset ||
            visible.indirect.binding.size > visible.indirect.binding.slice->size ||
            (u64(visible.firstDraw) + visible.maxDrawCount) * sizeof(VkDrawIndexedIndirectCommand) > visible.indirect.binding.size)
            throw std::invalid_argument("DepthPrepass: invalid native bindings, target or visible draw range");
        const auto& resources = graph.GetResources();
        if (target.handle.index > resources.size())
            throw std::invalid_argument("DepthPrepass: target is outside the current graph");
        const auto& desc = resources[target.handle.index - 1].desc;
        if (desc.width != ctx.view.width || desc.height != ctx.view.height || desc.format != RG::TextureFormat::D32_Float)
            throw std::invalid_argument("DepthPrepass: target extent or format does not match the view");
        const auto output = m_Native.AddDepthPrepass(graph, target.handle, visible, ctx.view.width, ctx.view.height,
            *bindings.native, *ctx.frame.draws, *ctx.frame.snapshot, m_Debugger);
        ctx.resources.Publish(RenderResources::PrepassDepth, GraphTextureRef{output, target.binding});
    }
}

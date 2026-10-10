#include "luthpch.h"
#include "luth/renderer/features/SlimGBufferFeature.h"
#include "luth/renderer/subsystems/GeometrySubsystem.h"

namespace Luth
{
    FeatureInfo SlimGBufferFeature::Describe() const
    {
        FeatureInfo info;
        info.name = "SlimGBuffer";
        info.phase = FeaturePhase::BeforeAsync;
        info.resources.reads = {{RenderResources::CameraVisibleDraws}, {RenderResources::PrepassDepth},
            {SlimGBufferResources::NormalTarget}, {SlimGBufferResources::RoughnessTarget},
            {SlimGBufferResources::MotionTarget}, {SlimGBufferResources::MaterialTarget}, {SlimGBufferResources::Bindings}};
        info.resources.writes = {
            {RenderResources::Normal, ResourceOutputPresence::Required, ResourceKeyRef{SlimGBufferResources::NormalTarget}, true},
            {RenderResources::Roughness, ResourceOutputPresence::Required, ResourceKeyRef{SlimGBufferResources::RoughnessTarget}, true},
            {RenderResources::MotionVectors, ResourceOutputPresence::Required, ResourceKeyRef{SlimGBufferResources::MotionTarget}, true},
            {RenderResources::MaterialID, ResourceOutputPresence::Required, ResourceKeyRef{SlimGBufferResources::MaterialTarget}, true},
            {RenderResources::SurfaceDepth, ResourceOutputPresence::Required, ResourceKeyRef{RenderResources::PrepassDepth}, true}};
        info.capabilities.consumes = {{&DeformationResources::DeformedGeometry}};
        return info;
    }
    void SlimGBufferFeature::Prepare(FeaturePrepareContext& ctx)
    {
        if (!ctx.frame.draws || !ctx.frame.snapshot || !ctx.view.width || !ctx.view.height)
            throw std::invalid_argument("SlimGBuffer: missing immutable frame/view inputs");
    }
    void SlimGBufferFeature::Build(RG::RenderGraph& graph, RenderFeatureContext& ctx)
    {
        const std::array targets{ctx.resources.Get(SlimGBufferResources::NormalTarget),
            ctx.resources.Get(SlimGBufferResources::RoughnessTarget), ctx.resources.Get(SlimGBufferResources::MotionTarget),
            ctx.resources.Get(SlimGBufferResources::MaterialTarget)};
        const auto& depth = ctx.resources.Get(RenderResources::PrepassDepth);
        const auto& visible = ctx.resources.Get(RenderResources::CameraVisibleDraws);
        const auto& bindings = ctx.resources.Get(SlimGBufferResources::Bindings);
        if (!bindings.native || !visible.indirect.handle.IsValid() || !visible.indirect.binding.slice ||
            visible.indirect.binding.offset != visible.indirect.binding.slice->offset ||
            visible.indirect.binding.size > visible.indirect.binding.slice->size ||
            (u64(visible.firstDraw) + visible.maxDrawCount) * sizeof(VkDrawIndexedIndirectCommand) > visible.indirect.binding.size)
            throw std::invalid_argument("SlimGBuffer: invalid native packet or visible draw range");
        const auto validate = [&](const GraphTextureRef& image, RG::TextureFormat format) {
            if (!image.handle.IsValid() || image.handle.index > graph.GetResources().size() || !image.binding.texture ||
                image.binding.baseMip || image.binding.mipCount != 1 || image.binding.baseLayer || image.binding.layerCount != 1)
                throw std::invalid_argument("SlimGBuffer: invalid graph-local target or subresource");
            const auto& desc = graph.GetResources()[image.handle.index - 1].desc;
            if (desc.width != ctx.view.width || desc.height != ctx.view.height || desc.format != format)
                throw std::invalid_argument("SlimGBuffer: target extent or format does not match the view");
        };
        const std::array formats{RG::TextureFormat::RG16_Float, RG::TextureFormat::R8_Unorm,
            RG::TextureFormat::RG16_Float, RG::TextureFormat::R16_Uint};
        validate(depth, RG::TextureFormat::D32_Float);
        for (u32 i = 0; i < 4; ++i)
        {
            validate(targets[i], formats[i]);
            for (u32 j = 0; j < i; ++j)
                if (targets[i].handle.index == targets[j].handle.index)
                    throw std::invalid_argument("SlimGBuffer: color targets must be distinct");
        }
        const auto output = m_Native.AddSlimGBufferPass(graph, targets, depth.handle, visible, ctx.view.width, ctx.view.height,
            *bindings.native, *ctx.frame.draws, *ctx.frame.snapshot, m_Debugger);
        ctx.resources.Publish(RenderResources::Normal, GraphTextureRef{output[0], targets[0].binding});
        ctx.resources.Publish(RenderResources::Roughness, GraphTextureRef{output[1], targets[1].binding});
        ctx.resources.Publish(RenderResources::MotionVectors, GraphTextureRef{output[2], targets[2].binding});
        ctx.resources.Publish(RenderResources::MaterialID, GraphTextureRef{output[3], targets[3].binding});
        ctx.resources.Publish(RenderResources::SurfaceDepth, GraphTextureRef{output[4], depth.binding});
    }
}

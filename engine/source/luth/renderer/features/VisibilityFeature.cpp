#include "luthpch.h"
#include "luth/renderer/features/VisibilityFeature.h"
#include "luth/renderer/subsystems/GeometrySubsystem.h"
#include "luth/renderer/draw/DrawCommand.h"
#include <limits>

namespace Luth
{
    FeatureInfo VisibilityFeature::Describe() const
    {
        FeatureInfo info;
        info.name = "Visibility";
        info.phase = FeaturePhase::BeforeAsync;
        info.activation = FeatureActivation::Conditional;
        info.resources.reads = {{RenderResources::ObjectData}, {RenderResources::InitializedIndirectData},
            {VisibilityResources::Parameters}};
        info.resources.writes = {{RenderResources::CameraVisibleDraws, ResourceOutputPresence::Optional},
            {RenderResources::CascadeVisibleDraws, ResourceOutputPresence::Optional},
            {VisibilityResources::CulledIndirect, ResourceOutputPresence::Optional,
                ResourceKeyRef{RenderResources::InitializedIndirectData}, true}};
        info.capabilities.consumes = {{&DeformationResources::DeformedGeometry}};
        return info;
    }
    FeatureFrameDecision VisibilityFeature::Evaluate(const FeaturePrepareContext& ctx) const
    {
        for (const auto bindings : {ctx.frame.resources, ctx.view.resources})
            for (const auto& binding : bindings)
                if (const auto* parameters = binding.TryGet(VisibilityResources::Parameters))
                    return {parameters->realtime};
        throw std::invalid_argument("Visibility: missing frozen parameters");
    }
    void VisibilityFeature::Build(RG::RenderGraph& graph, RenderFeatureContext& ctx)
    {
        const auto& parameters = ctx.resources.Get(VisibilityResources::Parameters);
        const auto& objects = ctx.resources.Get(RenderResources::ObjectData);
        auto indirect = ctx.resources.Get(RenderResources::InitializedIndirectData);
        const u64 regionCount = u64(parameters.maxViews) * 5;
        if (!parameters.regionStride || !parameters.maxViews || parameters.viewIndex >= parameters.maxViews ||
            parameters.objectCount > parameters.regionStride ||
            regionCount > std::numeric_limits<u32>::max() / parameters.regionStride)
            throw std::invalid_argument("Visibility: invalid draw region layout");
        const u64 commandCount = regionCount * parameters.regionStride;
        if (!objects.handle.IsValid() || !indirect.handle.IsValid() ||
            !objects.binding.slice || !indirect.binding.slice ||
            objects.binding.offset != objects.binding.slice->offset ||
            indirect.binding.offset != indirect.binding.slice->offset ||
            objects.binding.size > objects.binding.slice->size ||
            indirect.binding.size > indirect.binding.slice->size ||
            objects.binding.size < u64(parameters.objectCount) * sizeof(GPUObjectData) ||
            indirect.binding.size < commandCount * sizeof(VkDrawIndexedIndirectCommand))
            throw std::invalid_argument("Visibility: invalid draw regions or graph-local buffer bindings");

        const u32 firstDraw = parameters.viewIndex * 5 * parameters.regionStride;
        const auto native = m_Native.PrepareCullBindings(ctx.frame.renderFrameIndex, parameters.objectCount);
        indirect.handle = m_Native.AddCullPass(graph, objects.handle, indirect.handle,
            parameters.cameraPlanes, firstDraw, "FrustumCull.Cam", native, m_Debugger);
        if (parameters.cullCascades)
            for (u32 cascade = 0; cascade < 4; ++cascade)
            {
                const auto name = "FrustumCull.C" + std::to_string(cascade);
                indirect.handle = m_Native.AddCullPass(graph, objects.handle, indirect.handle,
                    parameters.cascadePlanes[cascade], firstDraw + (cascade + 1) * parameters.regionStride,
                    name.c_str(), native, m_Debugger);
            }
        // All writes address disjoint ranges in the same initialized physical slice.
        // Export its final version so later native readers observe every cull contribution.
        ctx.resources.Publish(VisibilityResources::CulledIndirect, indirect);
        ctx.resources.Publish(RenderResources::CameraVisibleDraws,
            VisibleDrawRange{indirect, firstDraw, parameters.objectCount});
        if (parameters.cullCascades)
        {
            CascadeDrawRanges cascades;
            for (u32 cascade = 0; cascade < 4; ++cascade)
                cascades.cascades[cascade] = {indirect,
                    firstDraw + (cascade + 1) * parameters.regionStride, parameters.objectCount};
            ctx.resources.Publish(RenderResources::CascadeVisibleDraws, cascades);
        }
        else ctx.resources.PublishAbsent(RenderResources::CascadeVisibleDraws);
    }
}

#include "luthpch.h"
#include "luth/renderer/debug/GraphInstrumentation.h"
#include "luth/core/diagnostics/Profiler.h"

namespace Luth
{
    RG::RenderGraphSnapshot CaptureGraphSnapshot(const RG::RenderGraph& rg)
    {
        LH_PROFILE_FUNCTION();

        RG::RenderGraphSnapshot snapshot;

        // Snapshot resources
        auto& resources = rg.GetResources();
        snapshot.resources.reserve(resources.size());
        for (auto& res : resources)
        {
            RG::ResourceSnapshot rs;
            rs.name        = res.desc.name;
            rs.width       = res.desc.width;
            rs.height      = res.desc.height;
            rs.format      = res.desc.format;
            rs.isExternal  = res.external;
            rs.isTransient = res.isTransient;
            snapshot.resources.push_back(std::move(rs));
        }

        // Snapshot passes
        auto& passes = rg.GetPasses();
        snapshot.passes.reserve(passes.size());
        for (auto& pass : passes)
        {
            RG::PassSnapshot ps;
            ps.name                = pass.name;
            ps.culled              = pass.culled;
            ps.numColorAttachments = (u32)pass.colorAttachments.size();
            ps.hasDepth            = pass.hasDepth;

            for (auto& r : pass.reads)
            {
                RG::PassSnapshotResource sr;
                sr.index = r.index;
                sr.name  = (r.index > 0 && r.index <= resources.size()) ? resources[r.index - 1].desc.name : "?";
                ps.reads.push_back(std::move(sr));
            }

            for (auto& w : pass.writes)
            {
                RG::PassSnapshotResource sw;
                sw.index = w.index;
                sw.name  = (w.index > 0 && w.index <= resources.size()) ? resources[w.index - 1].desc.name : "?";
                ps.writes.push_back(std::move(sw));
            }

            // Compute primaryOutputIndex from first color write, or depth if depth-only pass
            if (!pass.colorAttachments.empty())
            {
                u32 idx = pass.colorAttachments[0].handle.index;
                if (idx > 0 && idx <= resources.size())
                    ps.primaryOutputIndex = (int)(idx - 1);
            }
            else if (pass.hasDepth)
            {
                u32 idx = pass.depthAttachment.handle.index;
                if (idx > 0 && idx <= resources.size())
                    ps.primaryOutputIndex = (int)(idx - 1);
            }

            const auto& metadata = pass.debugMetadata;
            ps.shaderName = metadata.shaderName;
            ps.pipelineStateAvailable = metadata.pipelineStateAvailable;
            ps.pipelineStateMixed = metadata.pipelineStateMixed;
            ps.depthTest = metadata.depthTest; ps.depthWrite = metadata.depthWrite;
            ps.blendEnabled = metadata.blendEnabled; ps.cullMode = metadata.cullMode;
            ps.geometryStatsAvailable = metadata.geometryStatsAvailable;
            ps.indirectDraws = metadata.indirectDraws;
            if (!pass.culled) { ps.drawCalls = metadata.drawCalls; ps.indices = metadata.indices; }
            snapshot.passes.push_back(std::move(ps));
        }

        // Barrier inspector: fill from the solved graph when capture is on (off by default).
        if (RG::RenderGraph::BarrierCapture())
            rg.CaptureBarrierRecords(snapshot);

        return snapshot;
    }
}

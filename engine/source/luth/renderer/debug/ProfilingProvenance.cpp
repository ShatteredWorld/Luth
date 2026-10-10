#include "luthpch.h"
#include "luth/renderer/debug/ProfilingProvenance.h"
#include <stdexcept>

namespace Luth
{
    void ProfilingProvenance::Prepare(RenderViewId view, u64 generation, u64 frame, const RG::RenderGraph& graph)
    {
        if (!view.value || !generation) throw std::invalid_argument("Profiling requires a registered view generation");
        std::vector<ProfilePassIdentity> passes;
        bool afterAsync = false;
        const auto& nodes = graph.GetPasses();
        for (u32 i = 0; i < nodes.size(); ++i)
        {
            const auto& node = nodes[i];
            if (node.culled) continue;
            const bool async = node.queueFamily == RG::QueueFamily::AsyncCompute;
            if (async) afterAsync = true;
            passes.push_back({i, node.name, async ? ProfileSubmissionPhase::AsyncCompute
                : afterAsync ? ProfileSubmissionPhase::GraphicsB : ProfileSubmissionPhase::GraphicsA, node.isCompute});
        }
        auto& record = m_Views[view.value];
        if (!record.topologyGeneration || record.resourceGeneration != generation || record.passes != passes)
            ++record.topologyGeneration;
        record.view = view;
        record.resourceGeneration = generation;
        record.renderFrameIndex = frame;
        record.passes = std::move(passes);
        record.completion = {};
    }

    bool ProfilingProvenance::Submit(RenderViewId view, u64 renderFrameIndex, SubmissionCompletionToken token)
    {
        auto it = m_Views.find(view.value);
        if (it == m_Views.end() || it->second.renderFrameIndex != renderFrameIndex) return false;
        it->second.completion = token;
        return true;
    }

    const ProfileSubmissionProvenance* ProfilingProvenance::Find(RenderViewId view) const
    {
        auto it = m_Views.find(view.value);
        return it == m_Views.end() ? nullptr : &it->second;
    }
}

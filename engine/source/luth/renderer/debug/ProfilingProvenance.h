#pragma once
#include "luth/renderer/SubmissionCompletion.h"
#include "luth/renderer/features/RenderViewState.h"
#include "luth/renderer/rendergraph/RenderGraph.h"
#include <string>
#include <vector>
#include <unordered_map>

namespace Luth
{
    enum class ProfileSubmissionPhase : u8 { GraphicsA, AsyncCompute, GraphicsB };
    struct ProfilePassIdentity
    {
        u32 graphIndex = 0;
        std::string name;
        ProfileSubmissionPhase phase = ProfileSubmissionPhase::GraphicsA;
        bool compute = false;
        bool operator==(const ProfilePassIdentity&) const = default;
    };
    struct ProfileSubmissionProvenance
    {
        RenderViewId view;
        u64 resourceGeneration = 0, renderFrameIndex = 0, topologyGeneration = 0;
        std::vector<ProfilePassIdentity> passes; // Dense query index -> recorded graph pass.
        SubmissionCompletionToken completion;
    };

    // Foundation only: latest recording per view, not a query-slot retirement store.
    // A caller retaining a historical submission must copy the immutable record.
    class ProfilingProvenance
    {
    public:
        void Prepare(RenderViewId, u64 resourceGeneration, u64 renderFrameIndex, const RG::RenderGraph&);
        // Reject a missing/stale recording when graph construction exited early.
        bool Submit(RenderViewId, u64 renderFrameIndex, SubmissionCompletionToken);
        const ProfileSubmissionProvenance* Find(RenderViewId) const;
        void Release(RenderViewId view) { m_Views.erase(view.value); }
    private:
        std::unordered_map<u64, ProfileSubmissionProvenance> m_Views;
    };
}

#pragma once
#include "luth/renderer/debug/ProfilingProvenance.h"
#include "luth/renderer/rendergraph/RenderGraphSnapshot.h"
#include <array>
#include <optional>
#include <stdexcept>
#include "luth/core/FrameData.h"

namespace Luth
{
    class ProfilingQueryEpoch
    {
    public:
        bool Observe(const SubmissionCompletionToken& token)
        {
            if (!token.valid || !token.generation) return false;
            const bool changed = m_Generation && m_Generation != token.generation;
            m_Generation = token.generation;
            return changed;
        }
    private:
        u64 m_Generation = 0;
    };

    // CPU ownership only. Invalid submits stay quarantined until an external GPU-safe teardown.
    class ProfilingQuerySlots
    {
    public:
        enum class State { Free, Recording, Submitted, Quarantined };
        struct Slot { State state = State::Free; ProfileSubmissionProvenance record; bool stats = false; };
        static constexpr u32 Count = MAX_FRAMES_IN_FLIGHT;
        std::optional<u32> Acquire(const ProfileSubmissionProvenance& record, bool stats)
        {
            for (u32 i = 0; i < Count; ++i)
                if (m_Slots[i].state == State::Free)
                {
                    m_Slots[i] = {State::Recording, record, stats};
                    m_Slots[i].record.completion = {};
                    return i;
                }
            return {};
        }
        void Submit(u32 i, SubmissionCompletionToken token)
        {
            auto& slot = m_Slots.at(i);
            if (slot.state != State::Recording) throw std::logic_error("Query slot is not recording");
            slot.record.completion = token;
            slot.state = token.valid ? State::Submitted : State::Quarantined;
        }
        template<class IsComplete> bool CanRead(u32 i, IsComplete&& complete) const
        {
            const auto& slot = m_Slots.at(i);
            return slot.state == State::Submitted && complete(slot.record.completion);
        }
        template<class IsComplete> bool Retire(u32 i, IsComplete&& complete)
        {
            if (!CanRead(i, complete)) return false;
            m_Slots.at(i) = {};
            return true;
        }
        const Slot& Get(u32 i) const { return m_Slots.at(i); }
    private:
        std::array<Slot, Count> m_Slots;
    };

    struct ProfileQueryResults
    {
        ProfileSubmissionProvenance record;
        std::vector<float> times;
        std::vector<RG::GpuPipelineStats> stats;
    };

    inline bool ApplyProfileResults(const ProfileQueryResults& results,
        const ProfileSubmissionProvenance& current, RG::RenderGraphSnapshot& snapshot)
    {
        if (results.record.view != current.view || results.record.resourceGeneration != current.resourceGeneration
            || results.record.topologyGeneration != current.topologyGeneration || results.record.passes != current.passes
            || results.times.size() != current.passes.size() || results.stats.size() != current.passes.size()) return false;
        // Validate the whole mapping before modifying the snapshot.
        for (const auto& pass : current.passes)
            if (pass.graphIndex >= snapshot.passes.size() || snapshot.passes[pass.graphIndex].culled
                || snapshot.passes[pass.graphIndex].name != pass.name) return false;
        float totalMs = 0;
        RG::GpuPipelineStats total{};
        bool completeTiming = !current.passes.empty();
        for (u32 i = 0; i < current.passes.size(); ++i)
        {
            auto& pass = snapshot.passes[current.passes[i].graphIndex];
            pass.gpuTimeMs = results.times[i];
            pass.stats = results.stats[i];
            if (pass.gpuTimeMs >= 0) totalMs += pass.gpuTimeMs;
            else completeTiming = false;
            if (!pass.stats.valid) continue;
            total.inputVertices += pass.stats.inputVertices;
            total.inputPrimitives += pass.stats.inputPrimitives;
            total.vsInvocations += pass.stats.vsInvocations;
            total.clipInvocations += pass.stats.clipInvocations;
            total.clipPrimitives += pass.stats.clipPrimitives;
            total.fsInvocations += pass.stats.fsInvocations;
            total.valid = true;
        }
        snapshot.totalGpuTimeMs = completeTiming ? totalMs : -1.0f;
        snapshot.totalStats = total;
        snapshot.profilingAvailable = completeTiming;
        snapshot.profileRenderFrameIndex = results.record.renderFrameIndex;
        snapshot.profileTopologyGeneration = results.record.topologyGeneration;
        snapshot.profileViewId = results.record.view.value;
        return true;
    }
}

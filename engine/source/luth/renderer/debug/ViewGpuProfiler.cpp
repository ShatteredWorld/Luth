#include "luthpch.h"
#include "luth/renderer/debug/ViewGpuProfiler.h"
#include "luth/renderer/Renderer.h"

namespace Luth
{
    GPUTimerPool* ViewGpuProfiler::Prepare(const ProfileSubmissionProvenance& record,
        RG::RenderGraphSnapshot& snapshot, bool applyPrevious)
    {
        if (m_ResetRequired)
        {
            // Old timeline tokens cannot prove completion after sync recreation.
            // Drain all recorded work before replacing its query pools, including any
            // secondary view submitted earlier in this consuming frame.
            Renderer::WaitForGPU();
            Shutdown();
            m_ResetRequired = false;
        }
        auto& owner = m_Views[record.view.value];
        if (!owner) { owner = std::make_unique<View>(); owner->pool.Init(256); }
        auto& view = *owner;
        for (u32 i = 0; i < ProfilingQuerySlots::Count; ++i)
        {
            if (!view.slots.CanRead(i, [](const auto& token) { return Renderer::IsSubmissionComplete(token); })) continue;
            const auto& slot = view.slots.Get(i);
            ProfileQueryResults result;
            result.record = slot.record;
            view.pool.ReadResults(i, (u32)slot.record.passes.size(), result.times);
            view.pool.ReadStats(i, (u32)slot.record.passes.size(), slot.stats, result.stats);
            if (!view.latest || result.record.renderFrameIndex > view.latest->record.renderFrameIndex)
                view.latest = std::move(result);
            view.slots.Retire(i, [](const auto& token) { return Renderer::IsSubmissionComplete(token); });
        }
        if (applyPrevious && view.latest) ApplyProfileResults(*view.latest, record, snapshot);
        if (!view.pool.IsInitialized() || record.passes.size() > view.pool.MaxPasses()) return nullptr;
        view.recording = view.slots.Acquire(record, GPUTimerPool::StatsEnabled() && view.pool.StatsSupported());
        if (!view.recording) return nullptr; // GPU behind: never reset an in-flight pool.
        view.pool.SelectSlot(*view.recording, view.slots.Get(*view.recording).stats);
        return &view.pool;
    }

    void ViewGpuProfiler::Submit(RenderViewId id, u64 frame, SubmissionCompletionToken token)
    {
        m_ResetRequired |= m_Epoch.Observe(token);
        auto it = m_Views.find(id.value);
        if (it == m_Views.end() || !it->second->recording) return;
        auto& view = *it->second;
        const auto slot = *view.recording;
        if (view.slots.Get(slot).record.renderFrameIndex != frame) return;
        view.slots.Submit(slot, token);
        view.recording.reset();
    }
    void ViewGpuProfiler::Release(RenderViewId id)
    {
        auto it = m_Views.find(id.value);
        if (it == m_Views.end()) return;
        it->second->pool.Shutdown();
        m_Views.erase(it);
    }
    void ViewGpuProfiler::Shutdown()
    {
        for (auto& [id, view] : m_Views) view->pool.Shutdown();
        m_Views.clear();
    }
}

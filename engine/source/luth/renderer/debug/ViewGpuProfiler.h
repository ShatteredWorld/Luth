#pragma once
#include "luth/renderer/debug/ProfilingQuerySlots.h"
#include "luth/renderer/backend/vulkan/GPUTimerPool.h"
#include <memory>

namespace Luth
{
    // RenderingSystem owns this service; teardown requires a GPU-safe point.
    class ViewGpuProfiler
    {
    public:
        GPUTimerPool* Prepare(const ProfileSubmissionProvenance&, RG::RenderGraphSnapshot&, bool applyPrevious);
        void Submit(RenderViewId, u64 renderFrameIndex, SubmissionCompletionToken);
        void Release(RenderViewId);
        void Shutdown();
    private:
        struct View
        {
            GPUTimerPool pool;
            ProfilingQuerySlots slots;
            std::optional<u32> recording;
            std::optional<ProfileQueryResults> latest;
        };
        std::unordered_map<u64, std::unique_ptr<View>> m_Views;
        ProfilingQueryEpoch m_Epoch;
        bool m_ResetRequired = false;
    };
}

#pragma once
#include "luth/renderer/features/RenderViewState.h"

namespace Luth
{
    // Registration is provisional. Only successful recording and submission advance
    // history; copy-only denoising and absent producers never register it.
    struct DiTemporalHistory
    {
        ViewHistoryState committed;
        bool pending = false;
        u64 pendingFrame = 0, pendingGeneration = 0;
        bool CanReuse(u64 frame, u64 generation, bool cameraContinuous) const
        { return cameraContinuous && committed.CanReuse(frame, generation); }
        void Begin() { pending = false; }
        void Record(u64 frame, u64 generation)
        { pending = true; pendingFrame = frame; pendingGeneration = generation; }
        void Finish(u64 frame, u64 generation, bool submitted)
        {
            if (submitted && pending && pendingFrame == frame && pendingGeneration == generation)
                committed.Commit(frame, generation);
            else committed.Invalidate();
            pending = false;
        }
        void Invalidate() { committed.Invalidate(); pending = false; }
    };
}

#pragma once
#include "luth/core/types/LuthTypes.h"

namespace Luth
{
    struct SubmissionCompletionToken
    {
        u64 generation = 0;
        u64 graphicsValue = 0, computeValue = 0;
        u64 frameIndex = 0;
        u32 viewSlot = 0;
        bool valid = false;
    };

    struct SubmissionCompletionProgress
    {
        u64 generation = 0;
        u64 graphicsValue = 0, computeValue = 0;

        bool Contains(const SubmissionCompletionToken& token) const
        {
            return token.valid && token.generation && token.generation == generation
                && token.graphicsValue && graphicsValue >= token.graphicsValue
                && computeValue >= token.computeValue;
        }
    };
}

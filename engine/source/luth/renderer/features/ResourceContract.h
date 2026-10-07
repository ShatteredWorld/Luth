#pragma once

#include "luth/renderer/features/RenderResource.h"
#include <optional>
#include <vector>

namespace Luth
{
    enum class ResourceReadRequirement { Required, Optional };
    enum class ResourceOutputPresence { Required, Optional };

    struct ResourceRead
    {
        ResourceKeyRef key;
        ResourceReadRequirement requirement = ResourceReadRequirement::Required;
    };

    struct ResourceWrite
    {
        ResourceKeyRef key;
        ResourceOutputPresence presence = ResourceOutputPresence::Required;
        // Declares a physical alias, not an RG handle version. The compiler must order
        // consumers of aliasesInput before a transform that mutates that input.
        std::optional<ResourceKeyRef> aliasesInput;
        bool mutatesInput = false;
        bool exclusive = false;
        // An allocating transform can pass its input through when disabled without
        // claiming its active output is the same physical image (e.g. TAA).
        std::optional<ResourceKeyRef> disabledPassthrough;
    };

    struct ResourceContract
    {
        std::vector<ResourceRead> reads;
        std::vector<ResourceWrite> writes;
    };
}

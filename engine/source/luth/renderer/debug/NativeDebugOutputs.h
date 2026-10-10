#pragma once
#include "luth/renderer/debug/DebugOutputCatalog.h"

namespace Luth
{
    class FrameTargets;
    class LightingSubsystem;
    struct ViewResources;
    struct GtaoViewState;
    DebugOutputs CollectSharedDebugOutputs(const LightingSubsystem&);
    // ViewResources is an explicit migration bridge until the remaining RT states move.
    DebugOutputs CollectViewDebugOutputs(const FrameTargets&, const ViewResources*, const GtaoViewState*);
}

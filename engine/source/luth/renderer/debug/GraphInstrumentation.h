#pragma once

#include "luth/renderer/rendergraph/RenderGraph.h"
#include "luth/renderer/rendergraph/RenderGraphSnapshot.h"


namespace Luth
{
    // Detached compiled topology and contributor-owned frozen metadata.
    // Query readback and submission provenance belong to profiling infrastructure.
    RG::RenderGraphSnapshot CaptureGraphSnapshot(const RG::RenderGraph&);
}

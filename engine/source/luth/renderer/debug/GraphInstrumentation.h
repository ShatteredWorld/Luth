#pragma once

#include "luth/renderer/rendergraph/RenderGraph.h"
#include "luth/renderer/rendergraph/RenderGraphSnapshot.h"
#include "luth/renderer/draw/DrawList.h"

namespace Luth
{
    // Detached CPU snapshot of a compiled graph and its frozen draw list. Native
    // pass names and legacy pipeline-state enrichment remain compatible here;
    // query readback and submission provenance belong to profiling infrastructure.
    RG::RenderGraphSnapshot CaptureGraphSnapshot(const RG::RenderGraph&, const DrawList&);
}

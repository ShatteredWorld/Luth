#include "luthpch.h"
#include "luth/renderer/presentation/ViewPresentation.h"
#include <stdexcept>

namespace Luth
{
    void AddViewOutputExport(RG::RenderGraph& graph, RG::ResourceHandle finalLdr)
    {
        if (!finalLdr.IsValid()) throw std::invalid_argument("View output export requires final LDR");
        struct Data { RG::ResourceHandle output; };
        // An inline pass on Graphics avoids an empty rendering scope/secondary job.
        // RG's existing first-async split chooses graphics A or B for this contribution.
        graph.AddComputePass<Data>("ViewOutputExport", RG::QueueFamily::Graphics,
            [finalLdr](Data& data, RG::RenderPassBuilder& builder) {
                data.output = builder.Read(finalLdr);
                builder.SetHasSideEffect();
            }, [](Data&, RG::RenderPassContext&) {});
    }
}

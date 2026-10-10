#include <doctest/doctest.h>
#include "luth/renderer/presentation/ViewPresentation.h"
#include "luth/renderer/debug/ProfilingProvenance.h"
#include "luth/renderer/FrameDebugger.h"
#include <stdexcept>
using namespace Luth;
namespace
{
    struct Data {};
    RG::TextureDesc Desc(const char* name) { RG::TextureDesc d; d.name = name; d.width = 64; d.height = 32; return d; }
    u32 Transitions(const std::vector<RG::Barrier>& barriers, RG::ResourceState before, RG::ResourceState after)
    {
        u32 count = 0;
        for (const auto& b : barriers) if (b.before == before && b.after == after) ++count;
        return count;
    }
    ViewPresentationInputs Inputs()
    {
        ViewPresentationInputs inputs; inputs.backbuffer = Desc("Backbuffer");
        inputs.backbuffer.format = RG::TextureFormat::BGRA8_Unorm;
        inputs.image = (void*)1; inputs.imageView = (void*)2; return inputs;
    }
}

TEST_CASE("ViewPresentation: graphics-only export derives sampled transition in graphics A")
{
    Memory::LinearAllocator scratch(65536); RG::RenderGraph graph(scratch);
    const auto output = graph.RegisterResource(Desc("FinalLDR"));
    graph.AddPass<Data>("Tonemap", [&](Data&, RG::RenderPassBuilder& b) { b.Write(output); }, [](Data&, RG::RenderPassContext&) {});
    AddViewOutputExport(graph, output); graph.Compile();
    REQUIRE(graph.GetPasses().size() == 2);
    CHECK_FALSE(graph.GetPasses()[1].culled);
    CHECK(graph.GetPasses()[1].queueFamily == RG::QueueFamily::Graphics);
    REQUIRE(graph.GetPasses()[1].reads.size() == 1);
    CHECK(graph.GetPasses()[1].reads[0].index == output.index);
    CHECK(Transitions(graph.GetPasses()[1].preBarriers, RG::ResourceState::ColorAttachment, RG::ResourceState::ShaderResource) == 1);
    CHECK(graph.GetResources().size() == 1); // Export forwards the producer, without another import.
    ProfilingProvenance provenance; provenance.Prepare({1}, 1, 1, graph);
    CHECK(provenance.Find({1})->passes.back().phase == ProfileSubmissionPhase::GraphicsA);
    CHECK_THROWS_AS(AddViewOutputExport(graph, {}), std::invalid_argument);
}

TEST_CASE("ViewPresentation: async export stays live and transitions storage output in graphics B")
{
    Memory::LinearAllocator scratch(65536); RG::RenderGraph graph(scratch);
    const auto output = graph.RegisterResource(Desc("FinalLDR"));
    graph.AddComputePass<Data>("Resolve", RG::QueueFamily::AsyncCompute,
        [&](Data&, RG::RenderPassBuilder& b) { b.WriteStorageImage(output); }, [](Data&, RG::RenderPassContext&) {});
    AddViewOutputExport(graph, output); graph.Compile();
    CHECK_FALSE(graph.GetPasses()[0].culled); CHECK_FALSE(graph.GetPasses()[1].culled);
    REQUIRE(graph.GetPasses()[1].preBarriers.size() == 1);
    CHECK(graph.GetPasses()[1].preBarriers[0].before == RG::ResourceState::ComputeWrite);
    CHECK(graph.GetPasses()[1].preBarriers[0].after == RG::ResourceState::ShaderResource);
    CHECK(graph.GetPasses()[1].preBarriers[0].crossQueueSrc);
    ProfilingProvenance provenance; provenance.Prepare({2}, 1, 1, graph);
    CHECK(provenance.Find({2})->passes.back().phase == ProfileSubmissionPhase::GraphicsB);
}

TEST_CASE("ViewPresentation: primary UI and frozen minimal graph preserve backbuffer present")
{
    Memory::LinearAllocator scratch(65536); RG::RenderGraph graph(scratch), frozen(scratch);
    FrameDebugger debugger;
    const auto output = graph.RegisterResource(Desc("FinalLDR"));
    graph.AddPass<Data>("Tonemap", [&](Data&, RG::RenderPassBuilder& b) { b.Write(output); }, [](Data&, RG::RenderPassContext&) {});
    AddViewOutputExport(graph, output);
    AddViewImGuiPass(graph, Inputs(), debugger, output); graph.Compile();
    REQUIRE(graph.GetPasses().size() == 3); REQUIRE(graph.GetResources().size() == 2);
    const auto& ui = graph.GetPasses().back();
    CHECK(ui.name == "ImGuiPass"); CHECK_FALSE(ui.culled);
    REQUIRE(ui.reads.size() == 1); CHECK(ui.reads[0].index == output.index);
    CHECK(ui.preBarriers.size() == 1); // Backbuffer undefined -> attachment; LDR already sampled.
    CHECK(Transitions(ui.postBarriers, RG::ResourceState::ColorAttachment, RG::ResourceState::Present) == 1);
    AddViewImGuiPass(frozen, Inputs(), debugger); frozen.Compile();
    REQUIRE(frozen.GetPasses().size() == 1); REQUIRE(frozen.GetResources().size() == 1);
    CHECK(frozen.GetPasses()[0].reads.empty()); CHECK_FALSE(frozen.GetPasses()[0].culled);
    CHECK(Transitions(frozen.GetPasses()[0].postBarriers, RG::ResourceState::ColorAttachment, RG::ResourceState::Present) == 1);
    auto invalid = Inputs(); invalid.image = nullptr;
    CHECK_THROWS_AS(AddViewImGuiPass(frozen, invalid, debugger), std::invalid_argument);
    CHECK(frozen.GetResources().size() == 1);
}

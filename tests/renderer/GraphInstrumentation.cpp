#include <doctest/doctest.h>
#include "luth/renderer/debug/GraphInstrumentation.h"
#include "luth/memory/LinearAllocator.h"

using namespace Luth;
namespace
{
    struct Data {};
    struct BarrierCaptureScope
    {
        bool previous = RG::RenderGraph::BarrierCapture();
        explicit BarrierCaptureScope(bool enabled) { RG::RenderGraph::SetBarrierCapture(enabled); }
        ~BarrierCaptureScope() { RG::RenderGraph::SetBarrierCapture(previous); }
    };
    RG::TextureDesc Desc(const char* name, RG::TextureFormat format = RG::TextureFormat::RGBA16_Float)
    {
        RG::TextureDesc desc; desc.name = name; desc.width = 64; desc.height = 32; desc.format = format;
        return desc;
    }
}

TEST_CASE("GraphInstrumentation: compiled topology and primary outputs survive graph mutation")
{
    BarrierCaptureScope capture(false);
    Memory::LinearAllocator scratch(65536); RG::RenderGraph graph(scratch);
    const auto color = graph.RegisterResource(Desc("Color"));
    const auto depth = graph.ImportResource(Desc("Depth", RG::TextureFormat::D32_Float),
        (void*)1, (void*)2, RG::ResourceState::Undefined);
    graph.AddPass<Data>("Dead", [](Data&, RG::RenderPassBuilder&) {}, [](Data&, RG::RenderPassContext&) {});
    graph.AddPass<Data>("DepthPrepass", [&](Data&, RG::RenderPassBuilder& builder) { builder.WriteDepth(depth); },
        [](Data&, RG::RenderPassContext&) {});
    graph.AddPass<Data>("GeometryPass", [&](Data&, RG::RenderPassBuilder& builder) { builder.Read(depth); builder.Write(color); },
        [](Data&, RG::RenderPassContext&) {});
    graph.Compile();
    DrawList draws; draws.opaque.resize(2); draws.cutout.resize(1); draws.transparent.resize(1);
    const RG::RenderGraph& compiled = graph;
    auto snapshot = CaptureGraphSnapshot(compiled, draws);
    REQUIRE(snapshot.passes.size() == 3);
    REQUIRE(snapshot.resources.size() == 2);
    CHECK(snapshot.passes[0].culled);
    CHECK_FALSE(snapshot.passes[1].culled);
    CHECK(snapshot.passes[1].hasDepth);
    CHECK(snapshot.passes[1].primaryOutputIndex == 1);
    CHECK(snapshot.passes[2].primaryOutputIndex == 0);
    CHECK(snapshot.passes[2].numColorAttachments == 1);
    REQUIRE(snapshot.passes[2].reads.size() == 1);
    REQUIRE(snapshot.passes[2].writes.size() == 1);
    CHECK(snapshot.passes[2].reads[0].index == depth.index);
    CHECK(snapshot.passes[2].reads[0].name == "Depth");
    CHECK(snapshot.passes[2].writes[0].name == "Color");
    CHECK(snapshot.passes[2].drawCalls == 4);
    CHECK(snapshot.passes[2].indices == 0); // No mesh resources in this headless draw list.
    CHECK(snapshot.passes[2].shaderName == "pbr");
    CHECK(snapshot.passes[2].depthTest);
    CHECK(snapshot.passes[2].depthWrite);
    CHECK(snapshot.resources[0].width == 64);
    CHECK(snapshot.resources[0].height == 32);
    CHECK_FALSE(snapshot.resources[0].isExternal);
    CHECK(snapshot.resources[1].isExternal);
    CHECK(snapshot.resources[1].format == RG::TextureFormat::D32_Float);
    CHECK(snapshot.barriers.empty());
    CHECK(snapshot.passes[2].gpuTimeMs == -1);
    graph.GetResources()[0].desc.name = "Changed";
    draws.Clear();
    CHECK(snapshot.resources[0].name == "Color");
    CHECK(snapshot.passes[2].drawCalls == 4);
}

TEST_CASE("GraphInstrumentation: barrier inspection is detached and honors capture setting")
{
    BarrierCaptureScope capture(true);
    Memory::LinearAllocator scratch(65536); RG::RenderGraph graph(scratch);
    const auto color = graph.RegisterResource(Desc("Output"));
    graph.AddPass<Data>("Writer", [&](Data&, RG::RenderPassBuilder& builder) { builder.Write(color); },
        [](Data&, RG::RenderPassContext&) {});
    graph.AddPass<Data>("Reader", [&](Data&, RG::RenderPassBuilder& builder) { builder.Read(color); builder.SetHasSideEffect(); },
        [](Data&, RG::RenderPassContext&) {});
    graph.Compile();
    const auto snapshot = CaptureGraphSnapshot(graph, {});
    CHECK_FALSE(snapshot.barriers.empty());
    CHECK(snapshot.numImageBarriers > 0);
    CHECK(snapshot.passes[1].numImageBarriers > 0);
    CHECK(snapshot.barriers.back().resource == "Output");
    RG::RenderGraph::SetBarrierCapture(false);
    const auto disabled = CaptureGraphSnapshot(graph, {});
    CHECK(disabled.barriers.empty());
    CHECK(disabled.numImageBarriers == 0);
    CHECK_FALSE(snapshot.barriers.empty());
}

TEST_CASE("GraphInstrumentation: legacy enrichment and empty graphs keep default query state")
{
    BarrierCaptureScope capture(false);
    Memory::LinearAllocator scratch(65536); RG::RenderGraph graph(scratch);
    graph.Compile();
    const auto empty = CaptureGraphSnapshot(graph, {});
    CHECK(empty.resources.empty());
    CHECK(empty.passes.empty());
    CHECK(empty.totalGpuTimeMs == 0);
    CHECK_FALSE(empty.totalStats.valid);
    for (const auto* name : {"ShadowPass", "SkyboxPass", "PostProcess", "ImGuiPass", "Custom"})
        graph.AddPass<Data>(name, [](Data&, RG::RenderPassBuilder& builder) { builder.SetHasSideEffect(); },
            [](Data&, RG::RenderPassContext&) {});
    graph.Compile();
    DrawList draws; draws.opaque.resize(3);
    const auto snapshot = CaptureGraphSnapshot(graph, draws);
    CHECK(snapshot.passes[0].shaderName == "shadowDepth");
    CHECK(snapshot.passes[0].cullMode == VK_CULL_MODE_FRONT_BIT);
    CHECK(snapshot.passes[0].drawCalls == 3);
    CHECK(snapshot.passes[1].shaderName == "skybox");
    CHECK_FALSE(snapshot.passes[1].depthWrite);
    CHECK(snapshot.passes[1].drawCalls == 1);
    CHECK(snapshot.passes[2].shaderName == "postprocess");
    CHECK(snapshot.passes[3].shaderName == "imgui");
    CHECK(snapshot.passes[3].blendEnabled);
    CHECK(snapshot.passes[4].shaderName.empty());
    CHECK(snapshot.passes[4].primaryOutputIndex == -1);
}

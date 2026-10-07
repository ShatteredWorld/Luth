#include <doctest/doctest.h>
#include "luth/renderer/features/DeformationFeature.h"
#include "luth/renderer/features/RenderPipelineCompiler.h"
#include "luth/renderer/subsystems/SkinningSubsystem.h"
#include "luth/renderer/rendergraph/RenderGraph.h"
#include "luth/core/RenderSnapshot.h"
#include "luth/memory/LinearAllocator.h"

using namespace Luth;

namespace
{
    struct DeformationFixture
    {
        SkinningSubsystem native; // No Vulkan device: register empty native dispatch lists.
        std::unique_ptr<CompiledRenderPipeline> pipeline;
        DeformationFixture()
        {
            RenderPipelineDefinition definition;
            definition.AddFeature<DeformationFeature>(native);
            PipelineInputContract inputs;
            inputs.resources = {{DeformationResources::Parameters}};
            auto compiled = RenderPipelineCompiler{}.Compile(std::move(definition), {}, inputs);
            REQUIRE(compiled.pipeline);
            pipeline = std::move(compiled.pipeline);
        }
    };
}

TEST_CASE("DeformationFeature: empty views and disabled wind retain the graphics A pass [renderfeatures]")
{
    DeformationFixture fixture;
    const RenderSnapshot snapshot{};
    for (const bool windEnabled : {false, true})
        for (const u64 viewId : {1ull, 2ull})
        {
            Memory::LinearAllocator scratch(64 * 1024);
            RG::RenderGraph graph(scratch);
            DeformationParameters parameters;
            parameters.wind.enabled = windEnabled;
            const std::array bindings{RenderInputBinding::Present(DeformationResources::Parameters, parameters)};
            FrameRenderInputs frame;
            frame.snapshot = &snapshot;
            frame.renderFrameIndex = 17;
            frame.resources = bindings;
            ViewRenderInputs view;
            view.id = {viewId};
            REQUIRE(fixture.pipeline->Build(graph, frame, view, scratch).success);
            REQUIRE(graph.GetPasses().size() == 1);
            CHECK(graph.GetPasses()[0].name == "Deform");
            CHECK(graph.GetPasses()[0].queueFamily == RG::QueueFamily::Graphics);
            CHECK(graph.GetResources().empty()); // BDA buffers retain their existing native barrier.
            graph.Compile();
            CHECK_FALSE(graph.GetPasses()[0].culled);
        }
}

TEST_CASE("DeformationFeature: missing frozen inputs reject construction before registration [renderfeatures]")
{
    DeformationFixture fixture;
    Memory::LinearAllocator scratch(64 * 1024);
    RG::RenderGraph graph(scratch);
    const DeformationParameters parameters{};
    const std::array bindings{RenderInputBinding::Present(DeformationResources::Parameters, parameters)};
    FrameRenderInputs frame;
    ViewRenderInputs view;
    SUBCASE("snapshot missing") { frame.resources = bindings; }
    SUBCASE("parameters missing") { static const RenderSnapshot snapshot{}; frame.snapshot = &snapshot; }
    const auto result = fixture.pipeline->Build(graph, frame, view, scratch);
    CHECK_FALSE(result.success);
    CHECK_FALSE(result.diagnostics.empty());
    CHECK(graph.GetPasses().empty());
}

TEST_CASE("DeformationFeature: capability is explicit and does not require RT [renderfeatures]")
{
    SkinningSubsystem native;
    const auto info = DeformationFeature(native).Describe();
    CHECK(info.phase == FeaturePhase::BeforeAsync);
    CHECK(info.activation == FeatureActivation::Always);
    REQUIRE(info.capabilities.provides.size() == 1);
    CHECK(info.capabilities.provides[0] == &DeformationResources::DeformedGeometry);
    CHECK(info.capabilities.consumes.empty());
    CHECK(info.capabilities.deviceRequirements.empty());
}

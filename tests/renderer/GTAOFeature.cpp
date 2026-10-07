#include <doctest/doctest.h>
#include "luth/core/diagnostics/Log.h"
#include "luth/renderer/features/GTAOFeature.h"
#include "luth/renderer/features/RenderPipelineCompiler.h"
#include "luth/renderer/subsystems/GTAOSubsystem.h"
#include "luth/renderer/CameraParams.h"
#include "luth/memory/LinearAllocator.h"

using namespace Luth;
using namespace Luth::RenderResources;

namespace
{
    // Opaque native identities only. Tests register/compile graphs and never record Vulkan.
    template<class T> T Native(u64 value) { return reinterpret_cast<T>(static_cast<uintptr_t>(value)); }
    struct Fixture
    {
        GTAOSubsystem native;
        GtaoViewStateStore states;
        std::unique_ptr<CompiledRenderPipeline> pipeline;
        CameraParams camera;
        Fixture()
        {
            RenderPipelineDefinition definition;
            definition.AddFeature<GTAOFeature>(native, states);
            PipelineInputContract inputs;
            inputs.resources = {{SurfaceDepth}, {GtaoResources::Parameters}};
            auto compiled = RenderPipelineCompiler{}.Compile(std::move(definition), {}, inputs);
            REQUIRE(compiled.pipeline);
            pipeline = std::move(compiled.pipeline);
        }
        std::shared_ptr<GtaoViewState> State(RenderViewId id, u32 width, u32 height, int& waits)
        {
            return states.Ensure(id, {width, height}, [id, width, height](const ViewStateConfig&) {
                auto state = std::make_shared<GtaoViewState>();
                state->width = width; state->height = height;
                state->halfWidth = std::max(width / 2, 1u); state->halfHeight = std::max(height / 2, 1u);
                state->depthSource = Native<const Texture*>(id.value * 10);
                state->linearBinding = {Native<VkImage>(id.value * 10 + 1), Native<VkImageView>(id.value * 10 + 1), {}};
                state->rawBinding = {Native<VkImage>(id.value * 10 + 2), Native<VkImageView>(id.value * 10 + 2), {}};
                state->finalBinding = {Native<VkImage>(id.value * 10 + 3), Native<VkImageView>(id.value * 10 + 3),
                    {Native<const Texture*>(id.value * 10 + 3)}};
                return state;
            }, [&] { ++waits; });
        }
        PipelineBuildResult Build(RG::RenderGraph& graph, Memory::LinearAllocator& scratch,
            RenderViewId id, u32 width, u32 height, GtaoFrameParameters params,
            GraphTextureRef& output, const Texture* depthOverride = nullptr)
        {
            RG::TextureDesc desc;
            desc.name = "SurfaceDepth"; desc.width = width; desc.height = height;
            desc.format = RG::TextureFormat::D32_Float;
            const GraphTextureRef depth{graph.ImportResource(desc, (void*)1, (void*)1,
                RG::ResourceState::DepthStencilAttachment),
                {depthOverride ? depthOverride : Native<const Texture*>(id.value * 10)}};
            const std::array inputs{RenderInputBinding::Present(SurfaceDepth, depth),
                RenderInputBinding::Present(GtaoResources::Parameters, params)};
            ViewRenderInputs view;
            view.id = id; view.width = width; view.height = height; view.camera = &camera; view.resources = inputs;
            FrameRenderInputs frame;
            frame.renderFrameIndex = 17;
            const std::array outputs{RenderOutputBinding::Capture(AmbientOcclusion, output)};
            return pipeline->Build(graph, frame, view, scratch, outputs);
        }
    };
}

TEST_CASE("GTAOFeature: native three-pass graph preserves dependencies and external images [renderfeatures]")
{
    Fixture fixture;
    int waits = 0;
    auto state = fixture.State({1}, 1279, 719, waits);
    Memory::LinearAllocator scratch(64 * 1024);
    RG::RenderGraph graph(scratch);
    GraphTextureRef output;
    const auto result = fixture.Build(graph, scratch, {1}, 1279, 719, {true, true, 4}, output);
    REQUIRE(result.success);
    REQUIRE(graph.GetPasses().size() == 3);
    const auto& passes = graph.GetPasses();
    CHECK(passes[0].name == "GTAODepthPrefilter");
    CHECK(passes[1].name == "GTAOMain");
    CHECK(passes[2].name == "GTAODenoise");
    for (const auto& pass : passes) CHECK(pass.queueFamily == RG::QueueFamily::AsyncCompute);
    REQUIRE(passes[0].reads.size() == 1);
    REQUIRE(passes[1].reads.size() == 1);
    REQUIRE(passes[2].reads.size() == 2);
    for (const auto& pass : passes) REQUIRE(pass.writes.size() == 1);
    CHECK(passes[0].reads[0].index == 1);
    CHECK(passes[1].reads[0] == passes[0].writes[0]);
    CHECK(passes[2].reads[0] == passes[1].writes[0]);
    CHECK(passes[2].reads[1] == passes[0].writes[0]);
    CHECK(output.handle == passes[2].writes[0]);
    CHECK(output.binding.texture == state->finalBinding.binding.texture);
    REQUIRE(graph.GetResources().size() == 4); // Original depth and three GTAO imports.
    const auto& final = graph.GetResources()[output.handle.index - 1];
    CHECK(final.image == state->finalBinding.image);
    CHECK(final.desc.width == 639);
    CHECK(final.desc.height == 359);
    CHECK(final.desc.format == RG::TextureFormat::R8_Unorm);
    graph.Compile();
    for (const auto& pass : graph.GetPasses()) CHECK_FALSE(pass.culled);
    CHECK(waits == 0);
}

TEST_CASE("GTAOFeature: disabled and PT frames publish absence without native state or passes [renderfeatures]")
{
    Fixture fixture;
    GtaoFrameParameters params;
    SUBCASE("Disabled") { params = {false, true, 1}; }
    SUBCASE("Path tracing") { params = {true, false, 1}; }
    Memory::LinearAllocator scratch(64 * 1024);
    RG::RenderGraph graph(scratch);
    GraphTextureRef output{{99, 7}, {Native<const Texture*>(100)}};
    REQUIRE(fixture.Build(graph, scratch, {1}, 640, 480, params, output).success);
    CHECK(graph.GetPasses().empty());
    CHECK_FALSE(output.handle.IsValid());
    CHECK(output.binding.texture == nullptr);
    CHECK(graph.GetResources().size() == 1);
    CHECK(fixture.states.Find({1}) == nullptr);
}

TEST_CASE("GTAOFeature: two views retain separate state and resize only the affected domain [renderfeatures]")
{
    Fixture fixture;
    int waits = 0;
    auto scene = fixture.State({1}, 1280, 720, waits);
    auto game = fixture.State({2}, 800, 600, waits);
    CHECK(fixture.State({1}, 1280, 720, waits) == scene);
    auto resized = fixture.State({1}, 1920, 1080, waits);
    CHECK(resized != scene);
    CHECK(*fixture.states.Find({2}) == game);
    CHECK(waits == 1);
    for (const auto& state : {resized, game})
    {
        const RenderViewId id{state == game ? 2u : 1u};
        Memory::LinearAllocator scratch(64 * 1024);
        RG::RenderGraph graph(scratch);
        GraphTextureRef output;
        REQUIRE(fixture.Build(graph, scratch, id, state->width, state->height, {true, true, 8}, output).success);
        CHECK(graph.GetResources()[output.handle.index - 1].image == state->finalBinding.image);
        CHECK(output.binding.texture == state->finalBinding.binding.texture);
    }
    fixture.pipeline->ReleaseView({1});
    CHECK(fixture.states.Find({1}) == nullptr);
    CHECK(*fixture.states.Find({2}) == game);
}

TEST_CASE("GTAOFeature: invalid preparation and depth bindings reject construction and reset outputs [renderfeatures]")
{
    Fixture fixture;
    int waits = 0;
    const Texture* overrideDepth = nullptr;
    SUBCASE("Missing state") {}
    SUBCASE("Stale extent") { fixture.State({1}, 320, 240, waits); }
    SUBCASE("Different physical depth")
    {
        fixture.State({1}, 640, 480, waits);
        overrideDepth = Native<const Texture*>(500);
    }
    Memory::LinearAllocator scratch(64 * 1024);
    RG::RenderGraph graph(scratch);
    GraphTextureRef output{{99, 7}, {Native<const Texture*>(100)}};
    const auto result = fixture.Build(graph, scratch, {1}, 640, 480, {true, true, 1}, output, overrideDepth);
    CHECK_FALSE(result.success);
    REQUIRE_FALSE(result.diagnostics.empty());
    CHECK(result.diagnostics[0].code == PipelineDiagnosticCode::BuildContractViolation);
    CHECK(graph.GetPasses().empty());
    CHECK_FALSE(output.handle.IsValid());
    CHECK(output.binding.texture == nullptr);
}

TEST_CASE("GTAOFeature: enabled disabled enabled graphs do not retain previous references [renderfeatures]")
{
    Fixture fixture;
    int waits = 0;
    auto state = fixture.State({1}, 640, 480, waits);
    GraphTextureRef output;
    for (bool enabled : {true, false, true})
    {
        Memory::LinearAllocator scratch(64 * 1024);
        RG::RenderGraph graph(scratch);
        REQUIRE(fixture.Build(graph, scratch, {1}, 640, 480, {enabled, true, 1}, output).success);
        CHECK(output.handle.IsValid() == enabled);
        CHECK(graph.GetPasses().size() == (enabled ? 3u : 0u));
        CHECK(*fixture.states.Find({1}) == state);
    }
    CHECK(waits == 0);
}

TEST_CASE("GTAOFeature: graph-local exports reject unknown or wrongly typed keys before registration [renderfeatures]")
{
    Fixture fixture;
    int waits = 0;
    fixture.State({1}, 640, 480, waits);
    GraphTextureRef output{{99, 7}, {Native<const Texture*>(100)}};
    auto key = SurfaceDepth;
    SUBCASE("Unknown key") { key = BloomOutput; }
    SUBCASE("Wrong type") { key = RenderResourceKey<GraphTextureRef>{GtaoResources::Parameters.identity}; }
    const std::array outputs{RenderOutputBinding::Capture(key, output)};
    Memory::LinearAllocator scratch(64 * 1024);
    RG::RenderGraph graph(scratch);
    const auto result = fixture.pipeline->Build(graph, {}, {}, scratch, outputs);
    CHECK_FALSE(result.success);
    REQUIRE_FALSE(result.diagnostics.empty());
    CHECK(result.diagnostics[0].code == PipelineDiagnosticCode::InvalidInput);
    CHECK(result.diagnostics[0].resource.identity == key.identity);
    CHECK(graph.GetPasses().empty());
    CHECK_FALSE(output.handle.IsValid());
    CHECK(output.binding.texture == nullptr);
}

TEST_CASE("GTAOFeature: graphics B prefix prevents a later async contribution [renderfeatures]")
{
    Fixture fixture;
    int waits = 0;
    fixture.State({1}, 640, 480, waits);
    Memory::LinearAllocator scratch(64 * 1024);
    RG::RenderGraph graph(scratch);
    struct Data {};
    for (const auto queue : {RG::QueueFamily::AsyncCompute, RG::QueueFamily::Graphics})
        graph.AddComputePass<Data>("LegacyPrefix", queue,
            [](Data&, RG::RenderPassBuilder& builder) { builder.SetHasSideEffect(); },
            [](Data&, RG::RenderPassContext&) {});
    GraphTextureRef output;
    const auto result = fixture.Build(graph, scratch, {1}, 640, 480, {true, true, 1}, output);
    CHECK_FALSE(result.success);
    REQUIRE_FALSE(result.diagnostics.empty());
    CHECK(result.diagnostics[0].code == PipelineDiagnosticCode::PhaseViolation);
    CHECK_FALSE(output.handle.IsValid()); // Partial graph must be discarded, never recorded.
}

#include <doctest/doctest.h>
#include "luth/renderer/features/rt/RtSunShadowFeature.h"
#include "luth/renderer/features/rt/RtSunShadowBindings.h"
#include "luth/renderer/features/RenderPipelineCompiler.h"
#include "luth/renderer/rendergraph/RenderGraph.h"
#include "luth/memory/LinearAllocator.h"

using namespace Luth;
TEST_CASE("RtSunShadowFeature: scene demand follows shadow activation and PT exclusion [renderfeatures]")
{
    int prepares = 0, registrations = 0;
    RtSceneParameters params;
    auto packet = std::make_shared<PreparedRtScene>();
    packet->frameIndex = 42;
    RenderPipelineDefinition definition;
    definition.AddFeature<RtSceneFeature>(
        [&](const FrameRenderInputs&, const RtSceneParameters&) -> std::shared_ptr<const PreparedRtScene> {
            ++prepares; return packet;
        },
        [&](RG::RenderGraph&, std::shared_ptr<const PreparedRtScene>) { ++registrations; });
    definition.AddFeature<RtSunShadowDemandFeature>();
    PipelineInputContract inputs;
    inputs.resources = {{RtSceneResources::Parameters}};
    auto compiled = RenderPipelineCompiler{}.Compile(std::move(definition),
        RendererCapabilities{{&RtSceneResources::AccelerationStructures, &RtSceneResources::RayQueries}}, inputs);
    REQUIRE(compiled.pipeline);
    Memory::LinearAllocator scratch(64 * 1024);
    const std::array bindings{RenderInputBinding::Present(RtSceneResources::Parameters, params)};
    FrameRenderInputs frame;
    frame.renderFrameIndex = 42; frame.resources = bindings;
    for (int step = 0; step < 3; ++step) {
        params.active[static_cast<size_t>(RtSceneConsumer::SunShadow)] = step != 0;
        params.active[static_cast<size_t>(RtSceneConsumer::PathTrace)] = step == 2;
        RG::RenderGraph graph(scratch);
        REQUIRE(compiled.pipeline->Build(graph, frame, {}, scratch).success);
        CHECK(prepares == (step == 0 ? 0 : 1));
        CHECK(registrations == prepares);
    }
    RenderPipelineDefinition missing;
    missing.AddFeature<RtSunShadowDemandFeature>();
    CHECK_FALSE(RenderPipelineCompiler{}.Compile(std::move(missing),
        RendererCapabilities{{&RtSceneResources::RayQueries}}, inputs).pipeline);
    RenderPipelineDefinition unsupported;
    unsupported.AddFeature<RtSunShadowFeature>();
    CHECK_FALSE(RenderPipelineCompiler{}.Compile(std::move(unsupported), {}, inputs).pipeline);
}
namespace
{
    template<class T> T Native(u64 value) { return reinterpret_cast<T>(static_cast<uintptr_t>(value)); }
    struct Fixture
    {
        RtSceneParameters params;
        RtSunShadowBindings bindings;
        PreparedRtScene scene;
        std::unique_ptr<CompiledRenderPipeline> pipeline;
        Fixture()
        {
            params.active[static_cast<size_t>(RtSceneConsumer::SunShadow)] = true;
            bindings.pipeline = Native<VkPipeline>(1);
            bindings.layout = Native<VkPipelineLayout>(2);
            bindings.sets.fill(Native<VkDescriptorSet>(3));
            bindings.image = Native<VkImage>(4);
            bindings.imageView = Native<VkImageView>(5);
            bindings.mask = {Native<const Texture*>(6)};
            bindings.depthSource = Native<const Texture*>(7);
            bindings.normalSource = Native<const Texture*>(8);
            bindings.tlas = Native<VkAccelerationStructureKHR>(9);
            bindings.view = {1};
            bindings.generation = 3;
            bindings.frameIndex = 42;
            bindings.width = 1279;
            bindings.height = 719;
            scene.frameIndex = 42;
            scene.emptyFallback = bindings.tlas;
            RenderPipelineDefinition definition;
            definition.AddFeature<RtSunShadowFeature>();
            PipelineInputContract inputs;
            inputs.resources = {{RtSceneResources::Parameters}, {RtSunShadowResources::Bindings},
                {RenderResources::SurfaceDepth}, {RenderResources::Normal},
                {RtSceneResources::Scene, ResourceOutputPresence::Optional}};
            inputs.capabilities = {&RtSceneResources::RayScene};
            auto compiled = RenderPipelineCompiler{}.Compile(std::move(definition),
                RendererCapabilities{{&RtSceneResources::AccelerationStructures, &RtSceneResources::RayQueries}}, inputs);
            REQUIRE(compiled.pipeline);
            pipeline = std::move(compiled.pipeline);
        }
        PipelineBuildResult Build(RG::RenderGraph& graph, Memory::LinearAllocator& scratch,
            GraphTextureRef& output, bool haveScene = true, bool validDepth = true)
        {
            RG::TextureDesc desc;
            desc.width = bindings.width; desc.height = bindings.height;
            desc.format = RG::TextureFormat::D32_Float;
            auto depth = graph.ImportResource(desc, (void*)Native<VkImage>(10), (void*)Native<VkImageView>(11),
                RG::ResourceState::DepthStencilAttachment);
            desc.format = RG::TextureFormat::RG16_Float;
            auto normal = graph.ImportResource(desc, (void*)Native<VkImage>(12), (void*)Native<VkImageView>(13),
                RG::ResourceState::ColorAttachment);
            GraphTextureRef depthRef{validDepth ? depth : RG::ResourceHandle{99, 0}, {Native<const Texture*>(7)}};
            GraphTextureRef normalRef{normal, {Native<const Texture*>(8)}};
            const RtSunShadowBindingRef ref{&bindings};
            const RaySceneRef sceneRef{&scene};
            const std::array resources{RenderInputBinding::Present(RtSceneResources::Parameters, params),
                RenderInputBinding::Present(RtSunShadowResources::Bindings, ref),
                RenderInputBinding::Present(RenderResources::SurfaceDepth, depthRef),
                RenderInputBinding::Present(RenderResources::Normal, normalRef),
                haveScene ? RenderInputBinding::Present(RtSceneResources::Scene, sceneRef) : RenderInputBinding::Absent(RtSceneResources::Scene)};
            FrameRenderInputs frame;
            frame.renderFrameIndex = 42; frame.resources = resources;
            if (haveScene) frame.capabilities = RtSceneResources::Requests;
            ViewRenderInputs view;
            view.id = {1}; view.resourceGeneration = 3;
            view.width = 1279; view.height = 719;
            const std::array outputs{RenderOutputBinding::Capture(RtSunShadowResources::Mask, output)};
            return pipeline->Build(graph, frame, view, scratch, outputs);
        }
    };
}
TEST_CASE("RtSunShadowFeature: samples actual producers and publishes one async R8 mask [renderfeatures]")
{
    Fixture fixture;
    Memory::LinearAllocator scratch(64 * 1024);
    RG::RenderGraph graph(scratch);
    GraphTextureRef output;
    REQUIRE(fixture.Build(graph, scratch, output).success);
    REQUIRE(graph.GetPasses().size() == 1);
    const auto& pass = graph.GetPasses()[0];
    CHECK(pass.name == "RtSunShadows");
    CHECK(pass.queueFamily == RG::QueueFamily::AsyncCompute);
    REQUIRE(pass.reads.size() == 2);
    CHECK(pass.reads[0].index == 1);
    CHECK(pass.reads[1].index == 2);
    REQUIRE(pass.writes.size() == 1);
    CHECK(output.handle == pass.writes[0]);
    CHECK(output.binding.texture == fixture.bindings.mask.texture);
    REQUIRE(graph.GetResources().size() == 3);
    const auto& mask = graph.GetResources()[2];
    CHECK(mask.image == fixture.bindings.image);
    CHECK(mask.desc.format == RG::TextureFormat::R8_Unorm);
    CHECK(mask.desc.width == 1279);
    CHECK(mask.desc.height == 719);
    graph.Compile();
    CHECK_FALSE(graph.GetPasses()[0].culled);
}
TEST_CASE("RtSunShadowFeature: disabled CSM PT and missing shader publish absence [renderfeatures]")
{
    Fixture fixture;
    SUBCASE("CSM or shadows disabled") { fixture.params.active[0] = false; }
    SUBCASE("Path tracing") { fixture.params.active[static_cast<size_t>(RtSceneConsumer::PathTrace)] = true; }
    SUBCASE("Shader unavailable") { fixture.bindings.pipeline = VK_NULL_HANDLE; }
    Memory::LinearAllocator scratch(64 * 1024);
    RG::RenderGraph graph(scratch);
    GraphTextureRef output{{99, 4}, {Native<const Texture*>(100)}};
    REQUIRE(fixture.Build(graph, scratch, output, false).success);
    CHECK(graph.GetPasses().empty());
    CHECK_FALSE(output.handle.IsValid());
    CHECK(output.binding.texture == nullptr);
    CHECK(graph.GetResources().size() == 2);
}
TEST_CASE("RtSunShadowFeature: stale scene descriptors and wrong physical sources fail before registration [renderfeatures]")
{
    Fixture fixture;
    bool haveScene = true, validDepth = true;
    SUBCASE("Missing provider") { haveScene = false; }
    SUBCASE("Scene frame") { fixture.scene.frameIndex = 41; }
    SUBCASE("Binding frame") { fixture.bindings.frameIndex = 41; }
    SUBCASE("View") { fixture.bindings.view = {2}; }
    SUBCASE("Generation") { ++fixture.bindings.generation; }
    SUBCASE("TLAS pairing") { fixture.bindings.tlas = Native<VkAccelerationStructureKHR>(100); }
    SUBCASE("Table pairing") { fixture.bindings.geometryTable = 123; }
    SUBCASE("Incomplete populated scene") { fixture.scene.tlas.result.instanceCount = 1; }
    SUBCASE("Descriptor missing") { fixture.bindings.sets[0] = VK_NULL_HANDLE; }
    SUBCASE("Wrong sampled source") { fixture.bindings.depthSource = Native<const Texture*>(100); }
    SUBCASE("Mask alias") { fixture.bindings.mask.texture = fixture.bindings.normalSource; }
    SUBCASE("Physical mask alias") { fixture.bindings.image = Native<VkImage>(10); }
    SUBCASE("Mask subresource") { fixture.bindings.mask.baseLayer = 1; }
    SUBCASE("Extent") { ++fixture.bindings.width; }
    SUBCASE("Invalid RG handle") { validDepth = false; }
    Memory::LinearAllocator scratch(64 * 1024);
    RG::RenderGraph graph(scratch);
    GraphTextureRef output;
    CHECK_FALSE(fixture.Build(graph, scratch, output, haveScene, validDepth).success);
    CHECK(graph.GetPasses().empty());
    CHECK_FALSE(output.handle.IsValid());
}

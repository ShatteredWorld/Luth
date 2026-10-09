#include <doctest/doctest.h>
#include "luth/renderer/features/rt/RestirDiFeature.h"
#include "luth/renderer/features/rt/RestirDiBindings.h"
#include "luth/renderer/features/RenderPipelineCompiler.h"
#include "luth/renderer/rendergraph/RenderGraph.h"
#include "luth/memory/LinearAllocator.h"

using namespace Luth;
namespace {
    template<class T> T Native(u64 n) { return reinterpret_cast<T>(static_cast<uintptr_t>(n)); }
    struct Fixture {
        RestirDiBindings native; RtSceneParameters params; PreparedRtScene scene;
        std::unique_ptr<CompiledRenderPipeline> pipeline;
        Fixture() {
            params.active[2] = true;
            native.pipelines.fill(Native<VkPipeline>(1)); native.layouts.fill(Native<VkPipelineLayout>(2)); native.sets.fill(Native<VkDescriptorSet>(3));
            for (u32 i = 0; i < 4; ++i) { native.sources[i] = Native<const Texture*>(10 + i);
                native.sourceImages[i] = Native<VkImage>(20 + i); native.sourceViews[i] = Native<VkImageView>(30 + i); }
            for (u32 i = 0; i < 2; ++i) { native.outputs[i] = {Native<const Texture*>(40 + i)};
                native.images[i] = Native<VkImage>(50 + i); native.imageViews[i] = Native<VkImageView>(60 + i); }
            native.width = native.fullWidth = 1279; native.height = native.fullHeight = 719;
            native.lights = {Native<VkBuffer>(72), 16, 256};
            native.scratch = {Native<VkBuffer>(70), 0, 1279ull * 719 * 32}; native.spatial = {Native<VkBuffer>(71), 0, native.scratch.size};
            native.tlas = Native<VkAccelerationStructureKHR>(80); native.view = {1}; native.generation = 3;
            native.frameIndex = (1ull << 32) + 42; scene.frameIndex = native.frameIndex; scene.emptyFallback = native.tlas;
            RenderPipelineDefinition definition; definition.AddFeature<RestirDiFeature>(); PipelineInputContract inputs;
            inputs.resources = {{RtSceneResources::Parameters}, {RestirDiResources::Bindings}, {RenderResources::SurfaceDepth},
                {RenderResources::Normal}, {RenderResources::MotionVectors}, {RenderResources::Roughness}, {RenderResources::LightData}, {RtSceneResources::Scene, ResourceOutputPresence::Optional}};
            inputs.capabilities = {&RtSceneResources::RayScene};
            auto compiled = RenderPipelineCompiler{}.Compile(std::move(definition), RendererCapabilities{{&RtSceneResources::AccelerationStructures, &RtSceneResources::RayQueries}}, inputs);
            REQUIRE(compiled.pipeline); pipeline = std::move(compiled.pipeline);
        }
        PipelineBuildResult Build(RG::RenderGraph& graph, Memory::LinearAllocator& scratch, GraphTextureRef& diffuse, GraphTextureRef& specular, bool haveScene = true) {
            const std::array formats{RG::TextureFormat::D32_Float, RG::TextureFormat::RG16_Float, RG::TextureFormat::RG16_Float, RG::TextureFormat::R8_Unorm};
            std::array<GraphTextureRef, 4> textures;
            for (u32 i = 0; i < 4; ++i) { RG::TextureDesc desc; desc.width = 1279; desc.height = 719; desc.format = formats[i];
                textures[i] = {graph.ImportResource(desc, (void*)Native<VkImage>(20 + i), (void*)Native<VkImageView>(30 + i), RG::ResourceState::ColorAttachment), {Native<const Texture*>(10 + i)}}; }
            Memory::GPUSubRegion lightSlice{Native<VkBuffer>(72), 16, 256};
            RG::BufferDesc lightDesc{"LightSSBO", 256, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT};
            const GraphBufferRef lights{graph.ImportBuffer(lightDesc, (void*)lightSlice.buffer, RG::ResourceState::StorageBufferRead),
                {&lightSlice, lightSlice.offset, lightSlice.size}};
            const RestirDiBindingRef ref{&native}; const RaySceneRef sceneRef{&scene};
            const std::array resources{RenderInputBinding::Present(RtSceneResources::Parameters, params), RenderInputBinding::Present(RestirDiResources::Bindings, ref), RenderInputBinding::Present(RenderResources::LightData, lights),
                RenderInputBinding::Present(RenderResources::SurfaceDepth, textures[0]), RenderInputBinding::Present(RenderResources::Normal, textures[1]),
                RenderInputBinding::Present(RenderResources::MotionVectors, textures[2]), RenderInputBinding::Present(RenderResources::Roughness, textures[3]),
                haveScene ? RenderInputBinding::Present(RtSceneResources::Scene, sceneRef) : RenderInputBinding::Absent(RtSceneResources::Scene)};
            FrameRenderInputs frame; frame.renderFrameIndex = (1ull << 32) + 42; frame.resources = resources; if (haveScene) frame.capabilities = RtSceneResources::Requests;
            ViewRenderInputs view; view.id = {1}; view.resourceGeneration = 3; view.width = 1279; view.height = 719;
            const std::array outputs{RenderOutputBinding::Capture(RestirDiResources::Diffuse, diffuse), RenderOutputBinding::Capture(RestirDiResources::Specular, specular)};
            return pipeline->Build(graph, frame, view, scratch, outputs);
        }
    };
}
TEST_CASE("RestirDiFeature: four async passes preserve reservoir topology at full and half resolution") {
    Fixture f;
    SUBCASE("Full") {}
    SUBCASE("Half") { f.native.settings.halfResolution = true; f.native.width = 639; f.native.height = 359; f.native.scratch.size = f.native.spatial.size = 639ull * 359 * 32; }
    Memory::LinearAllocator scratch(128 * 1024); RG::RenderGraph graph(scratch); GraphTextureRef diffuse, specular;
    REQUIRE(f.Build(graph, scratch, diffuse, specular).success); REQUIRE(graph.GetPasses().size() == 4); REQUIRE(graph.GetBuffers().size() == 3);
    const std::array names{"RestirInitial", "RestirTemporal", "RestirSpatial", "RestirShade"};
    for (u32 i = 0; i < 4; ++i) { CHECK(graph.GetPasses()[i].name == names[i]); CHECK(graph.GetPasses()[i].queueFamily == RG::QueueFamily::AsyncCompute); }
    CHECK(graph.GetPasses()[0].bufferWrites[0].index == graph.GetPasses()[1].bufferWrites[0].index);
    CHECK(graph.GetPasses()[1].bufferReads[1].index == graph.GetPasses()[2].bufferWrites[0].index);
    CHECK(graph.GetPasses()[2].bufferWrites[0].index == graph.GetPasses()[3].bufferReads[1].index);
    CHECK(diffuse.handle.index != specular.handle.index); CHECK(diffuse.binding.texture == f.native.outputs[0].texture); CHECK(specular.binding.texture == f.native.outputs[1].texture);
    REQUIRE(graph.GetResources().size() == 6); CHECK(graph.GetResources()[4].desc.width == f.native.width); CHECK(graph.GetResources()[5].desc.height == f.native.height);
    graph.Compile(); for (const auto& pass : graph.GetPasses()) CHECK_FALSE(pass.culled);
}
TEST_CASE("RestirDiFeature: disabled PT and cold shaders clear both outputs") {
    Fixture f; SUBCASE("Disabled") { f.params.active[2] = false; } SUBCASE("PT") { f.params.active[5] = true; } SUBCASE("Cold") { f.native.pipelines[1] = VK_NULL_HANDLE; }
    Memory::LinearAllocator scratch(128 * 1024); RG::RenderGraph graph(scratch); GraphTextureRef diffuse{{99, 0}, {Native<const Texture*>(100)}}, specular = diffuse;
    REQUIRE(f.Build(graph, scratch, diffuse, specular, false).success); CHECK(graph.GetPasses().empty()); CHECK(graph.GetBuffers().size() == 1);
    CHECK_FALSE(diffuse.handle.IsValid()); CHECK_FALSE(specular.handle.IsValid()); CHECK(diffuse.binding.texture == nullptr); CHECK(specular.binding.texture == nullptr);
}
TEST_CASE("RestirDiFeature: invalid scene view and physical bindings fail before registration") {
    Fixture f; bool haveScene = true;
    SUBCASE("Provider") { haveScene = false; } SUBCASE("Frame") { f.native.frameIndex = 42; } SUBCASE("Scene frame") { --f.scene.frameIndex; }
    SUBCASE("View") { f.native.view = {2}; } SUBCASE("Generation") { ++f.native.generation; }
    SUBCASE("TLAS") { f.native.tlas = Native<VkAccelerationStructureKHR>(99); } SUBCASE("Table") { f.native.geometryTable = 123; }
    SUBCASE("Readiness") { f.scene.tlas.result.instanceCount = 1; } SUBCASE("Descriptor") { f.native.sets[3] = VK_NULL_HANDLE; }
    SUBCASE("Texture") { f.native.sources[2] = Native<const Texture*>(99); } SUBCASE("Image") { f.native.sourceImages[0] = Native<VkImage>(99); }
    SUBCASE("Image view") { f.native.sourceViews[3] = Native<VkImageView>(99); } SUBCASE("Output alias") { f.native.images[0] = f.native.sourceImages[0]; }
    SUBCASE("Channels alias") { f.native.outputs[1] = f.native.outputs[0]; } SUBCASE("Extent") { ++f.native.width; }
    SUBCASE("Buffer alias") { f.native.spatial.buffer = f.native.scratch.buffer; } SUBCASE("Offset") { f.native.spatial.offset = 16; }
    SUBCASE("Light slice") { ++f.native.lights.offset; }
    SUBCASE("Size") { --f.native.spatial.size; } SUBCASE("Subresource") { f.native.outputs[0].baseMip = 1; }
    Memory::LinearAllocator scratch(128 * 1024); RG::RenderGraph graph(scratch); GraphTextureRef diffuse, specular;
    CHECK_FALSE(f.Build(graph, scratch, diffuse, specular, haveScene).success); CHECK(graph.GetPasses().empty()); CHECK(graph.GetBuffers().size() == 1);
    CHECK_FALSE(diffuse.handle.IsValid()); CHECK_FALSE(specular.handle.IsValid());
}
TEST_CASE("RestirDiFeature: scene demand follows DI activation and PT exclusion [renderfeatures]")
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
    definition.AddFeature<RestirDiDemandFeature>();
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
        params.active[static_cast<size_t>(RtSceneConsumer::DirectLighting)] = step != 0;
        params.active[static_cast<size_t>(RtSceneConsumer::PathTrace)] = step == 2;
        RG::RenderGraph graph(scratch);
        REQUIRE(compiled.pipeline->Build(graph, frame, {}, scratch).success);
        CHECK(prepares == (step == 0 ? 0 : 1));
        CHECK(registrations == prepares);
    }
    RenderPipelineDefinition missing;
    missing.AddFeature<RestirDiDemandFeature>();
    CHECK_FALSE(RenderPipelineCompiler{}.Compile(std::move(missing),
        RendererCapabilities{{&RtSceneResources::RayQueries}}, inputs).pipeline);
    RenderPipelineDefinition unsupported;
    unsupported.AddFeature<RestirDiFeature>();
    CHECK_FALSE(RenderPipelineCompiler{}.Compile(std::move(unsupported), {}, inputs).pipeline);
}

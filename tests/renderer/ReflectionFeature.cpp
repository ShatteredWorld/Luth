#include <doctest/doctest.h>
#include "luth/renderer/features/rt/ReflectionFeature.h"
#include "luth/renderer/features/rt/ReflectionBindings.h"
#include "luth/renderer/features/RenderPipelineCompiler.h"
#include "luth/renderer/rendergraph/RenderGraph.h"
#include "luth/memory/LinearAllocator.h"
#include <limits>

using namespace Luth;
namespace {
    template<class T> T Native(u64 n) { return reinterpret_cast<T>(static_cast<uintptr_t>(n)); }
    struct Fixture {
        ReflectionBindings native; RtSceneParameters params; PreparedRtScene scene;
        u32 guideWidth = 1279;
        RG::TextureFormat roughnessFormat = RG::TextureFormat::R8_Unorm;
        std::unique_ptr<CompiledRenderPipeline> pipeline;
        Fixture() {
            params.active[4] = true;
            native.pipeline = Native<VkPipeline>(1); native.layout = Native<VkPipelineLayout>(2); native.sets.fill(Native<VkDescriptorSet>(3));
            for (u32 i = 0; i < 3; ++i) { native.sources[i] = Native<const Texture*>(10 + i);
                native.sourceImages[i] = Native<VkImage>(20 + i); native.sourceViews[i] = Native<VkImageView>(30 + i); }
            native.output = {Native<const Texture*>(40)}; native.image = Native<VkImage>(50); native.imageView = Native<VkImageView>(60);
            native.width = native.fullWidth = 1279; native.height = native.fullHeight = 719;
            native.lights = {Native<VkBuffer>(72), 16, 256};
            native.tlas = Native<VkAccelerationStructureKHR>(80); native.view = {1}; native.generation = 3;
            native.frameIndex = (1ull << 32) + 42; scene.frameIndex = native.frameIndex; scene.emptyFallback = native.tlas;
            RenderPipelineDefinition definition; definition.AddFeature<ReflectionFeature>(); PipelineInputContract inputs;
            inputs.resources = {{RtSceneResources::Parameters}, {ReflectionResources::Bindings}, {RenderResources::SurfaceDepth},
                {RenderResources::Normal}, {RenderResources::Roughness}, {RenderResources::LightData}, {RtSceneResources::Scene, ResourceOutputPresence::Optional}};
            inputs.capabilities = {&RtSceneResources::RayScene};
            auto compiled = RenderPipelineCompiler{}.Compile(std::move(definition), RendererCapabilities{{&RtSceneResources::AccelerationStructures, &RtSceneResources::RayQueries}}, inputs);
            REQUIRE(compiled.pipeline); pipeline = std::move(compiled.pipeline);
        }
        PipelineBuildResult Build(RG::RenderGraph& graph, Memory::LinearAllocator& scratch, GraphTextureRef& radiance, bool haveScene = true) {
            const std::array formats{RG::TextureFormat::D32_Float, RG::TextureFormat::RG16_Float, roughnessFormat};
            std::array<GraphTextureRef, 3> textures;
            for (u32 i = 0; i < 3; ++i) { RG::TextureDesc desc; desc.width = guideWidth; desc.height = 719; desc.format = formats[i];
                textures[i] = {graph.ImportResource(desc, (void*)Native<VkImage>(20 + i), (void*)Native<VkImageView>(30 + i), RG::ResourceState::ColorAttachment), {Native<const Texture*>(10 + i)}}; }
            Memory::GPUSubRegion lightSlice{Native<VkBuffer>(72), 16, 256};
            RG::BufferDesc lightDesc{"LightSSBO", 256, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT};
            const GraphBufferRef lights{graph.ImportBuffer(lightDesc, (void*)lightSlice.buffer, RG::ResourceState::StorageBufferRead),
                {&lightSlice, lightSlice.offset, lightSlice.size}};
            const ReflectionBindingRef ref{&native}; const RaySceneRef sceneRef{&scene};
            const std::array resources{RenderInputBinding::Present(RtSceneResources::Parameters, params), RenderInputBinding::Present(ReflectionResources::Bindings, ref), RenderInputBinding::Present(RenderResources::LightData, lights),
                RenderInputBinding::Present(RenderResources::SurfaceDepth, textures[0]), RenderInputBinding::Present(RenderResources::Normal, textures[1]),
                RenderInputBinding::Present(RenderResources::Roughness, textures[2]),
                haveScene ? RenderInputBinding::Present(RtSceneResources::Scene, sceneRef) : RenderInputBinding::Absent(RtSceneResources::Scene)};
            FrameRenderInputs frame; frame.renderFrameIndex = (1ull << 32) + 42; frame.resources = resources; if (haveScene) frame.capabilities = RtSceneResources::Requests;
            ViewRenderInputs view; view.id = {1}; view.resourceGeneration = 3; view.width = 1279; view.height = 719;
            const std::array outputs{RenderOutputBinding::Capture(ReflectionResources::Radiance, radiance)};
            return pipeline->Build(graph, frame, view, scratch, outputs);
        }
    };
}

TEST_CASE("ReflectionFeature: single async trace forwards guides and tagged light slice at full and half resolution")
{
    Fixture f;
    SUBCASE("Full") {}
    SUBCASE("Half") { f.native.settings.halfResolution = true; f.native.width = 639; f.native.height = 359; }
    Memory::LinearAllocator scratch(128 * 1024); RG::RenderGraph graph(scratch); GraphTextureRef radiance;
    REQUIRE(f.Build(graph, scratch, radiance).success);
    REQUIRE(graph.GetPasses().size() == 1);
    const auto& pass = graph.GetPasses()[0];
    CHECK(pass.name == "RtReflections"); CHECK(pass.queueFamily == RG::QueueFamily::AsyncCompute);
    REQUIRE(pass.reads.size() == 3); REQUIRE(pass.bufferReads.size() == 1);
    for (u32 i = 0; i < 3; ++i) CHECK(pass.reads[i].index == i + 1);
    CHECK(pass.bufferReads[0].index == 1); CHECK(graph.GetBuffers().size() == 1);
    CHECK(radiance.binding.texture == f.native.output.texture);
    CHECK(radiance.handle.index == 4); REQUIRE(graph.GetResources().size() == 4);
    CHECK(graph.GetResources()[3].image == f.native.image);
    CHECK(graph.GetResources()[3].desc.width == f.native.width); CHECK(graph.GetResources()[3].desc.height == f.native.height);
    CHECK(graph.GetResources()[3].desc.format == RG::TextureFormat::RGBA16_Float);
    graph.Compile(); CHECK_FALSE(graph.GetPasses()[0].culled);
}
TEST_CASE("ReflectionFeature: disabled PT and cold trace publish absence without registration")
{
    Fixture f;
    SUBCASE("Disabled") { f.params.active[4] = false; }
    SUBCASE("PT") { f.params.active[5] = true; }
    SUBCASE("Cold") { f.native.pipeline = VK_NULL_HANDLE; }
    Memory::LinearAllocator scratch(128 * 1024); RG::RenderGraph graph(scratch);
    GraphTextureRef radiance{{99, 0}, {Native<const Texture*>(100)}};
    REQUIRE(f.Build(graph, scratch, radiance, false).success);
    CHECK(graph.GetPasses().empty()); CHECK(graph.GetResources().size() == 3);
    CHECK_FALSE(radiance.handle.IsValid()); CHECK(radiance.binding.texture == nullptr);
}
TEST_CASE("ReflectionFeature: stale scene guide output and light bindings fail before native registration")
{
    Fixture f; bool haveScene = true;
    SUBCASE("Provider") { haveScene = false; }
    SUBCASE("Frame") { f.native.frameIndex = 42; }
    SUBCASE("Scene frame") { --f.scene.frameIndex; }
    SUBCASE("View") { f.native.view = {2}; }
    SUBCASE("Generation") { ++f.native.generation; }
    SUBCASE("TLAS") { f.native.tlas = Native<VkAccelerationStructureKHR>(99); }
    SUBCASE("Table") { f.native.geometryTable = 123; }
    SUBCASE("Readiness") { f.scene.tlas.result.instanceCount = 1; }
    SUBCASE("Descriptor") { f.native.sets[3] = VK_NULL_HANDLE; }
    SUBCASE("Layout") { f.native.layout = VK_NULL_HANDLE; }
    SUBCASE("Settings") { f.native.settings.enabled = false; }
    SUBCASE("Texture") { f.native.sources[2] = Native<const Texture*>(99); }
    SUBCASE("Image") { f.native.sourceImages[0] = Native<VkImage>(99); }
    SUBCASE("Image view") { f.native.sourceViews[2] = Native<VkImageView>(99); }
    SUBCASE("Output image alias") { f.native.image = f.native.sourceImages[0]; }
    SUBCASE("Output texture alias") { f.native.output.texture = f.native.sources[0]; }
    SUBCASE("Guide alias") { f.native.sourceImages[1] = f.native.sourceImages[0]; }
    SUBCASE("Working extent") { ++f.native.width; }
    SUBCASE("Guide extent") { --f.guideWidth; }
    SUBCASE("Guide format") { f.roughnessFormat = RG::TextureFormat::RG16_Float; }
    SUBCASE("Light slice") { ++f.native.lights.offset; }
    SUBCASE("Light size") { --f.native.lights.size; }
    SUBCASE("Light buffer") { f.native.lights.buffer = Native<VkBuffer>(99); }
    SUBCASE("Subresource") { f.native.output.baseMip = 1; }
    SUBCASE("Null output") { f.native.output.texture = nullptr; }
    SUBCASE("Non-finite setting") { f.native.settings.maxRayDistance = std::numeric_limits<float>::quiet_NaN(); }
    Memory::LinearAllocator scratch(128 * 1024); RG::RenderGraph graph(scratch); GraphTextureRef radiance;
    CHECK_FALSE(f.Build(graph, scratch, radiance, haveScene).success);
    CHECK(graph.GetPasses().empty()); CHECK_FALSE(radiance.handle.IsValid());
}
TEST_CASE("ReflectionFeature: output is imported once and previously imported physical output is rejected")
{
    Fixture f; Memory::LinearAllocator scratch(128 * 1024); RG::RenderGraph graph(scratch); GraphTextureRef radiance;
    RG::TextureDesc desc; desc.width = f.native.width; desc.height = f.native.height; desc.format = RG::TextureFormat::RGBA16_Float;
    graph.ImportResource(desc, (void*)f.native.image, (void*)f.native.imageView, RG::ResourceState::Undefined);
    CHECK_FALSE(f.Build(graph, scratch, radiance).success); CHECK(graph.GetPasses().empty());
}
TEST_CASE("ReflectionFeature: recording captures retained state independently of the preparation packet")
{
    Fixture f; Memory::LinearAllocator scratch(128 * 1024);
    auto retained = std::make_shared<ReflectionViewState>(); std::weak_ptr<ReflectionViewState> weak = retained;
    f.native.retained = retained;
    {
        RG::RenderGraph graph(scratch); GraphTextureRef radiance;
        REQUIRE(f.Build(graph, scratch, radiance).success);
        f.native.retained.reset(); retained.reset(); f.native.pipeline = VK_NULL_HANDLE;
        CHECK_FALSE(weak.expired());
    }
    CHECK(weak.expired());
}
TEST_CASE("ReflectionFeature: explicit scene demand follows reflection activation and excludes PT")
{
    int prepares = 0, registrations = 0; RtSceneParameters params;
    auto packet = std::make_shared<PreparedRtScene>(); packet->frameIndex = 42;
    RenderPipelineDefinition definition;
    definition.AddFeature<RtSceneFeature>(
        [&](const FrameRenderInputs&, const RtSceneParameters&) -> std::shared_ptr<const PreparedRtScene> { ++prepares; return packet; },
        [&](RG::RenderGraph&, std::shared_ptr<const PreparedRtScene>) { ++registrations; });
    definition.AddFeature<ReflectionDemandFeature>();
    PipelineInputContract inputs; inputs.resources = {{RtSceneResources::Parameters}};
    const RendererCapabilities caps{{&RtSceneResources::AccelerationStructures, &RtSceneResources::RayQueries}};
    auto compiled = RenderPipelineCompiler{}.Compile(std::move(definition), caps, inputs); REQUIRE(compiled.pipeline);
    Memory::LinearAllocator scratch(64 * 1024);
    const std::array bindings{RenderInputBinding::Present(RtSceneResources::Parameters, params)};
    FrameRenderInputs frame; frame.renderFrameIndex = 42; frame.resources = bindings;
    for (int step = 0; step < 3; ++step) {
        params.active[4] = step != 0; params.active[5] = step == 2;
        RG::RenderGraph graph(scratch); REQUIRE(compiled.pipeline->Build(graph, frame, {}, scratch).success);
        CHECK(prepares == (step == 0 ? 0 : 1)); CHECK(registrations == prepares);
    }
    RenderPipelineDefinition missing; missing.AddFeature<ReflectionDemandFeature>();
    CHECK_FALSE(RenderPipelineCompiler{}.Compile(std::move(missing), caps, inputs).pipeline);
    Fixture f;
    PipelineInputContract traceInputs;
    traceInputs.resources = {{RtSceneResources::Parameters}, {ReflectionResources::Bindings},
        {RenderResources::SurfaceDepth}, {RenderResources::Normal}, {RenderResources::Roughness},
        {RenderResources::LightData}, {RtSceneResources::Scene, ResourceOutputPresence::Optional}};
    traceInputs.capabilities = {&RtSceneResources::RayScene};
    SUBCASE("Unsupported") { RenderPipelineDefinition d; d.AddFeature<ReflectionFeature>();
        CHECK_FALSE(RenderPipelineCompiler{}.Compile(std::move(d), {}, traceInputs).pipeline); }
    SUBCASE("Missing guide") { traceInputs.resources.erase(traceInputs.resources.begin() + 4);
        RenderPipelineDefinition d; d.AddFeature<ReflectionFeature>();
        CHECK_FALSE(RenderPipelineCompiler{}.Compile(std::move(d), caps, traceInputs).pipeline); }
    SUBCASE("Duplicate producer") { RenderPipelineDefinition d; d.AddFeature<ReflectionFeature>(); d.AddFeature<ReflectionFeature>();
        CHECK_FALSE(RenderPipelineCompiler{}.Compile(std::move(d), caps, traceInputs).pipeline); }
}

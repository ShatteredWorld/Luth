#include <doctest/doctest.h>
#include "luth/core/diagnostics/Log.h"
#include "luth/renderer/features/FogComputeFeature.h"
#include "luth/renderer/features/rt/RtFogFeature.h"
#include "luth/renderer/features/rt/RtSunShadowFeature.h"
#include "luth/renderer/features/RenderPipelineCompiler.h"
#include "luth/renderer/subsystems/VolumetricSubsystem.h"
#include "luth/renderer/subsystems/LightingSubsystem.h"
#include "luth/renderer/lighting/FogVolumeGatherer.h"
#include "luth/renderer/CameraParams.h"
#include "luth/memory/LinearAllocator.h"

using namespace Luth;
namespace
{
    template<class T> T Native(u64 value) { return reinterpret_cast<T>(static_cast<uintptr_t>(value)); }
    struct Fixture
    {
        VolumetricSubsystem native;
        FogComputeBindings bindings;
        CameraParams camera;
        std::unique_ptr<CompiledRenderPipeline> pipeline;
        GraphTextureRef density, integrated, resolved;
        RtSceneParameters params;
        PreparedRtScene scene;
        bool rt = false, scenePresent = true, capabilityPresent = true;
        Fixture(bool useRt = false) : rt(useRt)
        {
            bindings.enabled = bindings.ready = true;
            bindings.global = Native<VkDescriptorSet>(1); bindings.material = Native<VkDescriptorSet>(2);
            bindings.bindless = Native<VkDescriptorSet>(3);
            for (u32 i = 0; i < 4; ++i)
            {
                bindings.pipelines[i] = Native<VkPipeline>(10 + i);
                bindings.layouts[i] = Native<VkPipelineLayout>(20 + i);
                bindings.sets[i] = Native<VkDescriptorSet>(30 + i);
            }
            auto image = [](u64 identity) { return FogImageBinding{Native<VkImage>(identity),
                Native<VkImageView>(identity + 1), {Native<const Texture*>(identity + 2)}}; };
            bindings.density = image(100); bindings.scratch = image(200);
            bindings.previous = image(300); bindings.current = image(400);
            bindings.inject.volDimX = bindings.integrate.volDimX = bindings.resolve.volDimX = 80;
            bindings.inject.volDimY = bindings.integrate.volDimY = bindings.resolve.volDimY = 45;
            bindings.inject.volDimZ = bindings.integrate.volDimZ = bindings.resolve.volDimZ = 64;
            bindings.volumes = {Native<VkBuffer>(1), 1024, sizeof(FogVolumeSSBOHeader)};
            bindings.lights = {Native<VkBuffer>(1), 2048, sizeof(LightSSBOHeader)};
            bindings.grid = {Native<VkBuffer>(1), 4096, u64(k_ClusterCount) * sizeof(GPUCluster)};
            bindings.indices = {Native<VkBuffer>(1), 1024 * 1024, u64(k_ClusterCount) * k_MaxLightsPerCluster * sizeof(u32)};
            params.active[static_cast<size_t>(RtSceneConsumer::Fog)] = true;
            scene.frameIndex = bindings.frameIndex = 42;
            bindings.view = {1}; bindings.generation = 3;
            if (rt) {
                bindings.rtShadows = true;
                scene.emptyFallback = bindings.tlas = Native<VkAccelerationStructureKHR>(99);
            }
            RenderPipelineDefinition definition;
            if (rt) definition.AddFeature<RtFogFeature>(native);
            else definition.AddFeature<FogComputeFeature>(native);
            PipelineInputContract inputs;
            inputs.resources = {{FogResources::Bindings},
                {FogResources::Volumes, ResourceOutputPresence::Optional},
                {RenderResources::LightData, ResourceOutputPresence::Optional},
                {RenderResources::ClusterGrid, ResourceOutputPresence::Optional},
                {RenderResources::LightIndices, ResourceOutputPresence::Optional},
                {RenderResources::ShadowCascades, ResourceOutputPresence::Optional}};
            RendererCapabilities capabilities;
            if (rt) {
                inputs.resources.push_back({RtSceneResources::Parameters});
                inputs.resources.push_back({RtSceneResources::Scene, ResourceOutputPresence::Optional});
                inputs.capabilities = {&RtSceneResources::RayScene};
                capabilities.supported = {&RtSceneResources::AccelerationStructures, &RtSceneResources::RayQueries};
            }
            auto compiled = RenderPipelineCompiler{}.Compile(std::move(definition), capabilities, inputs);
            REQUIRE(compiled.pipeline);
            pipeline = std::move(compiled.pipeline);
        }
        PipelineBuildResult Build(RG::RenderGraph& graph, Memory::LinearAllocator& scratch,
            bool volumePresent = true, bool shadowsPresent = false, bool wrongSlice = false, u64 viewId = 1)
        {
            auto volumes = LightingSubsystem::ImportLightingBuffer(graph, "Volumes", bindings.volumes);
            const auto lights = LightingSubsystem::ImportLightingBuffer(graph, "Lights", bindings.lights);
            const auto grid = LightingSubsystem::ImportLightingBuffer(graph, "Grid", bindings.grid);
            const auto indices = LightingSubsystem::ImportLightingBuffer(graph, "Indices", bindings.indices);
            if (wrongSlice) ++volumes.binding.offset;
            ShadowCascadeRefs shadows;
            if (shadowsPresent)
                for (u32 i = 0; i < 4; ++i)
                {
                    shadows.cascades[i].binding = {Native<const Texture*>(900), 0, 1, i, 1};
                    RG::TextureDesc desc; desc.name = "Cascade"; desc.width = desc.height = 16;
                    desc.format = RG::TextureFormat::D32_Float;
                    shadows.cascades[i].handle = graph.ImportResource(desc, (void*)(uintptr_t(500 + i)),
                        (void*)(uintptr_t(600 + i)), RG::ResourceState::DepthStencilAttachment);
                }
            const FogComputeBindingRef ref{&bindings};
            std::vector resources{RenderInputBinding::Present(FogResources::Bindings, ref),
                volumePresent ? RenderInputBinding::Present(FogResources::Volumes, volumes) : RenderInputBinding::Absent(FogResources::Volumes),
                RenderInputBinding::Present(RenderResources::LightData, lights),
                RenderInputBinding::Present(RenderResources::ClusterGrid, grid),
                RenderInputBinding::Present(RenderResources::LightIndices, indices),
                shadowsPresent ? RenderInputBinding::Present(RenderResources::ShadowCascades, shadows) : RenderInputBinding::Absent(RenderResources::ShadowCascades)};
            const RaySceneRef sceneRef{&scene};
            if (rt) {
                resources.push_back(RenderInputBinding::Present(RtSceneResources::Parameters, params));
                resources.push_back(scenePresent ? RenderInputBinding::Present(RtSceneResources::Scene, sceneRef)
                    : RenderInputBinding::Absent(RtSceneResources::Scene));
            }
            FrameRenderInputs frame; frame.resources = resources; frame.renderFrameIndex = 42;
            if (rt && capabilityPresent) frame.capabilities = RtSceneResources::Requests;
            ViewRenderInputs view; view.id = {viewId}; view.width = 640; view.height = 480; view.camera = &camera;
            view.resourceGeneration = 3;
            const std::array outputs{RenderOutputBinding::Capture(RenderResources::FogDensity, density),
                RenderOutputBinding::Capture(FogResources::IntegratedScatter, integrated),
                RenderOutputBinding::Capture(RenderResources::ResolvedFog, resolved)};
            return pipeline->Build(graph, frame, view, scratch, outputs);
        }
    };
}

TEST_CASE("FogComputeFeature: four native async passes reuse producer nodes [renderfeatures]")
{
    Fixture fixture;
    Memory::LinearAllocator scratch(64 * 1024); RG::RenderGraph graph(scratch);
    REQUIRE(fixture.Build(graph, scratch).success);
    REQUIRE(graph.GetPasses().size() == 4);
    CHECK(graph.GetResources().size() == 3);
    CHECK(graph.GetBuffers().size() == 4); // Tagged slices share a backing, retain distinct identity.
    const char* names[] = {"VolumetricInjectDensity", "VolumetricInjectScatter", "VolumetricIntegrate", "VolumetricResolve"};
    for (u32 i = 0; i < 4; ++i)
    {
        CHECK(graph.GetPasses()[i].name == names[i]);
        CHECK(graph.GetPasses()[i].isCompute);
        CHECK(graph.GetPasses()[i].queueFamily == RG::QueueFamily::AsyncCompute);
    }
    const auto& passes = graph.GetPasses();
    CHECK(passes[0].bufferReads.size() == 1);
    CHECK(passes[1].bufferReads.size() == 3);
    CHECK(passes[1].reads[0] == fixture.density.handle);
    CHECK(passes[2].reads[0] == fixture.density.handle);
    CHECK(passes[1].writes[0].index == fixture.integrated.handle.index);
    CHECK(passes[2].writes[0] == fixture.integrated.handle);
    CHECK(passes[3].reads[0] == fixture.integrated.handle);
    CHECK(passes[3].writes[0] == fixture.resolved.handle);
    CHECK(fixture.integrated.handle.version == 2);
    CHECK(fixture.resolved.binding.texture == fixture.bindings.current.binding.texture);
    graph.Compile();
    for (const auto& pass : graph.GetPasses()) CHECK_FALSE(pass.culled);
}

TEST_CASE("FogComputeFeature: disable and cold resources omit registration and clear outputs [renderfeatures]")
{
    Fixture fixture;
    { Memory::LinearAllocator scratch(64 * 1024); RG::RenderGraph graph(scratch);
      REQUIRE(fixture.Build(graph, scratch).success); }
    SUBCASE("disabled") { fixture.bindings.enabled = false; }
    SUBCASE("cold") { fixture.bindings.ready = false; }
    Memory::LinearAllocator scratch(64 * 1024); RG::RenderGraph graph(scratch);
    REQUIRE(fixture.Build(graph, scratch).success);
    CHECK(graph.GetPasses().empty());
    CHECK_FALSE(fixture.density.handle.IsValid());
    CHECK_FALSE(fixture.integrated.handle.IsValid());
    CHECK_FALSE(fixture.resolved.handle.IsValid());
}

TEST_CASE("FogComputeFeature: rejects missing or inconsistent preparation before registration [renderfeatures]")
{
    Fixture fixture;
    bool volumes = true, wrongSlice = false;
    SUBCASE("absent volumes") { volumes = false; }
    SUBCASE("slice offset") { wrongSlice = true; }
    SUBCASE("missing native set") { fixture.bindings.sets[1] = VK_NULL_HANDLE; }
    SUBCASE("inconsistent dimensions") { ++fixture.bindings.resolve.volDimZ; }
    SUBCASE("aliased history") { fixture.bindings.previous = fixture.bindings.current; }
    SUBCASE("missing RT provider") { fixture.bindings.rtShadows = true; }
    SUBCASE("RT handle in raster packet") { fixture.bindings.tlas = Native<VkAccelerationStructureKHR>(99); }
    SUBCASE("RT table in raster packet") { fixture.bindings.inject.geomTableBDA = 99; }
    SUBCASE("short cluster grid") { fixture.bindings.grid.size = 1; }
    Memory::LinearAllocator scratch(64 * 1024); RG::RenderGraph graph(scratch);
    CHECK_FALSE(fixture.Build(graph, scratch, volumes, false, wrongSlice).success);
    CHECK(graph.GetPasses().empty());
    CHECK_FALSE(fixture.resolved.handle.IsValid());
}

TEST_CASE("FogComputeFeature: consumes optional cascade handles without importing aliases [renderfeatures]")
{
    Fixture fixture;
    Memory::LinearAllocator scratch(64 * 1024); RG::RenderGraph graph(scratch);
    REQUIRE(fixture.Build(graph, scratch, true, true).success);
    CHECK(graph.GetResources().size() == 7);
    CHECK(graph.GetPasses()[1].reads.size() == 5);
    for (u32 i = 0; i < 4; ++i) CHECK(graph.GetPasses()[1].reads[i + 1].index == i + 1);
    graph.Compile();
    CHECK_FALSE(graph.GetPasses()[1].preBarriers.empty());
}

TEST_CASE("FogComputeFeature: native disabled preparation needs no device [renderfeatures]")
{
    VolumetricSubsystem native; FogViewState state; CameraParams camera;
    const u64 frame = (u64{1} << 32) + 17;
    const auto bindings = native.PrepareComputeBindings(state, frame, {1}, 1, camera, VK_NULL_HANDLE, false, false,
        nullptr, {}, {}, {}, {});
    CHECK_FALSE(bindings.enabled); CHECK_FALSE(bindings.ready);
    CHECK(bindings.frameIndex == frame); CHECK(bindings.view == RenderViewId{1}); CHECK(bindings.generation == 1);
}

TEST_CASE("RtFogFeature: empty fallback and populated scene preserve the four-pass chain [renderfeatures]")
{
    Fixture fixture(true);
    SUBCASE("populated") {
        fixture.scene.tlas.result.instanceCount = 1;
        fixture.scene.tlas.result.tlas = fixture.bindings.tlas;
        fixture.scene.tlas.result.geomTableBDA = fixture.bindings.inject.geomTableBDA = 12345;
    }
    Memory::LinearAllocator scratch(64 * 1024); RG::RenderGraph graph(scratch);
    REQUIRE(fixture.Build(graph, scratch, true, true).success);
    REQUIRE(graph.GetPasses().size() == 4);
    CHECK(graph.GetPasses()[1].name == "VolumetricInjectScatter");
    CHECK(graph.GetPasses()[1].queueFamily == RG::QueueFamily::AsyncCompute);
    CHECK(graph.GetPasses()[1].reads.size() == 5);
    CHECK(graph.GetResources().size() == 7); // Existing cascades and three fog imports only.
    CHECK(fixture.resolved.binding.texture == fixture.bindings.current.binding.texture);
    graph.Compile();
    for (const auto& pass : graph.GetPasses()) CHECK_FALSE(pass.culled);
}
TEST_CASE("RtFogFeature: stale or unpaired bindings fail before any fog registration [renderfeatures]")
{
    Fixture fixture(true);
    { Memory::LinearAllocator scratch(64 * 1024); RG::RenderGraph graph(scratch);
      REQUIRE(fixture.Build(graph, scratch).success); }
    SUBCASE("missing provider output") { fixture.scenePresent = false; }
    SUBCASE("missing capability") { fixture.capabilityPresent = false; }
    SUBCASE("stale scene frame") { ++fixture.scene.frameIndex; }
    SUBCASE("stale packet frame") { ++fixture.bindings.frameIndex; }
    SUBCASE("other view") { fixture.bindings.view = {2}; }
    SUBCASE("replaced view state") { ++fixture.bindings.generation; }
    SUBCASE("different TLAS") { fixture.bindings.tlas = Native<VkAccelerationStructureKHR>(100); }
    SUBCASE("different table") { fixture.bindings.inject.geomTableBDA = 12345; }
    SUBCASE("incomplete populated scene") { fixture.scene.tlas.result.instanceCount = 1; }
    SUBCASE("missing RT variant") { fixture.bindings.rtShadows = false; }
    Memory::LinearAllocator scratch(64 * 1024); RG::RenderGraph graph(scratch);
    CHECK_FALSE(fixture.Build(graph, scratch).success);
    CHECK(graph.GetPasses().empty());
    CHECK_FALSE(fixture.density.handle.IsValid()); CHECK_FALSE(fixture.resolved.handle.IsValid());
}
TEST_CASE("RtFogFeature: runtime disable PT and cold pipeline publish absence [renderfeatures]")
{
    Fixture fixture(true);
    SUBCASE("fog off") { fixture.params.active[static_cast<size_t>(RtSceneConsumer::Fog)] = false; }
    SUBCASE("PT selected") { fixture.params.active[static_cast<size_t>(RtSceneConsumer::PathTrace)] = true; }
    SUBCASE("native disabled") { fixture.bindings.enabled = false; }
    SUBCASE("cold") { fixture.bindings.ready = false; }
    Memory::LinearAllocator scratch(64 * 1024); RG::RenderGraph graph(scratch);
    REQUIRE(fixture.Build(graph, scratch).success);
    CHECK(graph.GetPasses().empty()); CHECK_FALSE(fixture.resolved.handle.IsValid());
}
TEST_CASE("RtFogFeature: demand is independent of surface shadows and excluded by PT [renderfeatures]")
{
    int prepares = 0, registrations = 0;
    RtSceneParameters params;
    auto packet = std::make_shared<PreparedRtScene>(); packet->frameIndex = 42;
    RenderPipelineDefinition definition;
    definition.AddFeature<RtSceneFeature>(
        [&](const FrameRenderInputs&, const RtSceneParameters&) -> std::shared_ptr<const PreparedRtScene> {
            ++prepares; return packet;
        }, [&](RG::RenderGraph&, std::shared_ptr<const PreparedRtScene>) { ++registrations; });
    definition.AddFeature<RtFogDemandFeature>();
    definition.AddFeature<RtSunShadowDemandFeature>();
    PipelineInputContract inputs; inputs.resources = {{RtSceneResources::Parameters}};
    const RendererCapabilities capabilities{{&RtSceneResources::AccelerationStructures, &RtSceneResources::RayQueries}};
    auto compiled = RenderPipelineCompiler{}.Compile(std::move(definition), capabilities, inputs);
    REQUIRE(compiled.pipeline);
    Memory::LinearAllocator scratch(64 * 1024);
    const std::array resources{RenderInputBinding::Present(RtSceneResources::Parameters, params)};
    FrameRenderInputs frame; frame.renderFrameIndex = 42; frame.resources = resources;
    for (int step = 0; step < 4; ++step) {
        params.active[static_cast<size_t>(RtSceneConsumer::Fog)] = step != 0;
        params.active[static_cast<size_t>(RtSceneConsumer::PathTrace)] = step == 2;
        params.active[static_cast<size_t>(RtSceneConsumer::SunShadow)] = step == 3;
        RG::RenderGraph graph(scratch);
        REQUIRE(compiled.pipeline->Build(graph, frame, {}, scratch).success);
        CHECK(prepares == (step == 3 ? 2 : step ? 1 : 0)); CHECK(registrations == prepares);
    }
    VolumetricSubsystem native;
    for (bool unsupported : {false, true}) {
        RenderPipelineDefinition invalid;
        if (unsupported) invalid.AddFeature<RtSceneFeature>(
            [&](const FrameRenderInputs&, const RtSceneParameters&) -> std::shared_ptr<const PreparedRtScene> { return packet; },
            [](RG::RenderGraph&, std::shared_ptr<const PreparedRtScene>) {});
        invalid.AddFeature<RtFogDemandFeature>();
        CHECK_FALSE(RenderPipelineCompiler{}.Compile(std::move(invalid),
            unsupported ? RendererCapabilities{} : capabilities, inputs).pipeline);
    }
    // The raster feature has no scene contract or device requirements.
    const auto raster = FogComputeFeature(native).Describe();
    CHECK(raster.capabilities.consumes.empty()); CHECK(raster.capabilities.deviceRequirements.empty());
}

TEST_CASE("FogComputeFeature: two views and history parity export distinct physical references [renderfeatures]")
{
    Fixture fixture;
    Memory::LinearAllocator firstScratch(64 * 1024); RG::RenderGraph first(firstScratch);
    REQUIRE(fixture.Build(first, firstScratch, true, false, false, 1).success);
    const auto firstTexture = fixture.resolved.binding.texture;
    std::swap(fixture.bindings.current, fixture.bindings.previous);
    fixture.bindings.currentHistoryA = true;
    Memory::LinearAllocator secondScratch(64 * 1024); RG::RenderGraph second(secondScratch);
    REQUIRE(fixture.Build(second, secondScratch, true, false, false, 2).success);
    CHECK(fixture.resolved.binding.texture != firstTexture);
    CHECK(first.GetResources()[2].desc.name == "VolInScatterHistB[curr]");
    CHECK(second.GetResources()[2].desc.name == "VolInScatterHistA[curr]");
}

#include <doctest/doctest.h>
#include "luth/core/diagnostics/Log.h"
#include "luth/renderer/features/FogComputeFeature.h"
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
        Fixture()
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
            RenderPipelineDefinition definition;
            definition.AddFeature<FogComputeFeature>(native);
            PipelineInputContract inputs;
            inputs.resources = {{FogResources::Bindings},
                {FogResources::Volumes, ResourceOutputPresence::Optional},
                {RenderResources::LightData, ResourceOutputPresence::Optional},
                {RenderResources::ClusterGrid, ResourceOutputPresence::Optional},
                {RenderResources::LightIndices, ResourceOutputPresence::Optional},
                {RenderResources::ShadowCascades, ResourceOutputPresence::Optional}};
            auto compiled = RenderPipelineCompiler{}.Compile(std::move(definition), {}, inputs);
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
            const std::array resources{RenderInputBinding::Present(FogResources::Bindings, ref),
                volumePresent ? RenderInputBinding::Present(FogResources::Volumes, volumes) : RenderInputBinding::Absent(FogResources::Volumes),
                RenderInputBinding::Present(RenderResources::LightData, lights),
                RenderInputBinding::Present(RenderResources::ClusterGrid, grid),
                RenderInputBinding::Present(RenderResources::LightIndices, indices),
                shadowsPresent ? RenderInputBinding::Present(RenderResources::ShadowCascades, shadows) : RenderInputBinding::Absent(RenderResources::ShadowCascades)};
            FrameRenderInputs frame; frame.resources = resources;
            ViewRenderInputs view; view.id = {viewId}; view.width = 640; view.height = 480; view.camera = &camera;
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
    const auto bindings = native.PrepareComputeBindings(state, 17, camera, VK_NULL_HANDLE, false, false,
        nullptr, {}, {}, {}, {});
    CHECK_FALSE(bindings.enabled); CHECK_FALSE(bindings.ready);
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

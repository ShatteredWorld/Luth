#include <doctest/doctest.h>
#include "luth/core/diagnostics/Log.h"
#include "luth/renderer/features/ClusteredLightingFeature.h"
#include "luth/renderer/features/RenderPipelineCompiler.h"
#include "luth/renderer/subsystems/LightingSubsystem.h"
#include "luth/memory/LinearAllocator.h"

using namespace Luth;
namespace
{
    template<class T> T Native(u64 value) { return reinterpret_cast<T>(static_cast<uintptr_t>(value)); }
    struct Fixture
    {
        LightingSubsystem native;
        ClusterBindings bindings;
        std::unique_ptr<CompiledRenderPipeline> pipeline;
        GraphBufferRef lights, grid, indices;
        Fixture()
        {
            bindings.build = Native<VkPipeline>(2); bindings.assign = Native<VkPipeline>(3);
            bindings.buildLayout = Native<VkPipelineLayout>(4); bindings.assignLayout = Native<VkPipelineLayout>(5);
            bindings.buildSet = Native<VkDescriptorSet>(6); bindings.assignSet = Native<VkDescriptorSet>(7);
            bindings.lights = {Native<VkBuffer>(1), 4096, sizeof(LightSSBOHeader)};
            bindings.aabb = {Native<VkBuffer>(1), 8192, u64(k_ClusterCount) * 32};
            bindings.grid = {Native<VkBuffer>(1), 1024 * 1024, u64(k_ClusterCount) * sizeof(GPUCluster)};
            bindings.indices = {Native<VkBuffer>(1), 2 * 1024 * 1024, u64(k_ClusterCount) * k_MaxLightsPerCluster * sizeof(u32)};
            bindings.counter = {Native<VkBuffer>(1), 8 * 1024 * 1024, 16};
            bindings.buildConstants.viewportSize = Vec2(640, 480);
            bindings.ready = true;
            RenderPipelineDefinition definition;
            definition.AddFeature<ClusteredLightingFeature>(native);
            PipelineInputContract inputs;
            inputs.resources = {{ClusterResources::Bindings},
                {ClusterResources::UploadedLights, ResourceOutputPresence::Optional}};
            auto compiled = RenderPipelineCompiler{}.Compile(std::move(definition), {}, inputs);
            REQUIRE(compiled.pipeline);
            pipeline = std::move(compiled.pipeline);
        }
        PipelineBuildResult Build(RG::RenderGraph& graph, Memory::LinearAllocator& scratch,
            bool lightPresent = true, bool packetPresent = true, u64 viewId = 1)
        {
            const auto uploaded = native.ImportLightingBuffer(graph, "LightSSBO", bindings.lights);
            const ClusterBindingRef binding{packetPresent ? &bindings : nullptr};
            const std::array resources{RenderInputBinding::Present(ClusterResources::Bindings, binding),
                lightPresent ? RenderInputBinding::Present(ClusterResources::UploadedLights, uploaded)
                    : RenderInputBinding::Absent(ClusterResources::UploadedLights)};
            FrameRenderInputs frame; frame.resources = resources;
            ViewRenderInputs view; view.id = {viewId}; view.width = 640; view.height = 480;
            const std::array exports{RenderOutputBinding::Capture(RenderResources::LightData, lights),
                RenderOutputBinding::Capture(RenderResources::ClusterGrid, grid),
                RenderOutputBinding::Capture(RenderResources::LightIndices, indices)};
            return pipeline->Build(graph, frame, view, scratch, exports);
        }
    };
}

TEST_CASE("ClusteredLightingFeature: native async passes share producer handles and slice identity [renderfeatures]")
{
    Fixture fixture;
    Memory::LinearAllocator scratch(64 * 1024); RG::RenderGraph graph(scratch);
    REQUIRE(fixture.Build(graph, scratch).success);
    REQUIRE(graph.GetPasses().size() == 2);
    REQUIRE(graph.GetBuffers().size() == 5); // Same backing buffer, five distinct logical slices.
    const auto& build = graph.GetPasses()[0];
    const auto& assign = graph.GetPasses()[1];
    CHECK(build.name == "ClusterBuild"); CHECK(assign.name == "LightAssign");
    for (const auto& pass : graph.GetPasses())
    {
        CHECK(pass.isCompute); CHECK(pass.queueFamily == RG::QueueFamily::AsyncCompute);
    }
    REQUIRE(build.bufferWrites.size() == 2);
    REQUIRE(assign.bufferReads.size() == 2);
    REQUIRE(assign.bufferWrites.size() == 3);
    CHECK(assign.bufferReads[0] == fixture.lights.handle);
    CHECK(assign.bufferReads[1] == build.bufferWrites[0]);
    CHECK(assign.bufferWrites[0].index == build.bufferWrites[1].index);
    CHECK(fixture.grid.handle == assign.bufferWrites[0]);
    CHECK(fixture.indices.handle == assign.bufferWrites[1]);
    CHECK(fixture.grid.handle.version == 2); CHECK(fixture.indices.handle.version == 1);
    CHECK(fixture.grid.binding.offset == fixture.bindings.grid.offset);
    CHECK(fixture.indices.binding.offset == fixture.bindings.indices.offset);
    CHECK(fixture.grid.binding.slice != &fixture.bindings.grid); // Arena copy survives native packet scope.
    CHECK(fixture.grid.binding.slice->buffer == fixture.bindings.lights.buffer);
    for (auto state : assign.bufferReadStates) CHECK(state == RG::ResourceState::StorageBufferRead);
    graph.Compile();
    for (const auto& pass : graph.GetPasses()) CHECK_FALSE(pass.culled);
}
TEST_CASE("ClusteredLightingFeature: cold native resources omit passes and reset cluster outputs [renderfeatures]")
{
    Fixture fixture;
    { Memory::LinearAllocator scratch(64 * 1024); RG::RenderGraph graph(scratch);
      REQUIRE(fixture.Build(graph, scratch).success); }
    fixture.bindings.ready = false;
    Memory::LinearAllocator scratch(64 * 1024); RG::RenderGraph graph(scratch);
    REQUIRE(fixture.Build(graph, scratch).success);
    CHECK(graph.GetPasses().empty()); CHECK(graph.GetBuffers().size() == 1);
    CHECK_FALSE(fixture.grid.handle.IsValid()); CHECK_FALSE(fixture.indices.handle.IsValid());
    CHECK(fixture.lights.handle.IsValid()); // Uploaded lights pass through, independently of clusters.
}
TEST_CASE("ClusteredLightingFeature: incomplete prepared inputs fail before registering passes [renderfeatures]")
{
    Fixture fixture;
    bool lights = true, packet = true;
    SUBCASE("lights absent") { lights = false; }
    SUBCASE("native packet absent") { packet = false; }
    SUBCASE("descriptor absent") { fixture.bindings.assignSet = VK_NULL_HANDLE; }
    SUBCASE("pipeline absent") { fixture.bindings.build = VK_NULL_HANDLE; }
    SUBCASE("view extent mismatch") { fixture.bindings.buildConstants.viewportSize.x = 641; }
    SUBCASE("grid undersized") { fixture.bindings.grid.size = 16; }
    SUBCASE("index undersized") { fixture.bindings.indices.size = 16; }
    SUBCASE("counter undersized") { fixture.bindings.counter.size = 0; }
    SUBCASE("light count outside slice") { fixture.bindings.assignConstants.pointLightCount = 1; }
    Memory::LinearAllocator scratch(64 * 1024); RG::RenderGraph graph(scratch);
    CHECK_FALSE(fixture.Build(graph, scratch, lights, packet).success);
    CHECK(graph.GetPasses().empty()); CHECK(graph.GetBuffers().size() == 1);
    CHECK_FALSE(fixture.lights.handle.IsValid()); CHECK_FALSE(fixture.grid.handle.IsValid());
    CHECK_FALSE(fixture.indices.handle.IsValid());
}
TEST_CASE("ClusteredLightingFeature: native camera constants and descriptor slots freeze before recording [renderfeatures]")
{
    LightingSubsystem native;
    CameraParams camera;
    camera.projection = Mat4(1); camera.view = Mat4(2); camera.nearZ = 0.2f; camera.farZ = 700;
    const auto packet = native.PrepareClusterBindings(17, Native<VkDescriptorSet>(1), Native<VkDescriptorSet>(2),
        camera, 123, 456, {}, 9, 11);
    camera.view = Mat4(3); camera.projection = Mat4(4);
    CHECK(packet.buildSet == Native<VkDescriptorSet>(1)); CHECK(packet.assignSet == Native<VkDescriptorSet>(2));
    CHECK(packet.buildConstants.invProjection[1][1] == -1);
    CHECK(packet.buildConstants.viewportSize == Vec2(123, 456));
    CHECK(packet.buildConstants.nearZ == 0.2f); CHECK(packet.buildConstants.farZ == 700);
    CHECK(packet.assignConstants.view == Mat4(2));
    CHECK(packet.assignConstants.pointLightCount == 9); CHECK(packet.assignConstants.spotLightCount == 11);
    CHECK(sizeof(ClusterBuildConstants) == 96); CHECK(sizeof(LightAssignConstants) == 80);
    CHECK_FALSE(packet.ready);
}
TEST_CASE("ClusteredLightingFeature: sequential views keep output slices and graphs independent [renderfeatures]")
{
    Fixture fixture;
    Memory::LinearAllocator scratchA(64 * 1024), scratchB(64 * 1024);
    RG::RenderGraph graphA(scratchA), graphB(scratchB);
    REQUIRE(fixture.Build(graphA, scratchA).success);
    const auto first = fixture.grid;
    fixture.bindings.grid.offset += 8192;
    REQUIRE(fixture.Build(graphB, scratchB, true, true, 2).success);
    CHECK(first.binding.offset + 8192 == fixture.grid.binding.offset);
    CHECK(first.binding.slice != fixture.grid.binding.slice);
    CHECK(first.binding.slice->offset == first.binding.offset);
    CHECK(graphA.GetPasses().size() == 2); CHECK(graphB.GetPasses().size() == 2);
}

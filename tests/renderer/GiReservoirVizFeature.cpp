#include <doctest/doctest.h>
#include "luth/core/diagnostics/Log.h"
#include "luth/renderer/features/rt/GiReservoirVizFeature.h"
#include "luth/renderer/features/RenderPipelineCompiler.h"
#include "luth/renderer/subsystems/RtRestirGiSubsystem.h"
#include "luth/renderer/subsystems/PostProcessSubsystem.h"
#include "luth/renderer/subsystems/LightingSubsystem.h"
#include "luth/renderer/subsystems/VolumetricSubsystem.h"
#include "luth/memory/LinearAllocator.h"
#include <limits>
using namespace Luth;
namespace
{
    template<class T> T Native(u64 n) { return reinterpret_cast<T>(static_cast<uintptr_t>(n)); }
    struct Fixture
    {
        RtRestirGiSubsystem native; GiReservoirVizBindings bindings;
        Memory::GPUSubRegion slice{Native<VkBuffer>(5), 4096, 640ull * 480 * 64};
        std::unique_ptr<CompiledRenderPipeline> pipeline;
        Fixture()
        {
            bindings.enabled = true; bindings.pipeline = Native<VkPipeline>(1); bindings.layout = Native<VkPipelineLayout>(2);
            bindings.set = Native<VkDescriptorSet>(3); bindings.depth = std::shared_ptr<Texture>(Native<Texture*>(10), [](Texture*){});
            bindings.reservoir = slice; bindings.parameters = {640, 480, 640, 480, 17, 8};
            RenderPipelineDefinition definition; definition.AddFeature<GiReservoirVizFeature>(native);
            PipelineInputContract inputs; inputs.resources = {{FogVizResources::Output}, {GiReservoirVizResources::Bindings},
                {RenderResources::SurfaceDepth, ResourceOutputPresence::Optional},
                {GiReservoirVizResources::SpatialReservoir, ResourceOutputPresence::Optional}};
            auto result = RenderPipelineCompiler{}.Compile(std::move(definition), {}, inputs); REQUIRE(result.pipeline);
            pipeline = std::move(result.pipeline);
        }
        PipelineBuildResult Build(RG::RenderGraph& graph, Memory::LinearAllocator& scratch, GraphTextureRef& output,
            bool present = true, bool depthPresent = true, bool packet = true, bool producer = false, u32 width = 640)
        {
            RG::TextureDesc desc; desc.width = width; desc.height = 480;
            GraphTextureRef color{graph.ImportResource(desc, (void*)20, (void*)21, RG::ResourceState::ShaderResource), {Native<const Texture*>(9)}};
            desc.format = RG::TextureFormat::D32_Float;
            GraphTextureRef depth{graph.ImportResource(desc, (void*)22, (void*)23, RG::ResourceState::ShaderResource), {Native<const Texture*>(10)}};
            GraphBufferRef reservoir{graph.ImportBuffer({"GiSpatial", slice.size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT},
                (void*)slice.buffer, RG::ResourceState::Undefined), {&slice, slice.offset, slice.size}};
            if (producer)
            {
                struct Data { RG::BufferHandle output; };
                graph.AddComputePass<Data>("GiSpatial", RG::QueueFamily::AsyncCompute, [&](Data& data, RG::RenderPassBuilder& builder) {
                    reservoir.handle = data.output = builder.WriteBuffer(reservoir.handle);
                }, [](Data&, RG::RenderPassContext&){});
            }
            const GiReservoirVizBindingRef ref{packet ? &bindings : nullptr};
            const std::array inputs{RenderInputBinding::Present(FogVizResources::Output, color),
                RenderInputBinding::Present(GiReservoirVizResources::Bindings, ref),
                depthPresent ? RenderInputBinding::Present(RenderResources::SurfaceDepth, depth) : RenderInputBinding::Absent(RenderResources::SurfaceDepth),
                present ? RenderInputBinding::Present(GiReservoirVizResources::SpatialReservoir, reservoir) : RenderInputBinding::Absent(GiReservoirVizResources::SpatialReservoir)};
            FrameRenderInputs frame; frame.resources = inputs; ViewRenderInputs view; view.id = {1}; view.width = 640; view.height = 480;
            const std::array exports{RenderOutputBinding::Capture(RenderResources::VisualizedLDR, output)};
            return pipeline->Build(graph, frame, view, scratch, exports);
        }
    };
}
TEST_CASE("GiReservoirVizFeature: full and half resolution consume producer slices without imports [renderfeatures]")
{
    for (bool half : {false, true})
    {
        Fixture f; if (half) { f.bindings.parameters.resW = 320; f.bindings.parameters.resH = 240; }
        Memory::LinearAllocator scratch(64 * 1024); RG::RenderGraph graph(scratch); GraphTextureRef output;
        REQUIRE(f.Build(graph, scratch, output).success); REQUIRE(graph.GetPasses().size() == 1);
        const auto& pass = graph.GetPasses()[0]; CHECK(pass.name == "GiReservoirVizPass"); CHECK_FALSE(pass.isCompute);
        CHECK(pass.queueFamily == RG::QueueFamily::Graphics); CHECK_FALSE(pass.hasDepth);
        REQUIRE(pass.colorAttachments.size() == 1); CHECK(pass.colorAttachments[0].loadOp == VK_ATTACHMENT_LOAD_OP_LOAD);
        CHECK(pass.colorAttachments[0].storeOp == VK_ATTACHMENT_STORE_OP_STORE); REQUIRE(pass.reads.size() == 1);
        CHECK(pass.reads[0].index == 2); REQUIRE(pass.bufferReads.size() == 1);
        CHECK(pass.bufferReadStates[0] == RG::ResourceState::FragmentStorageRead);
        CHECK(graph.GetResources().size() == 2); CHECK(graph.GetBuffers().size() == 1);
        CHECK(output.handle.index == 1); CHECK(output.handle.version == 1);
    }
}
TEST_CASE("GiReservoirVizFeature: disabled PT cold and absent producer alias fresh LDR [renderfeatures]")
{
    Fixture f; bool present = true;
    SUBCASE("disabled") { f.bindings.enabled = false; }
    SUBCASE("PT") { f.bindings.enabled = false; present = false; }
    SUBCASE("cold") { f.bindings.pipeline = VK_NULL_HANDLE; }
    SUBCASE("no GI scene/provider") { present = false; }
    for (u32 i = 0; i < 2; ++i)
    {
        Memory::LinearAllocator scratch(64 * 1024); RG::RenderGraph graph(scratch); GraphTextureRef output{{55, 1}, {}};
        REQUIRE(f.Build(graph, scratch, output, present, present).success); CHECK(graph.GetPasses().empty());
        CHECK(output.handle.index == 1); CHECK(output.handle.version == 0);
    }
}
TEST_CASE("GiReservoirVizFeature: rejects incompatible frozen depth buffer and viewport inputs [renderfeatures]")
{
    Fixture f; bool depth = true, packet = true; u32 width = 640;
    SUBCASE("packet") { packet = false; }
    SUBCASE("depth") { depth = false; }
    SUBCASE("owner") { f.bindings.depth.reset(); }
    SUBCASE("layout") { f.bindings.layout = VK_NULL_HANDLE; }
    SUBCASE("descriptor") { f.bindings.set = VK_NULL_HANDLE; }
    SUBCASE("buffer") { f.bindings.reservoir.buffer = Native<VkBuffer>(6); }
    SUBCASE("offset") { ++f.bindings.reservoir.offset; }
    SUBCASE("size") { --f.bindings.reservoir.size; }
    SUBCASE("undersized reservoir") { f.slice.size = f.bindings.reservoir.size = 64; }
    SUBCASE("extent") { ++width; }
    SUBCASE("working resolution") { f.bindings.parameters.resW = 0; }
    SUBCASE("fractional resolution") { f.bindings.parameters.resH = 240.5f; }
    SUBCASE("viewport") { f.bindings.parameters.vx = 641; }
    SUBCASE("m cap") { f.bindings.parameters.mCap = 0; }
    SUBCASE("age cap") { f.bindings.parameters.ageCap = -1; }
    SUBCASE("finite") { f.bindings.parameters.resH = std::numeric_limits<float>::infinity(); }
    Memory::LinearAllocator scratch(64 * 1024); RG::RenderGraph graph(scratch); GraphTextureRef output;
    CHECK_FALSE(f.Build(graph, scratch, output, true, depth, packet, false, width).success);
    CHECK_FALSE(output.handle.IsValid()); CHECK(graph.GetPasses().empty());
}
TEST_CASE("GiReservoirVizFeature: frozen jobs retain depth and declare async reservoir handoff [renderfeatures]")
{
    Fixture f; std::weak_ptr<Texture> retained = f.bindings.depth;
    {
        Memory::LinearAllocator scratch(64 * 1024); RG::RenderGraph graph(scratch); GraphTextureRef output;
        REQUIRE(f.Build(graph, scratch, output, true, true, true, true).success); f.bindings.depth.reset(); CHECK_FALSE(retained.expired());
        REQUIRE(graph.GetPasses().size() == 2); CHECK(graph.GetPasses()[1].bufferReads[0].version == 1);
        graph.Compile(); CHECK_FALSE(graph.GetPasses()[0].culled); CHECK_FALSE(graph.GetPasses()[1].culled);
        REQUIRE(graph.GetPasses()[1].bufferPreBarriers.size() == 1);
        const auto& barrier = graph.GetPasses()[1].bufferPreBarriers[0]; CHECK(barrier.before == RG::ResourceState::StorageBufferWrite);
        CHECK(barrier.after == RG::ResourceState::FragmentStorageRead); CHECK(barrier.crossQueueSrc);
    }
    CHECK(retained.expired());
}
TEST_CASE("GiReservoirVizFeature: cold preparation and producer export reset need no RT device [renderfeatures]")
{
    RtRestirGiSubsystem native;
    CHECK_FALSE(native.PrepareReservoirVizBindings(VK_NULL_HANDLE, {}, {}, 0, 0, 0, 0, 0, 0, 0, false).enabled);
    const auto cold = native.PrepareReservoirVizBindings(VK_NULL_HANDLE, {}, {}, 640, 480, 320, 240, 8, 4, 16, true);
    CHECK(cold.enabled); CHECK_FALSE(cold.pipeline); CHECK_FALSE(cold.depth);
    Memory::LinearAllocator scratch(64 * 1024); RG::RenderGraph graph(scratch);
    const auto absent = RtRestirGiSubsystem::AddPasses(graph, {}, {}, {}, RestirGiBindings{}, {});
    CHECK_FALSE(absent.irradiance.IsValid()); CHECK_FALSE(absent.spatial.IsValid()); CHECK(graph.GetPasses().empty());
}
TEST_CASE("GiReservoirVizFeature: combined stages sort reversed declarations and pass through to one final output [renderfeatures]")
{
    RtRestirGiSubsystem gi; VolumetricSubsystem fog; LightingSubsystem lighting; PostProcessSubsystem post;
    RenderPipelineDefinition definition;
    const auto giId = definition.AddFeature<GiReservoirVizFeature>(gi);
    const auto fogId = definition.AddFeature<FogVizFeature>(fog);
    const auto clusterId = definition.AddFeature<ClusterVizFeature>(lighting);
    const auto slimId = definition.AddFeature<SlimVizFeature>(post);
    PipelineInputContract inputs; inputs.resources = {{RenderResources::TonemappedLDR}, {SlimVizResources::Bindings},
        {ClusterVizResources::Bindings}, {FogVizResources::Bindings}, {GiReservoirVizResources::Bindings},
        {RenderResources::SurfaceDepth, ResourceOutputPresence::Optional}, {RenderResources::ClusterGrid, ResourceOutputPresence::Optional},
        {GiReservoirVizResources::SpatialReservoir, ResourceOutputPresence::Optional},
        {RenderResources::FogDensity, ResourceOutputPresence::Optional}, {RenderResources::ResolvedFog, ResourceOutputPresence::Optional},
        {RenderResources::Normal, ResourceOutputPresence::Optional}, {RenderResources::Roughness, ResourceOutputPresence::Optional},
        {RenderResources::MotionVectors, ResourceOutputPresence::Optional}, {RenderResources::MaterialID, ResourceOutputPresence::Optional}};
    auto result = RenderPipelineCompiler{}.Compile(std::move(definition), {}, inputs); REQUIRE(result.pipeline);
    CHECK(result.diagnostics.empty()); const auto order = result.pipeline->FeatureOrder(); REQUIRE(order.size() == 4);
    CHECK(order[0] == slimId); CHECK(order[1] == clusterId); CHECK(order[2] == fogId); CHECK(order[3] == giId);
    SlimVizBindings slim; ClusterVizBindings cluster; FogVizBindings fogPacket; GiReservoirVizBindings giPacket;
    const SlimVizBindingRef slimRef{&slim}; const ClusterVizBindingRef clusterRef{&cluster};
    const FogVizBindingRef fogRef{&fogPacket}; const GiReservoirVizBindingRef giRef{&giPacket};
    for (u32 i = 0; i < 2; ++i)
    {
        Memory::LinearAllocator scratch(64 * 1024); RG::RenderGraph graph(scratch);
        RG::TextureDesc desc; desc.width = 640; desc.height = 480;
        GraphTextureRef input{graph.ImportResource(desc, (void*)20, (void*)21, RG::ResourceState::ShaderResource), {Native<const Texture*>(9)}};
        const std::array resources{RenderInputBinding::Present(RenderResources::TonemappedLDR, input),
            RenderInputBinding::Present(SlimVizResources::Bindings, slimRef),
            RenderInputBinding::Present(ClusterVizResources::Bindings, clusterRef),
            RenderInputBinding::Present(FogVizResources::Bindings, fogRef),
            RenderInputBinding::Present(GiReservoirVizResources::Bindings, giRef),
            RenderInputBinding::Absent(RenderResources::SurfaceDepth), RenderInputBinding::Absent(RenderResources::ClusterGrid),
            RenderInputBinding::Absent(GiReservoirVizResources::SpatialReservoir), RenderInputBinding::Absent(RenderResources::FogDensity),
            RenderInputBinding::Absent(RenderResources::ResolvedFog), RenderInputBinding::Absent(RenderResources::Normal),
            RenderInputBinding::Absent(RenderResources::Roughness), RenderInputBinding::Absent(RenderResources::MotionVectors),
            RenderInputBinding::Absent(RenderResources::MaterialID)};
        FrameRenderInputs frame; frame.resources = resources; ViewRenderInputs view; view.id = {1}; view.width = 640; view.height = 480;
        GraphTextureRef output; const std::array exports{RenderOutputBinding::Capture(RenderResources::VisualizedLDR, output)};
        REQUIRE(result.pipeline->Build(graph, frame, view, scratch, exports).success);
        CHECK(output.handle == input.handle); CHECK(output.binding.texture == input.binding.texture); CHECK(graph.GetPasses().empty());
        CHECK(graph.GetResources().size() == 1);
    }
}

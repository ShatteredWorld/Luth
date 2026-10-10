#include <doctest/doctest.h>
#include "luth/core/diagnostics/Log.h"
#include "luth/renderer/features/FogCompositeFeature.h"
#include "luth/renderer/features/RenderPipelineCompiler.h"
#include "luth/renderer/subsystems/VolumetricSubsystem.h"
#include "luth/renderer/CameraParams.h"
#include "luth/memory/LinearAllocator.h"

using namespace Luth;
namespace
{
    template<class T> T Native(u64 value) { return reinterpret_cast<T>(static_cast<uintptr_t>(value)); }
    struct Fixture
    {
        VolumetricSubsystem native;
        FogCompositeBindings bindings;
        std::unique_ptr<CompiledRenderPipeline> pipeline;
        Fixture()
        {
            bindings.enabled = true; bindings.pipeline = Native<VkPipeline>(1);
            bindings.layout = Native<VkPipelineLayout>(2); bindings.sets.fill(Native<VkDescriptorSet>(3));
            bindings.depth = {Native<const Texture*>(10)}; bindings.resolved = {Native<const Texture*>(13)};
            RenderPipelineDefinition definition; definition.AddFeature<FogCompositeFeature>(native);
            PipelineInputContract inputs;
            inputs.resources = {{RenderResources::SkyHDR}, {RenderResources::SurfaceDepth},
                {FogCompositeResources::Bindings}, {RenderResources::ResolvedFog, ResourceOutputPresence::Optional}};
            auto compiled = RenderPipelineCompiler{}.Compile(std::move(definition), {}, inputs);
            REQUIRE(compiled.pipeline); pipeline = std::move(compiled.pipeline);
        }
        PipelineBuildResult Build(RG::RenderGraph& graph, Memory::LinearAllocator& scratch,
            GraphTextureRef& output, bool fogPresent = true, bool packetPresent = true, u32 width = 640, bool producer = false)
        {
            RG::TextureDesc colorDesc; colorDesc.name = "SceneColor"; colorDesc.width = width;
            colorDesc.height = 480; colorDesc.format = RG::TextureFormat::RGBA16_Float;
            auto depthDesc = colorDesc; depthDesc.name = "SceneDepth"; depthDesc.format = RG::TextureFormat::D32_Float;
            auto fogDesc = colorDesc; fogDesc.name = "ResolvedFog"; fogDesc.width = 160; fogDesc.height = 90;
            const GraphTextureRef color{graph.ImportResource(colorDesc, (void*)5, (void*)6, RG::ResourceState::ShaderResource),
                {Native<const Texture*>(7)}};
            const GraphTextureRef depth{graph.ImportResource(depthDesc, (void*)8, (void*)9, RG::ResourceState::Undefined),
                {Native<const Texture*>(10)}};
            GraphTextureRef fog{graph.ImportResource(fogDesc, (void*)11, (void*)12, RG::ResourceState::ComputeWrite),
                {Native<const Texture*>(13)}};
            if (producer)
            {
                struct Data { RG::ResourceHandle output; };
                graph.AddComputePass<Data>("VolumetricResolve", RG::QueueFamily::AsyncCompute,
                    [&](Data& data, RG::RenderPassBuilder& builder) { data.output = builder.WriteStorageImage(fog.handle); fog.handle = data.output; },
                    [](Data&, RG::RenderPassContext&) {});
            }
            const FogCompositeBindingRef binding{packetPresent ? &bindings : nullptr};
            const std::array resources{RenderInputBinding::Present(RenderResources::SkyHDR, color),
                RenderInputBinding::Present(RenderResources::SurfaceDepth, depth),
                RenderInputBinding::Present(FogCompositeResources::Bindings, binding),
                fogPresent ? RenderInputBinding::Present(RenderResources::ResolvedFog, fog)
                    : RenderInputBinding::Absent(RenderResources::ResolvedFog)};
            FrameRenderInputs frame; frame.resources = resources;
            ViewRenderInputs view; view.id = {1}; view.width = 640; view.height = 480;
            const std::array exports{RenderOutputBinding::Capture(RenderResources::FoggedHDR, output)};
            return pipeline->Build(graph, frame, view, scratch, exports);
        }
    };
}
TEST_CASE("FogCompositeFeature: native blend reuses stage and fog imports [renderfeatures]")
{
    Fixture fixture; Memory::LinearAllocator scratch(64 * 1024); RG::RenderGraph graph(scratch);
    GraphTextureRef output; REQUIRE(fixture.Build(graph, scratch, output).success);
    REQUIRE(graph.GetPasses().size() == 1); CHECK(graph.GetResources().size() == 3);
    const auto& pass = graph.GetPasses()[0]; CHECK(pass.name == "VolumetricComposite");
    CHECK_FALSE(pass.isCompute); CHECK(pass.queueFamily == RG::QueueFamily::Graphics);
    REQUIRE(pass.colorAttachments.size() == 1); CHECK(pass.colorAttachments[0].handle == output.handle);
    CHECK(pass.colorAttachments[0].loadOp == VK_ATTACHMENT_LOAD_OP_LOAD);
    CHECK(pass.colorAttachments[0].storeOp == VK_ATTACHMENT_STORE_OP_STORE);
    CHECK_FALSE(pass.hasDepth); REQUIRE(pass.reads.size() == 2);
    CHECK(pass.reads[0].index == 2); CHECK(pass.reads[1].index == 3);
    CHECK(output.handle.index == 1); CHECK(output.handle.version == 1);
    CHECK(output.binding.texture == Native<const Texture*>(7));
    graph.Compile(); CHECK_FALSE(graph.GetPasses()[0].culled); CHECK_FALSE(graph.GetPasses()[0].preBarriers.empty());
}
TEST_CASE("FogCompositeFeature: disabled absent and cold fog publish current sky alias [renderfeatures]")
{
    Fixture fixture; bool present = true;
    SUBCASE("disabled") { fixture.bindings.enabled = false; }
    SUBCASE("absent") { present = false; }
    SUBCASE("cold") { fixture.bindings.pipeline = VK_NULL_HANDLE; }
    for (u32 i = 0; i < 2; ++i)
    {
        Memory::LinearAllocator scratch(64 * 1024); RG::RenderGraph graph(scratch);
        GraphTextureRef output{{55, 1}, {Native<const Texture*>(99)}};
        REQUIRE(fixture.Build(graph, scratch, output, present).success);
        CHECK(graph.GetPasses().empty()); CHECK(output.handle.index == 1); CHECK(output.handle.version == 0);
        CHECK(output.binding.texture == Native<const Texture*>(7));
    }
}
TEST_CASE("FogCompositeFeature: invalid frozen bindings reject before registration [renderfeatures]")
{
    Fixture fixture; bool packet = true; u32 width = 640;
    SUBCASE("packet") { packet = false; }
    SUBCASE("layout") { fixture.bindings.layout = VK_NULL_HANDLE; }
    SUBCASE("set") { fixture.bindings.sets[1] = VK_NULL_HANDLE; }
    SUBCASE("depth binding") { fixture.bindings.depth.texture = Native<const Texture*>(99); }
    SUBCASE("resolved binding") { fixture.bindings.resolved.texture = Native<const Texture*>(99); }
    SUBCASE("extent") { width = 641; }
    Memory::LinearAllocator scratch(64 * 1024); RG::RenderGraph graph(scratch);
    GraphTextureRef output{{55, 1}, {Native<const Texture*>(99)}};
    CHECK_FALSE(fixture.Build(graph, scratch, output, true, packet, width).success);
    CHECK_FALSE(output.handle.IsValid()); CHECK(graph.GetPasses().empty());
}
TEST_CASE("FogCompositeFeature: disabled preparation requires no device [renderfeatures]")
{
    VolumetricSubsystem native; FogViewState state; CameraParams camera;
    const auto packet = native.PrepareCompositeBindings(state, 7, camera, VK_NULL_HANDLE, false);
    CHECK_FALSE(packet.enabled); CHECK_FALSE(packet.pipeline); CHECK_FALSE(packet.sets[1]);
}
TEST_CASE("FogCompositeFeature: async resolve handoff uses producer version and barrier [renderfeatures]")
{
    Fixture fixture; Memory::LinearAllocator scratch(64 * 1024); RG::RenderGraph graph(scratch);
    GraphTextureRef output; REQUIRE(fixture.Build(graph, scratch, output, true, true, 640, true).success);
    REQUIRE(graph.GetPasses().size() == 2);
    CHECK(graph.GetPasses()[1].reads[1].index == 3); CHECK(graph.GetPasses()[1].reads[1].version == 1);
    graph.Compile(); CHECK_FALSE(graph.GetPasses()[0].culled);
    bool handoff = false;
    for (const auto& barrier : graph.GetPasses()[1].preBarriers)
        if (barrier.resource.index == 3)
        {
            CHECK(barrier.before == RG::ResourceState::ComputeWrite);
            CHECK(barrier.after == RG::ResourceState::ShaderResource);
            CHECK(barrier.crossQueueSrc); handoff = true;
        }
    CHECK(handoff);
}
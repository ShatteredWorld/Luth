#include <doctest/doctest.h>
#include "luth/core/diagnostics/Log.h"
#include "luth/renderer/features/RefractionBackdropFeature.h"
#include "luth/renderer/features/RenderPipelineCompiler.h"
#include "luth/renderer/subsystems/TransparencySubsystem.h"
#include "luth/memory/LinearAllocator.h"

using namespace Luth;
namespace
{
    template<class T> T Native(u64 value) { return reinterpret_cast<T>(static_cast<uintptr_t>(value)); }
    struct Fixture
    {
        TransparencySubsystem native;
        RefractionBackdropBindings bindings;
        std::unique_ptr<CompiledRenderPipeline> pipeline;
        RG::ResourceHandle source;
        Fixture()
        {
            bindings.enabled = true; bindings.image = Native<VkImage>(11); bindings.view = Native<VkImageView>(12);
            bindings.binding = {Native<const Texture*>(13)}; bindings.width = 640; bindings.height = 480;
            RenderPipelineDefinition definition; definition.AddFeature<RefractionBackdropFeature>(native);
            PipelineInputContract inputs;
            inputs.resources = {{RenderResources::FoggedHDR}, {RefractionResources::Bindings}};
            auto compiled = RenderPipelineCompiler{}.Compile(std::move(definition), {}, inputs);
            REQUIRE(compiled.pipeline); pipeline = std::move(compiled.pipeline);
        }
        PipelineBuildResult Build(RG::RenderGraph& graph, Memory::LinearAllocator& scratch,
            GraphTextureRef& output, bool packetPresent = true, u32 width = 640,
            RG::TextureFormat format = RG::TextureFormat::RGBA16_Float, bool destinationImported = false)
        {
            RG::TextureDesc desc; desc.name = "SceneColor"; desc.width = width; desc.height = 480; desc.format = format;
            source = graph.ImportResource(desc, (void*)5, (void*)6, RG::ResourceState::ShaderResource);
            struct Producer { RG::ResourceHandle output; };
            graph.AddPass<Producer>("FogCompositeProducer",
                [&](Producer& data, RG::RenderPassBuilder& builder) { source = data.output = builder.Write(source); },
                [](Producer&, RG::RenderPassContext&) {});
            if (destinationImported)
                graph.ImportResource(desc, (void*)bindings.image, (void*)bindings.view, RG::ResourceState::ShaderResource);
            const GraphTextureRef color{source, {Native<const Texture*>(7)}};
            const RefractionBackdropBindingRef binding{packetPresent ? &bindings : nullptr};
            const std::array resources{RenderInputBinding::Present(RenderResources::FoggedHDR, color),
                RenderInputBinding::Present(RefractionResources::Bindings, binding)};
            FrameRenderInputs frame; frame.resources = resources;
            ViewRenderInputs view; view.id = {1}; view.width = 640; view.height = 480;
            const std::array exports{RenderOutputBinding::Capture(RenderResources::RefractionBackdrop, output)};
            return pipeline->Build(graph, frame, view, scratch, exports);
        }
    };
}
TEST_CASE("RefractionBackdropFeature: copy reads exact fog stage and imports destination once [renderfeatures]")
{
    Fixture fixture; Memory::LinearAllocator scratch(64 * 1024); RG::RenderGraph graph(scratch);
    GraphTextureRef output; REQUIRE(fixture.Build(graph, scratch, output).success);
    REQUIRE(graph.GetPasses().size() == 2); CHECK(graph.GetResources().size() == 2);
    const auto& pass = graph.GetPasses()[1]; CHECK(pass.name == "RefractionBackdropCopy");
    CHECK(pass.isCompute); CHECK(pass.queueFamily == RG::QueueFamily::Graphics);
    CHECK(pass.colorAttachments.empty()); CHECK_FALSE(pass.hasDepth);
    REQUIRE(pass.reads.size() == 1); CHECK(pass.reads[0] == fixture.source);
    CHECK(pass.readStates[0] == RG::ResourceState::TransferSrc);
    REQUIRE(pass.writes.size() == 1); CHECK(pass.writes[0] == output.handle);
    CHECK(pass.writeStates[0] == RG::ResourceState::TransferDst);
    CHECK(output.handle.index == 2); CHECK(output.handle.version == 1);
    CHECK(output.binding.texture == fixture.bindings.binding.texture);
    CHECK(graph.GetResources()[1].desc.name == "RefractionBackdrop");
    graph.Compile(); CHECK_FALSE(graph.GetPasses()[1].culled);
    CHECK_FALSE(graph.GetPasses()[1].preBarriers.empty());
}
TEST_CASE("RefractionBackdropFeature: empty and cold inputs reset graph-local output [renderfeatures]")
{
    Fixture fixture;
    SUBCASE("empty transparent bucket") { fixture.bindings.enabled = false; }
    SUBCASE("no native backdrop") { fixture.bindings.binding.texture = nullptr; }
    for (u32 i = 0; i < 2; ++i)
    {
        Memory::LinearAllocator scratch(64 * 1024); RG::RenderGraph graph(scratch);
        GraphTextureRef output{{55, 1}, {Native<const Texture*>(99)}};
        REQUIRE(fixture.Build(graph, scratch, output).success);
        CHECK_FALSE(output.handle.IsValid()); CHECK_FALSE(output.binding.texture);
        CHECK(graph.GetPasses().size() == 1); CHECK(graph.GetResources().size() == 1);
    }
}
TEST_CASE("RefractionBackdropFeature: rejects invalid copy before import or registration [renderfeatures]")
{
    Fixture fixture; bool packet = true, imported = false; u32 width = 640;
    auto format = RG::TextureFormat::RGBA16_Float;
    SUBCASE("packet") { packet = false; }
    SUBCASE("image") { fixture.bindings.image = VK_NULL_HANDLE; }
    SUBCASE("view") { fixture.bindings.view = VK_NULL_HANDLE; }
    SUBCASE("source extent") { width = 641; }
    SUBCASE("destination extent") { fixture.bindings.width = 641; }
    SUBCASE("source format") { format = RG::TextureFormat::RGBA8_Unorm; }
    SUBCASE("destination subresource") { fixture.bindings.binding.baseMip = 1; }
    SUBCASE("same physical image") { fixture.bindings.image = Native<VkImage>(5); }
    SUBCASE("same physical texture") { fixture.bindings.binding.texture = Native<const Texture*>(7); }
    SUBCASE("already imported") { imported = true; }
    Memory::LinearAllocator scratch(64 * 1024); RG::RenderGraph graph(scratch);
    GraphTextureRef output{{55, 1}, {Native<const Texture*>(99)}};
    CHECK_FALSE(fixture.Build(graph, scratch, output, packet, width, format, imported).success);
    CHECK_FALSE(output.handle.IsValid()); CHECK(graph.GetPasses().size() == 1);
    CHECK(graph.GetResources().size() == (imported ? 2 : 1));
}
TEST_CASE("RefractionBackdropFeature: transparent sampled read follows copy transition [renderfeatures]")
{
    Fixture fixture; Memory::LinearAllocator scratch(64 * 1024); RG::RenderGraph graph(scratch);
    GraphTextureRef output; REQUIRE(fixture.Build(graph, scratch, output).success);
    struct Consumer { RG::ResourceHandle sampled; };
    graph.AddPass<Consumer>("TransparentConsumer",
        [&](Consumer& data, RG::RenderPassBuilder& builder) { data.sampled = builder.Read(output.handle); builder.SetHasSideEffect(); },
        [](Consumer&, RG::RenderPassContext&) {});
    graph.Compile(); REQUIRE(graph.GetPasses().size() == 3);
    CHECK(graph.GetPasses()[2].reads[0] == output.handle);
    bool transitioned = false;
    for (const auto& barrier : graph.GetPasses()[2].preBarriers)
        if (barrier.resource.index == output.handle.index)
        {
            CHECK(barrier.before == RG::ResourceState::TransferDst);
            CHECK(barrier.after == RG::ResourceState::ShaderResource);
            CHECK_FALSE(barrier.crossQueueSrc); transitioned = true;
        }
    CHECK(transitioned);
}
TEST_CASE("RefractionBackdropFeature: view packets keep separate physical destinations [renderfeatures]")
{
    Fixture fixture;
    for (u32 i = 0; i < 2; ++i)
    {
        fixture.bindings.image = Native<VkImage>(11 + i * 10);
        fixture.bindings.view = Native<VkImageView>(12 + i * 10);
        fixture.bindings.binding.texture = Native<const Texture*>(13 + i * 10);
        Memory::LinearAllocator scratch(64 * 1024); RG::RenderGraph graph(scratch);
        GraphTextureRef output; REQUIRE(fixture.Build(graph, scratch, output).success);
        CHECK(graph.GetResources()[output.handle.index - 1].image == fixture.bindings.image);
        CHECK(output.binding.texture == fixture.bindings.binding.texture);
    }
}
TEST_CASE("RefractionBackdropFeature: absent native preparation requires no device [renderfeatures]")
{
    const auto packet = TransparencySubsystem::PrepareBackdropBindings({}, true);
    CHECK(packet.enabled); CHECK_FALSE(packet.binding.texture); CHECK_FALSE(packet.image); CHECK_FALSE(packet.view);
}

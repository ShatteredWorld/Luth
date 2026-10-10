#include <doctest/doctest.h>
#include "luth/core/diagnostics/Log.h"
#include "luth/renderer/features/SkyFeature.h"
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
        SkyBindings bindings;
        std::unique_ptr<CompiledRenderPipeline> pipeline;
        Fixture()
        {
            bindings.pipeline = Native<VkPipeline>(1); bindings.layout = Native<VkPipelineLayout>(2);
            bindings.vertex = Native<VkBuffer>(3); bindings.sets.fill(Native<VkDescriptorSet>(4));
            RenderPipelineDefinition definition; definition.AddFeature<SkyFeature>(native);
            PipelineInputContract inputs;
            inputs.resources = {{RenderResources::OpaqueHDR}, {RenderResources::LitDepth}, {SkyResources::Bindings}};
            auto compiled = RenderPipelineCompiler{}.Compile(std::move(definition), {}, inputs);
            REQUIRE(compiled.pipeline); pipeline = std::move(compiled.pipeline);
        }
        PipelineBuildResult Build(RG::RenderGraph& graph, Memory::LinearAllocator& scratch,
            GraphTextureRef& output, bool packetPresent = true, u32 colorWidth = 640,
            RG::TextureFormat depthFormat = RG::TextureFormat::D32_Float)
        {
            RG::TextureDesc colorDesc; colorDesc.name = "SceneColor"; colorDesc.width = colorWidth;
            colorDesc.height = 480; colorDesc.format = RG::TextureFormat::RGBA16_Float;
            RG::TextureDesc depthDesc; depthDesc.name = "SceneDepth"; depthDesc.width = 640;
            depthDesc.height = 480; depthDesc.format = depthFormat;
            const GraphTextureRef color{graph.ImportResource(colorDesc, (void*)5, (void*)6, RG::ResourceState::ShaderResource),
                {Native<const Texture*>(7)}};
            const GraphTextureRef depth{graph.ImportResource(depthDesc, (void*)8, (void*)9, RG::ResourceState::Undefined),
                {Native<const Texture*>(10)}};
            const SkyBindingRef binding{packetPresent ? &bindings : nullptr};
            const std::array resources{RenderInputBinding::Present(RenderResources::OpaqueHDR, color),
                RenderInputBinding::Present(RenderResources::LitDepth, depth),
                RenderInputBinding::Present(SkyResources::Bindings, binding)};
            FrameRenderInputs frame; frame.resources = resources;
            ViewRenderInputs view; view.id = {1}; view.width = 640; view.height = 480;
            const std::array exports{RenderOutputBinding::Capture(RenderResources::SkyHDR, output)};
            return pipeline->Build(graph, frame, view, scratch, exports);
        }
    };
}
TEST_CASE("SkyFeature: native pass aliases opaque HDR and preserves attachment policy [renderfeatures]")
{
    Fixture fixture;
    Memory::LinearAllocator scratch(64 * 1024); RG::RenderGraph graph(scratch);
    GraphTextureRef output;
    REQUIRE(fixture.Build(graph, scratch, output).success);
    REQUIRE(graph.GetPasses().size() == 1); CHECK(graph.GetResources().size() == 2);
    const auto& pass = graph.GetPasses()[0];
    CHECK(pass.name == "SkyboxPass"); CHECK_FALSE(pass.isCompute); CHECK(pass.queueFamily == RG::QueueFamily::Graphics);
    REQUIRE(pass.colorAttachments.size() == 1);
    CHECK(pass.colorAttachments[0].handle == output.handle);
    CHECK(pass.colorAttachments[0].loadOp == VK_ATTACHMENT_LOAD_OP_LOAD);
    CHECK(pass.colorAttachments[0].storeOp == VK_ATTACHMENT_STORE_OP_STORE);
    CHECK(pass.hasDepth); CHECK(pass.depthAttachment.loadOp == VK_ATTACHMENT_LOAD_OP_LOAD);
    CHECK(pass.depthAttachment.storeOp == VK_ATTACHMENT_STORE_OP_DONT_CARE);
    CHECK(output.handle.index == 1); CHECK(output.handle.version == 1);
    CHECK(output.binding.texture == Native<const Texture*>(7));
    graph.Compile(); CHECK_FALSE(graph.GetPasses()[0].culled);
}
TEST_CASE("SkyFeature: incomplete inputs reject before registration and reset exports [renderfeatures]")
{
    Fixture fixture; bool packet = true; u32 width = 640;
    auto format = RG::TextureFormat::D32_Float;
    SUBCASE("packet absent") { packet = false; }
    SUBCASE("layout absent") { fixture.bindings.layout = VK_NULL_HANDLE; }
    SUBCASE("descriptor absent") { fixture.bindings.sets[4] = VK_NULL_HANDLE; }
    SUBCASE("extent mismatch") { width = 641; }
    SUBCASE("wrong depth format") { format = RG::TextureFormat::R8_Unorm; }
    Memory::LinearAllocator scratch(64 * 1024); RG::RenderGraph graph(scratch);
    GraphTextureRef output{{55, 1}, {Native<const Texture*>(7)}};
    CHECK_FALSE(fixture.Build(graph, scratch, output, packet, width, format).success);
    CHECK_FALSE(output.handle.IsValid()); CHECK(graph.GetPasses().empty());
}
TEST_CASE("SkyFeature: cold native resources preserve attachment pass and independent graphs [renderfeatures]")
{
    Fixture fixture;
    fixture.bindings = {};
    for (u32 i = 0; i < 2; ++i)
    {
        Memory::LinearAllocator scratch(64 * 1024); RG::RenderGraph graph(scratch);
        GraphTextureRef output;
        REQUIRE(fixture.Build(graph, scratch, output).success);
        CHECK(output.handle.IsValid()); CHECK(graph.GetPasses().size() == 1);
    }
}
TEST_CASE("SkyFeature: native preparation copies descriptor slots [renderfeatures]")
{
    LightingSubsystem native;
    std::array<VkDescriptorSet, 5> sets; sets.fill(Native<VkDescriptorSet>(1));
    const auto packet = native.PrepareSkyBindings(sets);
    sets.fill(Native<VkDescriptorSet>(2));
    for (auto set : packet.sets) CHECK(set == Native<VkDescriptorSet>(1));
    CHECK_FALSE(packet.pipeline); CHECK_FALSE(packet.vertex);
}

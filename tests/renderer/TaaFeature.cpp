#include <doctest/doctest.h>
#include "luth/core/diagnostics/Log.h"
#include "luth/renderer/features/TaaFeature.h"
#include "luth/renderer/features/RenderPipelineCompiler.h"
#include "luth/renderer/subsystems/PostProcessSubsystem.h"
#include "luth/memory/LinearAllocator.h"
#include <limits>

using namespace Luth;
namespace
{
    template<class T> T Native(u64 value) { return reinterpret_cast<T>(static_cast<uintptr_t>(value)); }
    struct Fixture
    {
        PostProcessSubsystem native;
        TaaBindings packet;
        std::unique_ptr<CompiledRenderPipeline> pipeline;
        Fixture()
        {
            packet.enabled = true; packet.pipeline = Native<VkPipeline>(1); packet.layout = Native<VkPipelineLayout>(2);
            packet.set = Native<VkDescriptorSet>(3); packet.width = 640; packet.height = 480;
            packet.state = std::make_shared<TaaViewState>();
            for (u32 i = 0; i < 3; ++i) packet.sources[i] = {Native<const Texture*>(10 + i)};
            packet.previous = {Native<const Texture*>(20)}; packet.current = {Native<const Texture*>(21)};
            packet.previousImage = Native<VkImage>(30); packet.previousView = Native<VkImageView>(31);
            packet.currentImage = Native<VkImage>(32); packet.currentView = Native<VkImageView>(33);
            RenderPipelineDefinition definition; definition.AddFeature<TaaFeature>(native);
            PipelineInputContract inputs;
            inputs.resources = {{RenderResources::TransparentHDR}, {RenderResources::MotionVectors},
                {RenderResources::LitDepth}, {TaaResources::Bindings}};
            auto compiled = RenderPipelineCompiler{}.Compile(std::move(definition), {}, inputs);
            REQUIRE(compiled.pipeline); pipeline = std::move(compiled.pipeline);
        }
        PipelineBuildResult Build(RG::RenderGraph& graph, Memory::LinearAllocator& scratch,
            GraphTextureRef& output, bool packetPresent = true, u32 width = 640, bool producer = false)
        {
            std::array<GraphTextureRef, 3> textures;
            const std::array formats{RG::TextureFormat::RGBA16_Float, RG::TextureFormat::RG16_Float, RG::TextureFormat::D32_Float};
            for (u32 i = 0; i < 3; ++i)
            {
                RG::TextureDesc desc; desc.name = "Input"; desc.width = width; desc.height = 480; desc.format = formats[i];
                textures[i] = {graph.ImportResource(desc, (void*)(uintptr_t)(40 + i), (void*)(uintptr_t)(50 + i),
                    RG::ResourceState::ShaderResource), {Native<const Texture*>(10 + i)}};
            }
            if (producer)
            {
                struct Data {};
                graph.AddPass<Data>("TransparentProducer", [&](Data&, RG::RenderPassBuilder& builder) {
                    textures[0].handle = builder.Write(textures[0].handle, VK_ATTACHMENT_LOAD_OP_LOAD, VK_ATTACHMENT_STORE_OP_STORE);
                    textures[2].handle = builder.WriteDepth(textures[2].handle, VK_ATTACHMENT_LOAD_OP_LOAD, VK_ATTACHMENT_STORE_OP_STORE);
                }, [](Data&, RG::RenderPassContext&) {});
            }
            const TaaBindingRef binding{packetPresent ? &packet : nullptr};
            const std::array resources{RenderInputBinding::Present(RenderResources::TransparentHDR, textures[0]),
                RenderInputBinding::Present(RenderResources::MotionVectors, textures[1]),
                RenderInputBinding::Present(RenderResources::LitDepth, textures[2]),
                RenderInputBinding::Present(TaaResources::Bindings, binding)};
            FrameRenderInputs frame; frame.resources = resources; frame.renderFrameIndex = 7;
            ViewRenderInputs view; view.id = {1}; view.width = 640; view.height = 480;
            const std::array exports{RenderOutputBinding::Capture(RenderResources::ResolvedHDR, output)};
            return pipeline->Build(graph, frame, view, scratch, exports);
        }
    };
}
TEST_CASE("TaaFeature: resolve exports an allocated stage and declares all sampled reads [renderfeatures]")
{
    Fixture fixture; Memory::LinearAllocator scratch(64 * 1024); RG::RenderGraph graph(scratch); GraphTextureRef output;
    REQUIRE(fixture.Build(graph, scratch, output, true, 640, true).success);
    REQUIRE(graph.GetPasses().size() == 2); REQUIRE(graph.GetResources().size() == 5);
    const auto& pass = graph.GetPasses()[1]; CHECK(pass.name == "TaaResolve");
    CHECK_FALSE(pass.isCompute); CHECK(pass.queueFamily == RG::QueueFamily::Graphics); CHECK_FALSE(pass.hasDepth);
    REQUIRE(pass.colorAttachments.size() == 1); CHECK(pass.colorAttachments[0].handle == output.handle);
    CHECK(pass.colorAttachments[0].loadOp == VK_ATTACHMENT_LOAD_OP_DONT_CARE);
    CHECK(pass.colorAttachments[0].storeOp == VK_ATTACHMENT_STORE_OP_STORE);
    REQUIRE(pass.reads.size() == 4);
    for (u32 i = 0; i < 4; ++i) CHECK(pass.reads[i].index == i + 1);
    CHECK(pass.reads[0].version == 1); CHECK(pass.reads[2].version == 1);
    CHECK(output.handle.index == 5); CHECK(output.handle.version == 1);
    CHECK(output.binding.texture == fixture.packet.current.texture); CHECK_FALSE(fixture.packet.state->recorded);
    CHECK(graph.GetResources()[3].initialState == RG::ResourceState::ShaderResource);
    CHECK(graph.GetResources()[4].initialState == RG::ResourceState::Undefined);
    graph.Compile(); CHECK_FALSE(graph.GetPasses()[0].culled); CHECK_FALSE(graph.GetPasses()[1].culled);
    bool colorBarrier = false, depthBarrier = false;
    for (const auto& barrier : graph.GetPasses()[1].preBarriers)
    {
        if (barrier.resource.index == 1) { CHECK(barrier.after == RG::ResourceState::ShaderResource); colorBarrier = true; }
        if (barrier.resource.index == 3) { CHECK(barrier.after == RG::ResourceState::ShaderResource); depthBarrier = true; }
    }
    CHECK(colorBarrier); CHECK(depthBarrier);
}
TEST_CASE("TaaFeature: disabled and cold states publish this graph's input and invalidate history [renderfeatures]")
{
    Fixture fixture;
    SUBCASE("disabled") { fixture.packet.enabled = false; }
    SUBCASE("cold") { fixture.packet.pipeline = VK_NULL_HANDLE; }
    for (u32 i = 0; i < 2; ++i)
    {
        fixture.packet.state->history.Commit(6, 1); fixture.packet.state->recorded = true;
        Memory::LinearAllocator scratch(64 * 1024); RG::RenderGraph graph(scratch);
        GraphTextureRef output{{99, 2}, {Native<const Texture*>(99)}};
        REQUIRE(fixture.Build(graph, scratch, output).success);
        CHECK(graph.GetPasses().empty()); CHECK(graph.GetResources().size() == 3);
        CHECK(output.handle.index == 1); CHECK(output.handle.version == 0); CHECK(output.binding.texture == fixture.packet.sources[0].texture);
        CHECK_FALSE(fixture.packet.state->recorded); CHECK_FALSE(fixture.packet.state->history.valid);
    }
}
TEST_CASE("TaaFeature: malformed frozen bindings fail before history import [renderfeatures]")
{
    Fixture fixture; bool present = true; u32 width = 640;
    SUBCASE("packet") { present = false; }
    SUBCASE("layout") { fixture.packet.layout = VK_NULL_HANDLE; }
    SUBCASE("set") { fixture.packet.set = VK_NULL_HANDLE; }
    SUBCASE("state") { fixture.packet.state.reset(); }
    SUBCASE("extent") { width = 641; }
    SUBCASE("history extent") { fixture.packet.width = 641; }
    SUBCASE("sample binding") { fixture.packet.sources[1].texture = Native<const Texture*>(99); }
    SUBCASE("history image") { fixture.packet.currentImage = VK_NULL_HANDLE; }
    SUBCASE("history view") { fixture.packet.previousView = VK_NULL_HANDLE; }
    SUBCASE("history alias") { fixture.packet.currentImage = fixture.packet.previousImage; }
    SUBCASE("history texture alias") { fixture.packet.current.texture = fixture.packet.previous.texture; }
    SUBCASE("subresource") { fixture.packet.current.mipCount = 2; }
    SUBCASE("duplicate physical history") { fixture.packet.currentImage = Native<VkImage>(40); }
    SUBCASE("feedback") { fixture.packet.constants.temporalAlpha = 1.1f; }
    SUBCASE("feedback NaN") { fixture.packet.constants.temporalAlpha = std::numeric_limits<float>::quiet_NaN(); }
    SUBCASE("sky NaN") { fixture.packet.constants.skyReproj[0][0] = std::numeric_limits<float>::quiet_NaN(); }
    Memory::LinearAllocator scratch(64 * 1024); RG::RenderGraph graph(scratch);
    GraphTextureRef output{{99, 2}, {Native<const Texture*>(99)}};
    CHECK_FALSE(fixture.Build(graph, scratch, output, present, width).success);
    CHECK_FALSE(output.handle.IsValid()); CHECK(graph.GetPasses().empty()); CHECK(graph.GetResources().size() == 3);
}
TEST_CASE("TaaFeature: cached topology switches on off on without stale output [renderfeatures]")
{
    Fixture fixture;
    for (u32 i = 0; i < 3; ++i)
    {
        fixture.packet.enabled = i != 1;
        Memory::LinearAllocator scratch(64 * 1024); RG::RenderGraph graph(scratch); GraphTextureRef output;
        REQUIRE(fixture.Build(graph, scratch, output).success);
        CHECK(graph.GetPasses().size() == (i == 1 ? 0 : 1));
        CHECK(output.handle.index == (i == 1 ? 1 : 5));
        CHECK(output.binding.texture == (i == 1 ? fixture.packet.sources[0].texture : fixture.packet.current.texture));
    }
}
TEST_CASE("TaaFeature: cold preparation invalidates without requiring a Vulkan device [renderfeatures]")
{
    PostProcessSubsystem native; auto state = std::make_shared<TaaViewState>(); state->history.Commit(6, 1);
    state->recorded = true;
    const auto packet = native.PrepareTaaBindings(state, 7, 1, Mat4{1.0f}, 0.1f, true);
    CHECK(packet.enabled); CHECK_FALSE(packet.pipeline); CHECK_FALSE(state->recorded); CHECK_FALSE(state->history.valid);
    CHECK(packet.state == state); CHECK(state->shaderGeneration != 0);
    CHECK_FALSE(native.PrepareTaaBindings({}, 7, 1, Mat4{1.0f}, 0.1f, false).enabled);
    native.InvalidateTaaView({99});
}
TEST_CASE("TaaFeature: recorded graph retains domain state until recording references retire [renderfeatures]")
{
    Fixture fixture; std::weak_ptr<TaaViewState> retained = fixture.packet.state;
    {
        Memory::LinearAllocator scratch(64 * 1024); RG::RenderGraph graph(scratch); GraphTextureRef output;
        REQUIRE(fixture.Build(graph, scratch, output).success);
        fixture.packet.state.reset(); CHECK_FALSE(retained.expired());
    }
    CHECK(retained.expired());
}

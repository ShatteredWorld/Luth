#include <doctest/doctest.h>
#include "luth/core/diagnostics/Log.h"
#include "luth/renderer/features/SelectionMaskFeature.h"
#include "luth/renderer/features/RenderPipelineCompiler.h"
#include "luth/renderer/subsystems/EditorOverlaysSubsystem.h"
#include "luth/renderer/CameraParams.h"
#include "luth/renderer/draw/DrawList.h"
#include "luth/core/RenderSnapshot.h"
#include "luth/memory/LinearAllocator.h"
using namespace Luth;
namespace
{
    template<class T> T Native(u64 value) { return reinterpret_cast<T>(static_cast<uintptr_t>(value)); }
    struct Fixture
    {
        EditorOverlaysSubsystem native;
        SelectionMaskBindings bindings;
        std::unique_ptr<CompiledRenderPipeline> pipeline;
        Fixture()
        {
            bindings.enabled = true; bindings.width = 640; bindings.height = 480;
            bindings.images = {Native<VkImage>(1), Native<VkImage>(2)};
            bindings.views = {Native<VkImageView>(3), Native<VkImageView>(4)};
            bindings.pipeline = Native<VkPipeline>(5); bindings.layout = Native<VkPipelineLayout>(6);
            bindings.sets.fill(Native<VkDescriptorSet>(7));
            bindings.state = std::make_shared<EditorOverlayViewState>();
            bindings.state->sources[0] = std::shared_ptr<Texture>(Native<Texture*>(8), [](Texture*){});
            bindings.state->sources[1] = std::shared_ptr<Texture>(Native<Texture*>(9), [](Texture*){});
            RenderPipelineDefinition definition; definition.AddFeature<SelectionMaskFeature>(native);
            PipelineInputContract inputs; inputs.resources = {{SelectionMaskResources::Bindings}};
            auto compiled = RenderPipelineCompiler{}.Compile(std::move(definition), {}, inputs);
            REQUIRE(compiled.pipeline); pipeline = std::move(compiled.pipeline);
        }
        void Draw(bool skinned)
        {
            SelectionMaskDraw draw; draw.vertex = Native<VkBuffer>(10); draw.index = Native<VkBuffer>(11);
            draw.vertexOwner = std::shared_ptr<VertexBuffer>(Native<VertexBuffer*>(12), [](VertexBuffer*){});
            draw.indexOwner = std::shared_ptr<IndexBuffer>(Native<IndexBuffer*>(13), [](IndexBuffer*){});
            draw.indexCount = 3; draw.skinned = skinned; bindings.draws.push_back(std::move(draw));
        }
        PipelineBuildResult Build(RG::RenderGraph& graph, Memory::LinearAllocator& scratch,
            GraphTextureRef& mask, GraphTextureRef& depth, bool packetPresent = true)
        {
            const SelectionMaskBindingRef ref{packetPresent ? &bindings : nullptr};
            const std::array inputs{RenderInputBinding::Present(SelectionMaskResources::Bindings, ref)};
            FrameRenderInputs frame; frame.resources = inputs;
            ViewRenderInputs view; view.id = {1}; view.width = 640; view.height = 480;
            const std::array outputs{RenderOutputBinding::Capture(RenderResources::SelectionMask, mask),
                RenderOutputBinding::Capture(RenderResources::SelectionDepth, depth)};
            return pipeline->Build(graph, frame, view, scratch, outputs);
        }
    };
}
TEST_CASE("SelectionMaskFeature: empty and cold selection clear both outputs [renderfeatures]")
{
    Fixture f;
    SUBCASE("empty selection") {}
    SUBCASE("cold pipeline") { f.bindings.pipeline = VK_NULL_HANDLE; f.bindings.layout = VK_NULL_HANDLE; }
    Memory::LinearAllocator scratch(64 * 1024); RG::RenderGraph graph(scratch); GraphTextureRef mask, depth;
    REQUIRE(f.Build(graph, scratch, mask, depth).success); REQUIRE(graph.GetPasses().size() == 1);
    REQUIRE(graph.GetResources().size() == 2);
    const auto& pass = graph.GetPasses()[0]; CHECK(pass.name == "SelectionMaskPass");
    CHECK_FALSE(pass.isCompute); CHECK(pass.queueFamily == RG::QueueFamily::Graphics); CHECK(pass.reads.empty());
    REQUIRE(pass.colorAttachments.size() == 1); const auto& color = pass.colorAttachments[0];
    CHECK(color.loadOp == VK_ATTACHMENT_LOAD_OP_CLEAR); CHECK(color.storeOp == VK_ATTACHMENT_STORE_OP_STORE);
    CHECK(color.clearValue.color.float32[0] == 0); CHECK(color.clearValue.color.float32[3] == 0);
    REQUIRE(pass.hasDepth); CHECK(pass.depthAttachment.loadOp == VK_ATTACHMENT_LOAD_OP_CLEAR);
    CHECK(pass.depthAttachment.storeOp == VK_ATTACHMENT_STORE_OP_STORE); CHECK(pass.depthAttachment.clearValue.depthStencil.depth == 1);
    CHECK(mask.handle.index == 1); CHECK(mask.handle.version == 1); CHECK(mask.binding.texture == Native<const Texture*>(8));
    CHECK(depth.handle.index == 2); CHECK(depth.handle.version == 1); CHECK(depth.binding.texture == Native<const Texture*>(9));
    CHECK(graph.GetResources()[0].desc.format == RG::TextureFormat::RGBA8_Unorm);
    CHECK(graph.GetResources()[1].desc.format == RG::TextureFormat::D32_Float);
    graph.Compile(); CHECK_FALSE(graph.GetPasses()[0].culled);
}
TEST_CASE("SelectionMaskFeature: disabled graph publishes absence without imports [renderfeatures]")
{
    Fixture f; f.bindings.enabled = false; f.bindings.state.reset();
    for (u32 i = 0; i < 2; ++i)
    {
        Memory::LinearAllocator scratch(64 * 1024); RG::RenderGraph graph(scratch);
        GraphTextureRef mask{{55, 1}, {}}, depth{{56, 1}, {}};
        REQUIRE(f.Build(graph, scratch, mask, depth).success);
        CHECK_FALSE(mask.handle.IsValid()); CHECK_FALSE(depth.handle.IsValid());
        CHECK(graph.GetPasses().empty()); CHECK(graph.GetResources().empty());
    }
}
TEST_CASE("SelectionMaskFeature: frozen jobs retain state and mesh owners [renderfeatures]")
{
    Fixture f; f.Draw(false); f.Draw(true);
    SUBCASE("rigid fallback for cold skinned pipeline") {}
    SUBCASE("skinned variant") { f.bindings.skinnedPipeline = Native<VkPipeline>(14); f.bindings.skinnedLayout = Native<VkPipelineLayout>(15); }
    std::weak_ptr<EditorOverlayViewState> state = f.bindings.state;
    std::weak_ptr<VertexBuffer> vertex = f.bindings.draws[0].vertexOwner;
    std::weak_ptr<IndexBuffer> index = f.bindings.draws[1].indexOwner;
    {
        Memory::LinearAllocator scratch(64 * 1024); RG::RenderGraph graph(scratch); GraphTextureRef mask, depth;
        REQUIRE(f.Build(graph, scratch, mask, depth).success);
        f.bindings.state.reset(); f.bindings.draws.clear();
        CHECK_FALSE(state.expired()); CHECK_FALSE(vertex.expired()); CHECK_FALSE(index.expired());
        REQUIRE(graph.GetPasses().size() == 1); CHECK(graph.GetResources().size() == 2);
    }
    CHECK(state.expired()); CHECK(vertex.expired()); CHECK(index.expired());
}
TEST_CASE("SelectionMaskFeature: malformed native packets fail before registering imports [renderfeatures]")
{
    Fixture f; bool present = true; f.Draw(true);
    SUBCASE("packet") { present = false; }
    SUBCASE("extent") { f.bindings.width = 641; }
    SUBCASE("state") { f.bindings.state.reset(); }
    SUBCASE("source") { f.bindings.state->sources[1].reset(); }
    SUBCASE("image") { f.bindings.images[0] = VK_NULL_HANDLE; }
    SUBCASE("alias") { f.bindings.images[0] = f.bindings.images[1]; }
    SUBCASE("view") { f.bindings.views[1] = VK_NULL_HANDLE; }
    SUBCASE("pipeline") { f.bindings.pipeline = VK_NULL_HANDLE; }
    SUBCASE("set") { f.bindings.sets[4] = VK_NULL_HANDLE; }
    SUBCASE("vertex") { f.bindings.draws[0].vertex = VK_NULL_HANDLE; }
    SUBCASE("owner") { f.bindings.draws[0].indexOwner.reset(); }
    SUBCASE("skinned layout") { f.bindings.skinnedPipeline = Native<VkPipeline>(14); }
    Memory::LinearAllocator scratch(64 * 1024); RG::RenderGraph graph(scratch); GraphTextureRef mask{{55, 1}, {}}, depth{{56, 1}, {}};
    CHECK_FALSE(f.Build(graph, scratch, mask, depth, present).success);
    CHECK_FALSE(mask.handle.IsValid()); CHECK_FALSE(depth.handle.IsValid());
    CHECK(graph.GetPasses().empty()); CHECK(graph.GetResources().empty());
}
TEST_CASE("SelectionMaskFeature: duplicate image import is rejected [renderfeatures]")
{
    Fixture f; Memory::LinearAllocator scratch(64 * 1024); RG::RenderGraph graph(scratch);
    RG::TextureDesc desc; desc.width = 640; desc.height = 480; desc.format = RG::TextureFormat::RGBA8_Unorm;
    graph.ImportResource(desc, (void*)f.bindings.images[0], (void*)f.bindings.views[0], RG::ResourceState::Undefined);
    GraphTextureRef mask, depth; CHECK_FALSE(f.Build(graph, scratch, mask, depth).success);
    CHECK(graph.GetResources().size() == 1); CHECK(graph.GetPasses().empty());
}
TEST_CASE("SelectionMaskFeature: disabled preparation and missing state are headless [renderfeatures]")
{
    EditorOverlaysSubsystem native; CameraParams camera; DrawList draws; RenderSnapshot snapshot;
    const auto disabled = native.PrepareSelectionMaskBindings({}, {}, camera, {}, draws, snapshot, false);
    CHECK_FALSE(disabled.enabled); CHECK_FALSE(disabled.state); CHECK(disabled.draws.empty());
    CHECK_THROWS_AS(native.PrepareSelectionMaskBindings({}, {}, camera, {}, draws, snapshot, true), std::invalid_argument);
}

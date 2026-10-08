#include <doctest/doctest.h>
#include "luth/core/diagnostics/Log.h"
#include "luth/renderer/features/OutlineFeature.h"
#include "luth/renderer/features/SelectionMaskFeature.h"
#include "luth/renderer/features/RenderPipelineCompiler.h"
#include "luth/renderer/subsystems/EditorOverlaysSubsystem.h"
#include "luth/renderer/CameraParams.h"
#include "luth/memory/LinearAllocator.h"
using namespace Luth;
namespace
{
    template<class T> T Native(u64 value) { return reinterpret_cast<T>(static_cast<uintptr_t>(value)); }
    struct Fixture
    {
        EditorOverlaysSubsystem native;
        OutlineBindings bindings;
        std::unique_ptr<CompiledRenderPipeline> pipeline;
        Fixture()
        {
            bindings.enabled = true; bindings.pipeline = Native<VkPipeline>(1);
            bindings.layout = Native<VkPipelineLayout>(2); bindings.set = Native<VkDescriptorSet>(3);
            bindings.state = std::make_shared<EditorOverlayViewState>(); bindings.state->outlineSet = bindings.set;
            for (u32 i = 0; i < 3; ++i)
                bindings.state->sources[i] = std::shared_ptr<Texture>(Native<Texture*>(10 + i), [](Texture*){});
            bindings.parameters = MakeOutlineParameters(CameraParams{}, 640, 480);
            RenderPipelineDefinition definition; definition.AddFeature<OutlineFeature>(native);
            PipelineInputContract inputs;
            inputs.resources = {{RenderResources::VisualizedLDR}, {OutlineResources::Bindings},
                {RenderResources::SelectionMask, ResourceOutputPresence::Optional},
                {RenderResources::SelectionDepth, ResourceOutputPresence::Optional},
                {RenderResources::LitDepth, ResourceOutputPresence::Optional}};
            auto compiled = RenderPipelineCompiler{}.Compile(std::move(definition), {}, inputs);
            REQUIRE(compiled.pipeline); pipeline = std::move(compiled.pipeline);
        }
        PipelineBuildResult Build(RG::RenderGraph& graph, Memory::LinearAllocator& scratch,
            GraphTextureRef& output, bool maskPresent = true, bool selectedDepthPresent = true,
            bool depthPresent = true, bool packetPresent = true, u32 width = 640, bool feedback = false)
        {
            RG::TextureDesc desc; desc.width = width; desc.height = 480; desc.format = RG::TextureFormat::RGBA8_Unorm;
            const GraphTextureRef color{graph.ImportResource(desc, (void*)20, (void*)21, RG::ResourceState::ShaderResource),
                {Native<const Texture*>(9)}};
            GraphTextureRef mask{graph.ImportResource(desc, (void*)22, (void*)23, RG::ResourceState::Undefined),
                {Native<const Texture*>(10)}};
            desc.format = RG::TextureFormat::D32_Float;
            const GraphTextureRef selectedDepth{graph.ImportResource(desc, (void*)24, (void*)25, RG::ResourceState::Undefined),
                {Native<const Texture*>(11)}};
            const GraphTextureRef depth{graph.ImportResource(desc, (void*)26, (void*)27, RG::ResourceState::Undefined),
                {Native<const Texture*>(12)}};
            if (feedback) mask.handle = color.handle;
            const OutlineBindingRef binding{packetPresent ? &bindings : nullptr};
            const std::array inputs{RenderInputBinding::Present(RenderResources::VisualizedLDR, color),
                RenderInputBinding::Present(OutlineResources::Bindings, binding),
                maskPresent ? RenderInputBinding::Present(RenderResources::SelectionMask, mask) : RenderInputBinding::Absent(RenderResources::SelectionMask),
                selectedDepthPresent ? RenderInputBinding::Present(RenderResources::SelectionDepth, selectedDepth) : RenderInputBinding::Absent(RenderResources::SelectionDepth),
                depthPresent ? RenderInputBinding::Present(RenderResources::LitDepth, depth) : RenderInputBinding::Absent(RenderResources::LitDepth)};
            FrameRenderInputs frame; frame.resources = inputs;
            ViewRenderInputs view; view.id = {1}; view.width = 640; view.height = 480;
            const std::array exports{RenderOutputBinding::Capture(RenderResources::OutlinedLDR, output)};
            return pipeline->Build(graph, frame, view, scratch, exports);
        }
    };
}
TEST_CASE("OutlineFeature: in-place LOAD blend consumes typed selection inputs [renderfeatures]")
{
    Fixture f; Memory::LinearAllocator scratch(64 * 1024); RG::RenderGraph graph(scratch); GraphTextureRef output;
    REQUIRE(f.Build(graph, scratch, output).success); REQUIRE(graph.GetPasses().size() == 1);
    CHECK(graph.GetResources().size() == 4); const auto& pass = graph.GetPasses()[0];
    CHECK(pass.name == "OutlinePass"); CHECK_FALSE(pass.isCompute); CHECK(pass.queueFamily == RG::QueueFamily::Graphics);
    REQUIRE(pass.colorAttachments.size() == 1); CHECK(pass.colorAttachments[0].loadOp == VK_ATTACHMENT_LOAD_OP_LOAD);
    CHECK(pass.colorAttachments[0].storeOp == VK_ATTACHMENT_STORE_OP_STORE); CHECK_FALSE(pass.hasDepth);
    REQUIRE(pass.reads.size() == 3); CHECK(pass.reads[0].index == 2); CHECK(pass.reads[1].index == 3); CHECK(pass.reads[2].index == 4);
    CHECK(output.handle.index == 1); CHECK(output.handle.version == 1); CHECK(output.binding.texture == Native<const Texture*>(9));
    std::weak_ptr<EditorOverlayViewState> retained = f.bindings.state; f.bindings.state.reset(); CHECK_FALSE(retained.expired());
    graph.Compile(); CHECK_FALSE(graph.GetPasses()[0].culled); CHECK_FALSE(graph.GetPasses()[0].preBarriers.empty());
}
TEST_CASE("OutlineFeature: disabled PT cold and absent selection publish fresh LDR aliases [renderfeatures]")
{
    Fixture f; bool selection = true, depth = true;
    SUBCASE("disabled") { f.bindings.enabled = false; f.bindings.state.reset(); }
    SUBCASE("PT") { f.bindings.enabled = false; selection = false; depth = false; }
    SUBCASE("cold") { f.bindings.pipeline = VK_NULL_HANDLE; }
    SUBCASE("absent selection") { selection = false; }
    for (u32 i = 0; i < 2; ++i)
    {
        Memory::LinearAllocator scratch(64 * 1024); RG::RenderGraph graph(scratch); GraphTextureRef output{{55, 1}, {}};
        REQUIRE(f.Build(graph, scratch, output, selection, selection, depth).success);
        CHECK(graph.GetPasses().empty()); CHECK(graph.GetResources().size() == 4);
        CHECK(output.handle.index == 1); CHECK(output.handle.version == 0); CHECK(output.binding.texture == Native<const Texture*>(9));
    }
}
TEST_CASE("OutlineFeature: invalid stage and native packet fail before registration [renderfeatures]")
{
    Fixture f; bool mask = true, selectedDepth = true, depth = true, packet = true, feedback = false; u32 width = 640;
    SUBCASE("packet") { packet = false; }
    SUBCASE("layout") { f.bindings.layout = VK_NULL_HANDLE; }
    SUBCASE("set") { f.bindings.set = Native<VkDescriptorSet>(99); }
    SUBCASE("state") { f.bindings.state.reset(); }
    SUBCASE("source") { f.bindings.state->sources[0].reset(); }
    SUBCASE("wrong depth source") { f.bindings.state->sources[2] = f.bindings.state->sources[1]; }
    SUBCASE("mask absent") { mask = false; }
    SUBCASE("selection depth absent") { selectedDepth = false; }
    SUBCASE("scene depth absent") { depth = false; }
    SUBCASE("extent") { width = 641; }
    SUBCASE("texel size") { f.bindings.parameters.texelSizeX = 0; }
    SUBCASE("feedback") { feedback = true; }
    Memory::LinearAllocator scratch(64 * 1024); RG::RenderGraph graph(scratch); GraphTextureRef output{{55, 1}, {}};
    CHECK_FALSE(f.Build(graph, scratch, output, mask, selectedDepth, depth, packet, width, feedback).success);
    CHECK_FALSE(output.handle.IsValid()); CHECK(graph.GetPasses().empty()); CHECK(graph.GetResources().size() == 4);
}
TEST_CASE("OutlineFeature: settings and texel dimensions freeze without device state [renderfeatures]")
{
    CameraParams camera; camera.outlineWidth = 3.5f; camera.outlineColor = {0.1f, 0.2f, 0.3f, 0.4f}; camera.outlineOccludedAlpha = 0.7f;
    const auto frozen = MakeOutlineParameters(camera, 640, 480); camera.outlineWidth = 99;
    CHECK(frozen.width == 3.5f); CHECK(frozen.texelSizeX == 1.0f / 640); CHECK(frozen.texelSizeY == 1.0f / 480);
    CHECK(frozen.colorR == 0.1f); CHECK(frozen.colorG == 0.2f); CHECK(frozen.colorB == 0.3f); CHECK(frozen.colorA == 0.4f);
    CHECK(frozen.occludedAlpha == 0.7f);
    CHECK_THROWS_AS(MakeOutlineParameters(camera, 0, 480), std::invalid_argument);
    CHECK_THROWS_AS(MakeOutlineParameters(camera, 640, 0), std::invalid_argument);
    EditorOverlaysSubsystem native;
    const auto disabled = native.PrepareOutlineBindings({}, camera, 640, 480, false);
    CHECK_FALSE(disabled.enabled); CHECK_FALSE(disabled.pipeline); CHECK_FALSE(disabled.state);
    const auto cold = native.PrepareOutlineBindings({}, camera, 640, 480, true);
    CHECK(cold.enabled); CHECK_FALSE(cold.pipeline); CHECK_FALSE(cold.state);
}
TEST_CASE("OutlineFeature: compiled selection producer supplies current versions and barriers [renderfeatures]")
{
    Fixture f; SelectionMaskBindings selection;
    selection.enabled = true; selection.state = f.bindings.state; selection.width = 640; selection.height = 480;
    selection.images = {Native<VkImage>(22), Native<VkImage>(24)};
    selection.views = {Native<VkImageView>(23), Native<VkImageView>(25)};
    RenderPipelineDefinition definition;
    const auto outlineId = definition.AddFeature<OutlineFeature>(f.native);
    const auto selectionId = definition.AddFeature<SelectionMaskFeature>(f.native);
    PipelineInputContract inputs; inputs.resources = {{RenderResources::VisualizedLDR}, {OutlineResources::Bindings},
        {SelectionMaskResources::Bindings}, {RenderResources::LitDepth, ResourceOutputPresence::Optional}};
    auto compiled = RenderPipelineCompiler{}.Compile(std::move(definition), {}, inputs);
    REQUIRE(compiled.pipeline); REQUIRE(compiled.pipeline->FeatureOrder().size() == 2);
    CHECK(compiled.pipeline->FeatureOrder()[0] == selectionId); CHECK(compiled.pipeline->FeatureOrder()[1] == outlineId);
    Memory::LinearAllocator scratch(64 * 1024); RG::RenderGraph graph(scratch);
    RG::TextureDesc desc; desc.width = 640; desc.height = 480; desc.format = RG::TextureFormat::RGBA8_Unorm;
    const GraphTextureRef color{graph.ImportResource(desc, (void*)20, (void*)21, RG::ResourceState::ShaderResource),
        {Native<const Texture*>(9)}};
    desc.format = RG::TextureFormat::D32_Float;
    const GraphTextureRef depth{graph.ImportResource(desc, (void*)26, (void*)27, RG::ResourceState::Undefined),
        {Native<const Texture*>(12)}};
    const OutlineBindingRef outlineRef{&f.bindings}; const SelectionMaskBindingRef selectionRef{&selection};
    const std::array resources{RenderInputBinding::Present(RenderResources::VisualizedLDR, color),
        RenderInputBinding::Present(RenderResources::LitDepth, depth),
        RenderInputBinding::Present(OutlineResources::Bindings, outlineRef),
        RenderInputBinding::Present(SelectionMaskResources::Bindings, selectionRef)};
    FrameRenderInputs frame; frame.resources = resources; ViewRenderInputs view; view.id = {1}; view.width = 640; view.height = 480;
    GraphTextureRef output; const std::array exports{RenderOutputBinding::Capture(RenderResources::OutlinedLDR, output)};
    REQUIRE(compiled.pipeline->Build(graph, frame, view, scratch, exports).success);
    REQUIRE(graph.GetPasses().size() == 2); CHECK(graph.GetResources().size() == 4);
    CHECK(graph.GetPasses()[0].name == "SelectionMaskPass"); CHECK(graph.GetPasses()[1].name == "OutlinePass");
    REQUIRE(graph.GetPasses()[1].reads.size() == 3);
    CHECK(graph.GetPasses()[1].reads[0].index == 3); CHECK(graph.GetPasses()[1].reads[0].version == 1);
    CHECK(graph.GetPasses()[1].reads[1].index == 4); CHECK(graph.GetPasses()[1].reads[1].version == 1);
    CHECK(output.handle.index == color.handle.index); CHECK(output.handle.version == 1);
    graph.Compile(); CHECK_FALSE(graph.GetPasses()[0].culled); CHECK_FALSE(graph.GetPasses()[1].culled);
    bool maskBarrier = false, depthBarrier = false;
    for (const auto& barrier : graph.GetPasses()[1].preBarriers)
    {
        if (barrier.resource.index == 3 && barrier.after == RG::ResourceState::ShaderResource) maskBarrier = true;
        if (barrier.resource.index == 4 && barrier.after == RG::ResourceState::ShaderResource) depthBarrier = true;
    }
    CHECK(maskBarrier); CHECK(depthBarrier);
}

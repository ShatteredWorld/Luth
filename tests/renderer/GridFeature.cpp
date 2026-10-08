#include <doctest/doctest.h>
#include "luth/core/diagnostics/Log.h"
#include "luth/renderer/features/GridFeature.h"
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
        GridBindings bindings;
        std::unique_ptr<CompiledRenderPipeline> pipeline;
        Fixture()
        {
            bindings.enabled = true; bindings.pipeline = Native<VkPipeline>(1);
            bindings.layout = Native<VkPipelineLayout>(2); bindings.set = Native<VkDescriptorSet>(3);
            bindings.depth = {Native<const Texture*>(10)};
            bindings.state = std::make_shared<EditorOverlayViewState>();
            bindings.state->gridSets.fill(bindings.set);
            bindings.state->sources[2] = std::shared_ptr<Texture>(Native<Texture*>(10), [](Texture*) {});
            RenderPipelineDefinition definition; definition.AddFeature<GridFeature>(native);
            PipelineInputContract inputs;
            inputs.resources = {{RenderResources::ResolvedHDR}, {GridResources::Bindings},
                {RenderResources::LitDepth, ResourceOutputPresence::Optional},
                {RenderResources::BloomOutput, ResourceOutputPresence::Optional}};
            auto compiled = RenderPipelineCompiler{}.Compile(std::move(definition), {}, inputs);
            REQUIRE(compiled.pipeline); pipeline = std::move(compiled.pipeline);
        }
        PipelineBuildResult Build(RG::RenderGraph& graph, Memory::LinearAllocator& scratch,
            GraphTextureRef& output, bool depthPresent = true, bool packetPresent = true, u32 width = 640)
        {
            RG::TextureDesc desc; desc.name = "HDR"; desc.width = width; desc.height = 480;
            desc.format = RG::TextureFormat::RGBA16_Float;
            const GraphTextureRef color{graph.ImportResource(desc, (void*)5, (void*)6, RG::ResourceState::ShaderResource),
                {Native<const Texture*>(7)}};
            desc.name = "Depth"; desc.format = RG::TextureFormat::D32_Float;
            const GraphTextureRef depth{graph.ImportResource(desc, (void*)8, (void*)9, RG::ResourceState::Undefined),
                {Native<const Texture*>(10)}};
            const GridBindingRef binding{packetPresent ? &bindings : nullptr};
            const std::array resources{RenderInputBinding::Present(RenderResources::ResolvedHDR, color),
                RenderInputBinding::Present(GridResources::Bindings, binding),
                depthPresent ? RenderInputBinding::Present(RenderResources::LitDepth, depth)
                    : RenderInputBinding::Absent(RenderResources::LitDepth),
                RenderInputBinding::Absent(RenderResources::BloomOutput)};
            FrameRenderInputs frame; frame.resources = resources;
            ViewRenderInputs view; view.id = {1}; view.width = 640; view.height = 480;
            const std::array exports{RenderOutputBinding::Capture(RenderResources::GridHDR, output)};
            return pipeline->Build(graph, frame, view, scratch, exports);
        }
    };
    class BloomProbe final : public IRenderFeature
    {
    public:
        FeatureInfo Describe() const override
        {
            FeatureInfo info; info.name = "BloomProbe"; info.phase = FeaturePhase::AfterAsync;
            info.resources.reads = {{RenderResources::ResolvedHDR}};
            info.resources.writes = {{RenderResources::BloomOutput, ResourceOutputPresence::Optional}};
            return info;
        }
        FeatureFrameDecision Evaluate(const FeaturePrepareContext&) const override { return {}; }
        void Build(RG::RenderGraph&, RenderFeatureContext& ctx) override { ctx.resources.PublishAbsent(RenderResources::BloomOutput); }
    };
}
TEST_CASE("GridFeature: in-place LOAD blend preserves native graph routing [renderfeatures]")
{
    Fixture f; Memory::LinearAllocator scratch(64 * 1024); RG::RenderGraph graph(scratch); GraphTextureRef output;
    REQUIRE(f.Build(graph, scratch, output).success);
    REQUIRE(graph.GetPasses().size() == 1); CHECK(graph.GetResources().size() == 2);
    const auto& pass = graph.GetPasses()[0]; CHECK(pass.name == "GridPass");
    CHECK_FALSE(pass.isCompute); CHECK(pass.queueFamily == RG::QueueFamily::Graphics);
    REQUIRE(pass.colorAttachments.size() == 1); CHECK(pass.colorAttachments[0].loadOp == VK_ATTACHMENT_LOAD_OP_LOAD);
    CHECK(pass.colorAttachments[0].storeOp == VK_ATTACHMENT_STORE_OP_STORE); CHECK_FALSE(pass.hasDepth);
    REQUIRE(pass.reads.size() == 1); CHECK(pass.reads[0].index == 2);
    CHECK(output.handle.index == 1); CHECK(output.handle.version == 1); CHECK(output.binding.texture == Native<const Texture*>(7));
    std::weak_ptr<EditorOverlayViewState> retained = f.bindings.state; f.bindings.state.reset();
    CHECK_FALSE(retained.expired());
    graph.Compile(); CHECK_FALSE(graph.GetPasses()[0].culled); CHECK_FALSE(graph.GetPasses()[0].preBarriers.empty());
}
TEST_CASE("GridFeature: disabled PT and cold pipeline publish fresh HDR aliases [renderfeatures]")
{
    Fixture f; bool depth = true;
    SUBCASE("disabled") { f.bindings.enabled = false; }
    SUBCASE("PT without depth") { f.bindings.enabled = false; depth = false; }
    SUBCASE("cold pipeline") { f.bindings.pipeline = VK_NULL_HANDLE; }
    for (u32 i = 0; i < 2; ++i)
    {
        Memory::LinearAllocator scratch(64 * 1024); RG::RenderGraph graph(scratch);
        GraphTextureRef output{{55, 1}, {Native<const Texture*>(99)}};
        REQUIRE(f.Build(graph, scratch, output, depth).success); CHECK(graph.GetPasses().empty());
        CHECK(output.handle.index == 1); CHECK(output.handle.version == 0); CHECK(output.binding.texture == Native<const Texture*>(7));
    }
}
TEST_CASE("GridFeature: invalid stage and native inputs fail before registration [renderfeatures]")
{
    Fixture f; bool depth = true, packet = true; u32 width = 640;
    SUBCASE("packet") { packet = false; }
    SUBCASE("layout") { f.bindings.layout = VK_NULL_HANDLE; }
    SUBCASE("set") { f.bindings.set = Native<VkDescriptorSet>(99); }
    SUBCASE("state") { f.bindings.state.reset(); }
    SUBCASE("depth absent") { depth = false; }
    SUBCASE("depth mismatch") { f.bindings.depth.texture = Native<const Texture*>(99); }
    SUBCASE("subresource") { f.bindings.depth.baseMip = 1; }
    SUBCASE("extent") { width = 641; }
    Memory::LinearAllocator scratch(64 * 1024); RG::RenderGraph graph(scratch); GraphTextureRef output{{55, 1}, {}};
    CHECK_FALSE(f.Build(graph, scratch, output, depth, packet, width).success);
    CHECK_FALSE(output.handle.IsValid()); CHECK(graph.GetPasses().empty());
}
TEST_CASE("GridFeature: bloom consumer precedes alias mutation independent of declaration order [renderfeatures]")
{
    EditorOverlaysSubsystem native; RenderPipelineDefinition definition;
    const auto grid = definition.AddFeature<GridFeature>(native);
    const auto bloom = definition.AddFeature<BloomProbe>();
    PipelineInputContract inputs; inputs.resources = {{RenderResources::ResolvedHDR}, {GridResources::Bindings}};
    auto compiled = RenderPipelineCompiler{}.Compile(std::move(definition), {}, inputs);
    REQUIRE(compiled.pipeline); REQUIRE(compiled.pipeline->FeatureOrder().size() == 2);
    CHECK(compiled.pipeline->FeatureOrder()[0] == bloom); CHECK(compiled.pipeline->FeatureOrder()[1] == grid);
}
TEST_CASE("GridFeature: disabled and cold preparation need no device or view state [renderfeatures]")
{
    EditorOverlaysSubsystem native; CameraParams camera;
    const auto disabled = native.PrepareGridBindings({}, camera, {0.1f, 0.2f}, 7, false);
    CHECK_FALSE(disabled.enabled); CHECK_FALSE(disabled.pipeline); CHECK_FALSE(disabled.state);
    const auto cold = native.PrepareGridBindings({}, camera, {}, 7, true);
    CHECK(cold.enabled); CHECK_FALSE(cold.pipeline); CHECK_FALSE(cold.state);
}

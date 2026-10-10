#include <doctest/doctest.h>
#include "luth/core/diagnostics/Log.h"
#include "luth/renderer/features/SlimGBufferFeature.h"
#include "luth/renderer/features/DepthPrepassFeature.h"
#include "luth/renderer/features/RenderPipelineCompiler.h"
#include "luth/renderer/subsystems/GeometrySubsystem.h"
#include "luth/renderer/draw/DrawList.h"
#include "luth/core/RenderSnapshot.h"
#include "luth/memory/LinearAllocator.h"

using namespace Luth;
using namespace Luth::RenderResources;

namespace
{
    template<class T> T Native(u64 value) { return reinterpret_cast<T>(static_cast<uintptr_t>(value)); }
    struct Fixture
    {
        GeometrySubsystem native;
        DepthPrepassBindings depthBindings{};
        SlimGBufferBindings slimBindings{};
        std::unique_ptr<CompiledRenderPipeline> pipeline;
        DrawList draws;
        RenderSnapshot snapshot;
        Memory::GPUSubRegion slice{Native<VkBuffer>(1), 8192, 819200};
        std::array<GraphTextureRef, 5> outputs;
        GraphTextureRef prepass;
        bool framePresent = true, bindingsPresent = true, duplicateTarget = false;
        u32 targetWidth = 640;
        RG::TextureFormat normalFormat = RG::TextureFormat::RG16_Float;
        Fixture()
        {
            slimBindings.opaque.rigid = Native<VkPipeline>(2);
            slimBindings.opaque.rigidLayout = Native<VkPipelineLayout>(3);
            slimBindings.opaque.sets.fill(Native<VkDescriptorSet>(4));
            RenderPipelineDefinition definition;
            const auto slim = definition.AddFeature<SlimGBufferFeature>(native);
            const auto depth = definition.AddFeature<DepthPrepassFeature>(native);
            PipelineInputContract inputs;
            inputs.resources = {{CameraVisibleDraws}, {DepthPrepassResources::Target}, {DepthPrepassResources::Bindings},
                {SlimGBufferResources::NormalTarget}, {SlimGBufferResources::RoughnessTarget},
                {SlimGBufferResources::MotionTarget}, {SlimGBufferResources::MaterialTarget}, {SlimGBufferResources::Bindings}};
            inputs.capabilities = {&DeformationResources::DeformedGeometry};
            auto compiled = RenderPipelineCompiler{}.Compile(std::move(definition), {}, inputs);
            REQUIRE(compiled.pipeline);
            CHECK(compiled.pipeline->FeatureOrder()[0] == depth);
            CHECK(compiled.pipeline->FeatureOrder()[1] == slim);
            pipeline = std::move(compiled.pipeline);
        }
        PipelineBuildResult Build(RG::RenderGraph& graph, Memory::LinearAllocator& scratch, u32 viewIndex = 0)
        {
            const auto import = [&](const char* name, RG::TextureFormat format, u32 width, u64 identity) {
                RG::TextureDesc desc;
                desc.name = name; desc.width = width; desc.height = 480; desc.format = format;
                return GraphTextureRef{graph.ImportResource(desc, (void*)(uintptr_t)identity,
                    (void*)(uintptr_t)identity, RG::ResourceState::Undefined), {Native<const Texture*>(identity)}};
            };
            const auto depth = import("SceneDepth", RG::TextureFormat::D32_Float, 640, 10);
            const auto normal = import("SlimNormal", normalFormat, targetWidth, 11);
            const auto roughness = import("SlimRoughness", RG::TextureFormat::R8_Unorm, 640, 12);
            const auto motion = duplicateTarget ? normal : import("SlimMotion", RG::TextureFormat::RG16_Float, 640, 13);
            const auto material = import("SlimMaterialID", RG::TextureFormat::R16_Uint, 640, 14);
            const auto indirect = graph.ImportBuffer({"Indirect", slice.size, VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT},
                (void*)slice.buffer, RG::ResourceState::Undefined);
            const VisibleDrawRange visible{{indirect, {&slice, slice.offset, slice.size}}, viewIndex * 5 * 4096, 0};
            const DepthPrepassBindingRef depthRef{&depthBindings};
            const SlimGBufferBindingRef slimRef{bindingsPresent ? &slimBindings : nullptr};
            const std::array bindings{RenderInputBinding::Present(CameraVisibleDraws, visible),
                RenderInputBinding::Present(DepthPrepassResources::Target, depth),
                RenderInputBinding::Present(DepthPrepassResources::Bindings, depthRef),
                RenderInputBinding::Present(SlimGBufferResources::NormalTarget, normal),
                RenderInputBinding::Present(SlimGBufferResources::RoughnessTarget, roughness),
                RenderInputBinding::Present(SlimGBufferResources::MotionTarget, motion),
                RenderInputBinding::Present(SlimGBufferResources::MaterialTarget, material),
                RenderInputBinding::Present(SlimGBufferResources::Bindings, slimRef)};
            const std::array capabilities{&DeformationResources::DeformedGeometry};
            FrameRenderInputs frame;
            frame.resources = bindings; frame.capabilities = capabilities;
            if (framePresent) { frame.draws = &draws; frame.snapshot = &snapshot; }
            ViewRenderInputs view;
            view.id = {u64(viewIndex) + 1}; view.width = 640; view.height = 480;
            const std::array exports{RenderOutputBinding::Capture(Normal, outputs[0]),
                RenderOutputBinding::Capture(Roughness, outputs[1]), RenderOutputBinding::Capture(MotionVectors, outputs[2]),
                RenderOutputBinding::Capture(MaterialID, outputs[3]), RenderOutputBinding::Capture(SurfaceDepth, outputs[4]),
                RenderOutputBinding::Capture(PrepassDepth, prepass)};
            return pipeline->Build(graph, frame, view, scratch, exports);
        }
    };
    struct PrepassReader final : IRenderFeature
    {
        FeatureInfo Describe() const override
        {
            FeatureInfo info;
            info.name = "PrepassReader";
            info.resources.reads = {{PrepassDepth}};
            return info;
        }
        FeatureFrameDecision Evaluate(const FeaturePrepareContext&) const override { return {}; }
        void Build(RG::RenderGraph&, RenderFeatureContext&) override {}
    };
}

TEST_CASE("SlimGBufferFeature: native graph publishes cutout surface depth and four attachments [renderfeatures]")
{
    Fixture fixture;
    Memory::LinearAllocator scratch(64 * 1024);
    RG::RenderGraph graph(scratch);
    REQUIRE(fixture.Build(graph, scratch).success);
    REQUIRE(graph.GetPasses().size() == 2);
    CHECK(graph.GetPasses()[0].name == "DepthPrepass");
    const auto& pass = graph.GetPasses()[1];
    CHECK(pass.name == "SlimGBufferPass");
    CHECK_FALSE(pass.isCompute);
    CHECK(pass.queueFamily == RG::QueueFamily::Graphics);
    REQUIRE(pass.colorAttachments.size() == 4);
    for (u32 i = 0; i < 4; ++i)
    {
        CHECK(pass.colorAttachments[i].handle == fixture.outputs[i].handle);
        CHECK(pass.colorAttachments[i].loadOp == VK_ATTACHMENT_LOAD_OP_CLEAR);
        CHECK(pass.colorAttachments[i].storeOp == VK_ATTACHMENT_STORE_OP_STORE);
        CHECK(fixture.outputs[i].handle.version == 1);
    }
    CHECK(pass.colorAttachments[0].clearValue.color.float32[0] == 0.5f);
    CHECK(pass.colorAttachments[0].clearValue.color.float32[1] == 0.5f);
    CHECK(pass.colorAttachments[1].clearValue.color.float32[0] == 1.0f);
    CHECK(pass.colorAttachments[2].clearValue.color.float32[0] == 0.0f);
    CHECK(pass.colorAttachments[3].clearValue.color.uint32[0] == 0u);
    CHECK(pass.depthAttachment.loadOp == VK_ATTACHMENT_LOAD_OP_LOAD);
    CHECK(pass.depthAttachment.storeOp == VK_ATTACHMENT_STORE_OP_STORE);
    CHECK(fixture.outputs[4].handle.index == fixture.prepass.handle.index);
    CHECK(fixture.outputs[4].handle.version == fixture.prepass.handle.version + 1);
    CHECK(fixture.outputs[4].binding.texture == fixture.prepass.binding.texture);
    CHECK(pass.depthAttachment.handle == fixture.outputs[4].handle);
    REQUIRE(pass.bufferReads.size() == 1);
    CHECK(pass.bufferReadStates[0] == RG::ResourceState::IndirectRead);
    CHECK(graph.GetResources().size() == 5);
    graph.Compile();
    for (const auto& node : graph.GetPasses()) CHECK_FALSE(node.culled);
}

TEST_CASE("SlimGBufferFeature: invalid state clears every graph-local export [renderfeatures]")
{
    Fixture fixture;
    SUBCASE("extent mismatch") { fixture.targetWidth = 639; }
    SUBCASE("wrong format") { fixture.normalFormat = RG::TextureFormat::R8_Unorm; }
    SUBCASE("missing frame") { fixture.framePresent = false; }
    SUBCASE("missing native packet") { fixture.bindingsPresent = false; }
    SUBCASE("duplicate target") { fixture.duplicateTarget = true; }
    SUBCASE("incomplete descriptors") { fixture.slimBindings.opaque.sets[0] = VK_NULL_HANDLE; }
    Memory::LinearAllocator scratch(64 * 1024);
    RG::RenderGraph graph(scratch);
    const auto result = fixture.Build(graph, scratch);
    CHECK_FALSE(result.success);
    CHECK_FALSE(result.diagnostics.empty());
    for (const auto& output : fixture.outputs) CHECK_FALSE(output.handle.IsValid());
    CHECK_FALSE(fixture.prepass.handle.IsValid());
    CHECK(graph.GetPasses().size() <= 1); // A failed partial graph must never be recorded.
}

TEST_CASE("SlimGBufferFeature: cold pipeline and two views retain deterministic attachments [renderfeatures]")
{
    Fixture fixture;
    fixture.slimBindings = {};
    for (const u32 view : {0u, 1u})
    {
        Memory::LinearAllocator scratch(64 * 1024);
        RG::RenderGraph graph(scratch);
        REQUIRE(fixture.Build(graph, scratch, view).success);
        CHECK(graph.GetResources().size() == 5);
        CHECK(fixture.outputs[4].handle.version == 2);
        CHECK(graph.GetPasses().size() == 2);
    }
}

TEST_CASE("SlimGBufferFeature: opaque and cutout native descriptor packets are frozen [renderfeatures]")
{
    GeometrySubsystem native;
    std::array<VkDescriptorSet, 6> sets;
    sets.fill(Native<VkDescriptorSet>(17));
    const auto prepared = native.PrepareSlimGBufferBindings(sets, true);
    sets.fill(Native<VkDescriptorSet>(33));
    for (u32 i = 0; i < 6; ++i)
    {
        CHECK(prepared.opaque.sets[i] == Native<VkDescriptorSet>(17));
        CHECK(prepared.cutout.sets[i] == Native<VkDescriptorSet>(17));
    }
    CHECK(prepared.opaque.captureDraws);
    CHECK(prepared.cutout.captureDraws);
}

TEST_CASE("SlimGBufferFeature: prepass readers precede the depth mutation [renderfeatures]")
{
    GeometrySubsystem native;
    RenderPipelineDefinition definition;
    const auto slim = definition.AddFeature<SlimGBufferFeature>(native);
    const auto reader = definition.AddFeature<PrepassReader>();
    PipelineInputContract inputs;
    inputs.resources = {{CameraVisibleDraws}, {PrepassDepth}, {SlimGBufferResources::NormalTarget},
        {SlimGBufferResources::RoughnessTarget}, {SlimGBufferResources::MotionTarget},
        {SlimGBufferResources::MaterialTarget}, {SlimGBufferResources::Bindings}};
    inputs.capabilities = {&DeformationResources::DeformedGeometry};
    SUBCASE("stable alias ordering")
    {
        auto compiled = RenderPipelineCompiler{}.Compile(std::move(definition), {}, inputs);
        REQUIRE(compiled.pipeline);
        CHECK(compiled.pipeline->FeatureOrder()[0] == reader);
        CHECK(compiled.pipeline->FeatureOrder()[1] == slim);
    }
    SUBCASE("conflicting order fails")
    {
        definition.Before(slim, reader);
        auto compiled = RenderPipelineCompiler{}.Compile(std::move(definition), {}, inputs);
        CHECK_FALSE(compiled.pipeline);
        REQUIRE_FALSE(compiled.diagnostics.empty());
        CHECK(compiled.diagnostics[0].code == PipelineDiagnosticCode::OrderingCycle);
    }
}

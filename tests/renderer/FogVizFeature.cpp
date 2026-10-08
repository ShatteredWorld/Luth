#include <doctest/doctest.h>
#include "luth/core/diagnostics/Log.h"
#include "luth/renderer/features/FogVizFeature.h"
#include "luth/renderer/features/RenderPipelineCompiler.h"
#include "luth/renderer/subsystems/VolumetricSubsystem.h"
#include "luth/renderer/subsystems/PostProcessSubsystem.h"
#include "luth/renderer/subsystems/LightingSubsystem.h"
#include "luth/memory/LinearAllocator.h"
#include <limits>
using namespace Luth;
namespace
{
    template<class T> T Native(u64 n) { return reinterpret_cast<T>(static_cast<uintptr_t>(n)); }
    struct Fixture
    {
        VolumetricSubsystem native; FogVizBindings bindings; std::unique_ptr<CompiledRenderPipeline> pipeline;
        Fixture()
        {
            bindings.enabled = true; bindings.pipeline = Native<VkPipeline>(1); bindings.layout = Native<VkPipelineLayout>(2);
            bindings.sets = {Native<VkDescriptorSet>(3), Native<VkDescriptorSet>(4)};
            bindings.state = std::make_shared<FogViewState>(); auto& state = *bindings.state;
            state.volVizDescSet.fill(bindings.sets[1]); state.volDimX = 160; state.volDimY = 90; state.volDimZ = 64;
            auto owner = [](u64 n) { return std::shared_ptr<Texture>(Native<Texture*>(n), [](Texture*){}); };
            state.depthSource = owner(10); state.volDensity = owner(11); state.volInScatterHistA = owner(12); state.volInScatterHistB = owner(13);
            RenderPipelineDefinition definition; definition.AddFeature<FogVizFeature>(native);
            PipelineInputContract inputs; inputs.resources = {{ClusterVizResources::Output}, {FogVizResources::Bindings},
                {RenderResources::SurfaceDepth, ResourceOutputPresence::Optional}, {RenderResources::FogDensity, ResourceOutputPresence::Optional},
                {RenderResources::ResolvedFog, ResourceOutputPresence::Optional}};
            auto compiled = RenderPipelineCompiler{}.Compile(std::move(definition), {}, inputs); REQUIRE(compiled.pipeline);
            pipeline = std::move(compiled.pipeline);
        }
        PipelineBuildResult Build(RG::RenderGraph& graph, Memory::LinearAllocator& scratch, GraphTextureRef& output,
            bool resolvedPresent = true, bool densityPresent = true, bool depthPresent = true, bool producer = false, bool packet = true, bool wrongParity = false)
        {
            RG::TextureDesc desc; desc.width = 640; desc.height = 480;
            const GraphTextureRef color{graph.ImportResource(desc, (void*)20, (void*)21, RG::ResourceState::ShaderResource), {Native<const Texture*>(9)}};
            desc.format = RG::TextureFormat::D32_Float;
            const GraphTextureRef depth{graph.ImportResource(desc, (void*)22, (void*)23, RG::ResourceState::ShaderResource), {Native<const Texture*>(10)}};
            desc.format = RG::TextureFormat::RGBA16_Float; desc.width = 160; desc.height = 90;
            GraphTextureRef density{graph.ImportResource(desc, (void*)24, (void*)25, RG::ResourceState::ComputeWrite), {Native<const Texture*>(11)}};
            GraphTextureRef resolved{graph.ImportResource(desc, (void*)26, (void*)27, RG::ResourceState::ComputeWrite),
                {Native<const Texture*>(((bindings.renderFrameIndex & 1u) != wrongParity) ? 12 : 13)}};
            if (producer)
            {
                struct Data { RG::ResourceHandle density, resolved; };
                graph.AddComputePass<Data>("FogProducer", RG::QueueFamily::AsyncCompute, [&](Data& data, RG::RenderPassBuilder& builder) {
                    density.handle = data.density = builder.WriteStorageImage(density.handle);
                    resolved.handle = data.resolved = builder.WriteStorageImage(resolved.handle);
                }, [](Data&, RG::RenderPassContext&){});
            }
            const FogVizBindingRef ref{packet ? &bindings : nullptr};
            const std::array inputs{RenderInputBinding::Present(ClusterVizResources::Output, color),
                RenderInputBinding::Present(FogVizResources::Bindings, ref),
                depthPresent ? RenderInputBinding::Present(RenderResources::SurfaceDepth, depth) : RenderInputBinding::Absent(RenderResources::SurfaceDepth),
                densityPresent ? RenderInputBinding::Present(RenderResources::FogDensity, density) : RenderInputBinding::Absent(RenderResources::FogDensity),
                resolvedPresent ? RenderInputBinding::Present(RenderResources::ResolvedFog, resolved) : RenderInputBinding::Absent(RenderResources::ResolvedFog)};
            FrameRenderInputs frame; frame.resources = inputs; frame.renderFrameIndex = bindings.renderFrameIndex;
            ViewRenderInputs view; view.id = {1}; view.width = 640; view.height = 480;
            const std::array exports{RenderOutputBinding::Capture(RenderResources::VisualizedLDR, output)};
            return pipeline->Build(graph, frame, view, scratch, exports);
        }
    };
}
TEST_CASE("FogVizFeature: density and scatter modes reuse imports and frozen absolute parity [renderfeatures]")
{
    for (u32 mode = 0; mode < 2; ++mode) for (u64 frame : {0ull, 1ull, (1ull << 32) + 3})
    {
        Fixture f; f.bindings.parameters = {mode, 2.5f, .7f}; f.bindings.renderFrameIndex = frame;
        Memory::LinearAllocator scratch(64 * 1024); RG::RenderGraph graph(scratch); GraphTextureRef output;
        REQUIRE(f.Build(graph, scratch, output).success); REQUIRE(graph.GetPasses().size() == 1);
        const auto& pass = graph.GetPasses()[0]; CHECK(pass.name == "VolumetricVizPass"); CHECK_FALSE(pass.isCompute);
        CHECK(pass.queueFamily == RG::QueueFamily::Graphics); CHECK_FALSE(pass.hasDepth);
        REQUIRE(pass.colorAttachments.size() == 1); CHECK(pass.colorAttachments[0].loadOp == VK_ATTACHMENT_LOAD_OP_LOAD);
        CHECK(pass.colorAttachments[0].storeOp == VK_ATTACHMENT_STORE_OP_STORE); REQUIRE(pass.reads.size() == 3);
        for (u32 i = 0; i < 3; ++i) CHECK(pass.reads[i].index == i + 2);
        CHECK(graph.GetResources().size() == 4); CHECK(output.handle.index == 1); CHECK(output.handle.version == 1);
        graph.Compile(); CHECK_FALSE(graph.GetPasses()[0].culled);
    }
}
TEST_CASE("FogVizFeature: disabled PT cold and absent fog alias fresh LDR [renderfeatures]")
{
    Fixture f; bool present = true;
    SUBCASE("disabled") { f.bindings.enabled = false; f.bindings.state.reset(); }
    SUBCASE("PT") { f.bindings.enabled = false; present = false; }
    SUBCASE("cold") { f.bindings.pipeline = VK_NULL_HANDLE; }
    SUBCASE("absent producer") { present = false; }
    for (u32 i = 0; i < 2; ++i)
    {
        Memory::LinearAllocator scratch(64 * 1024); RG::RenderGraph graph(scratch); GraphTextureRef output{{55, 1}, {}};
        REQUIRE(f.Build(graph, scratch, output, present, present, present).success);
        CHECK(graph.GetPasses().empty()); CHECK(output.handle.index == 1); CHECK(output.handle.version == 0);
    }
}
TEST_CASE("FogVizFeature: rejects missing or mismatched frozen sources before registration [renderfeatures]")
{
    Fixture f; bool depth = true, density = true, packet = true, parity = false;
    SUBCASE("packet") { packet = false; }
    SUBCASE("layout") { f.bindings.layout = VK_NULL_HANDLE; }
    SUBCASE("global") { f.bindings.sets[0] = VK_NULL_HANDLE; }
    SUBCASE("descriptor ownership") { f.bindings.state->volVizDescSet[0] = VK_NULL_HANDLE; }
    SUBCASE("state") { f.bindings.state.reset(); }
    SUBCASE("depth") { depth = false; }
    SUBCASE("density") { density = false; }
    SUBCASE("depth owner") { f.bindings.state->depthSource.reset(); }
    SUBCASE("density owner") { f.bindings.state->volDensity.reset(); }
    SUBCASE("resolved owner") { f.bindings.state->volInScatterHistB.reset(); }
    SUBCASE("parity") { parity = true; }
    SUBCASE("atlas extent") { ++f.bindings.state->volDimX; }
    SUBCASE("mode") { f.bindings.parameters.mode = 2; }
    SUBCASE("scale") { f.bindings.parameters.scale = std::numeric_limits<float>::quiet_NaN(); }
    SUBCASE("opacity") { f.bindings.parameters.overlayAlpha = std::numeric_limits<float>::infinity(); }
    Memory::LinearAllocator scratch(64 * 1024); RG::RenderGraph graph(scratch); GraphTextureRef output;
    CHECK_FALSE(f.Build(graph, scratch, output, true, density, depth, false, packet, parity).success);
    CHECK_FALSE(output.handle.IsValid()); CHECK(graph.GetPasses().empty());
}
TEST_CASE("FogVizFeature: retained domain state and async sampled dependencies survive preparation [renderfeatures]")
{
    Fixture f; std::weak_ptr<FogViewState> retained = f.bindings.state;
    {
        Memory::LinearAllocator scratch(64 * 1024); RG::RenderGraph graph(scratch); GraphTextureRef output;
        REQUIRE(f.Build(graph, scratch, output, true, true, true, true).success); f.bindings.state.reset(); CHECK_FALSE(retained.expired());
        REQUIRE(graph.GetPasses().size() == 2); CHECK(graph.GetPasses()[1].reads[1].version == 1); CHECK(graph.GetPasses()[1].reads[2].version == 1);
        graph.Compile(); CHECK_FALSE(graph.GetPasses()[0].culled);
        u32 handoffs = 0; for (const auto& barrier : graph.GetPasses()[1].preBarriers)
            if (barrier.after == RG::ResourceState::ShaderResource && barrier.crossQueueSrc) ++handoffs;
        CHECK(handoffs == 2);
    }
    CHECK(retained.expired());
}
TEST_CASE("FogVizFeature: disabled and cold preparation need no native device [renderfeatures]")
{
    VolumetricSubsystem native;
    CHECK_FALSE(native.PrepareVizBindings({}, 0, VK_NULL_HANDLE, 0, 1, 1, false).enabled);
    const auto cold = native.PrepareVizBindings({}, 0, VK_NULL_HANDLE, 0, 1, 1, true);
    CHECK(cold.enabled); CHECK_FALSE(cold.pipeline); CHECK_FALSE(cold.state);
}
TEST_CASE("FogVizFeature: combined visualization definition orders private stages independently of insertion [renderfeatures]")
{
    VolumetricSubsystem fog; LightingSubsystem lighting; PostProcessSubsystem post;
    RenderPipelineDefinition definition;
    const auto fogId = definition.AddFeature<FogVizFeature>(fog);
    const auto clusterId = definition.AddFeature<ClusterVizFeature>(lighting);
    const auto slimId = definition.AddFeature<SlimVizFeature>(post);
    PipelineInputContract inputs; inputs.resources = {{RenderResources::TonemappedLDR}, {SlimVizResources::Bindings},
        {ClusterVizResources::Bindings}, {FogVizResources::Bindings},
        {RenderResources::SurfaceDepth, ResourceOutputPresence::Optional}, {RenderResources::ClusterGrid, ResourceOutputPresence::Optional},
        {RenderResources::FogDensity, ResourceOutputPresence::Optional}, {RenderResources::ResolvedFog, ResourceOutputPresence::Optional},
        {RenderResources::Normal, ResourceOutputPresence::Optional}, {RenderResources::Roughness, ResourceOutputPresence::Optional},
        {RenderResources::MotionVectors, ResourceOutputPresence::Optional}, {RenderResources::MaterialID, ResourceOutputPresence::Optional}};
    auto result = RenderPipelineCompiler{}.Compile(std::move(definition), {}, inputs);
    REQUIRE(result.pipeline); CHECK(result.diagnostics.empty());
    const auto order = result.pipeline->FeatureOrder(); REQUIRE(order.size() == 3);
    CHECK(order[0] == slimId); CHECK(order[1] == clusterId); CHECK(order[2] == fogId);
}

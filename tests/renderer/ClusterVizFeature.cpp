#include <doctest/doctest.h>
#include "luth/core/diagnostics/Log.h"
#include "luth/renderer/features/ClusterVizFeature.h"
#include "luth/renderer/features/RenderPipelineCompiler.h"
#include "luth/renderer/subsystems/LightingSubsystem.h"
#include "luth/renderer/subsystems/PostProcessSubsystem.h"
#include "luth/memory/LinearAllocator.h"
#include <limits>
using namespace Luth;
namespace
{
    template<class T> T Native(u64 value) { return reinterpret_cast<T>(static_cast<uintptr_t>(value)); }
    struct Fixture
    {
        LightingSubsystem native; PostProcessSubsystem post;
        ClusterVizBindings bindings; SlimVizBindings slim;
        std::unique_ptr<CompiledRenderPipeline> pipeline;
        Fixture()
        {
            bindings.enabled = true; bindings.pipeline = Native<VkPipeline>(1);
            bindings.layout = Native<VkPipelineLayout>(2);
            bindings.sets = {Native<VkDescriptorSet>(3), Native<VkDescriptorSet>(4)};
            bindings.state = std::make_shared<ClusterVizViewState>(); bindings.state->set = bindings.sets[0];
            bindings.state->depth = std::shared_ptr<Texture>(Native<Texture*>(10), [](Texture*){});
            bindings.grid = {Native<VkBuffer>(5), 4096, u64(k_ClusterCount) * sizeof(GPUCluster)};
            bindings.parameters = {Vec2(640, 480), 0.1f, 1000};
            RenderPipelineDefinition definition;
            // Reverse declaration order: the semantic intermediate must order the adapters.
            definition.AddFeature<ClusterVizFeature>(native); definition.AddFeature<SlimVizFeature>(post);
            PipelineInputContract inputs; inputs.resources = {{RenderResources::TonemappedLDR},
                {SlimVizResources::Bindings}, {ClusterVizResources::Bindings},
                {RenderResources::SurfaceDepth, ResourceOutputPresence::Optional},
                {RenderResources::ClusterGrid, ResourceOutputPresence::Optional},
                {RenderResources::Normal, ResourceOutputPresence::Optional}, {RenderResources::Roughness, ResourceOutputPresence::Optional},
                {RenderResources::MotionVectors, ResourceOutputPresence::Optional}, {RenderResources::MaterialID, ResourceOutputPresence::Optional}};
            auto compiled = RenderPipelineCompiler{}.Compile(std::move(definition), {}, inputs);
            REQUIRE(compiled.pipeline); pipeline = std::move(compiled.pipeline);
        }
        PipelineBuildResult Build(RG::RenderGraph& graph, Memory::LinearAllocator& scratch, GraphTextureRef& output,
            bool gridPresent = true, bool depthPresent = true, bool packetPresent = true, bool producer = false, u32 width = 640, bool slimActive = false)
        {
            RG::TextureDesc desc; desc.width = width; desc.height = 480; desc.format = RG::TextureFormat::RGBA8_Unorm;
            GraphTextureRef color{graph.ImportResource(desc, (void*)20, (void*)21, RG::ResourceState::ShaderResource), {Native<const Texture*>(9)}};
            desc.format = RG::TextureFormat::D32_Float;
            GraphTextureRef depth{graph.ImportResource(desc, (void*)22, (void*)23, RG::ResourceState::ShaderResource), {Native<const Texture*>(10)}};
            const Memory::GPUSubRegion slice{Native<VkBuffer>(5), 4096, u64(k_ClusterCount) * sizeof(GPUCluster)};
            auto grid = native.ImportLightingBuffer(graph, "ClusterGrid", slice);
            if (producer)
            {
                struct Data { RG::BufferHandle grid; };
                graph.AddComputePass<Data>("LightAssign", RG::QueueFamily::AsyncCompute, [&](Data& data, RG::RenderPassBuilder& builder) {
                    grid.handle = data.grid = builder.WriteBuffer(grid.handle);
                }, [](Data&, RG::RenderPassContext&){});
            }
            std::array<GraphTextureRef, 4> sources{};
            if (slimActive)
            {
                slim.enabled = true; slim.pipeline = Native<VkPipeline>(6); slim.layout = Native<VkPipelineLayout>(7);
                slim.set = Native<VkDescriptorSet>(8); slim.parameters = {0, 20};
                slim.state = std::make_shared<SlimVizViewState>(); slim.state->set = slim.set;
                const std::array formats{RG::TextureFormat::RG16_Float, RG::TextureFormat::R8_Unorm,
                    RG::TextureFormat::RG16_Float, RG::TextureFormat::R16_Uint};
                for (u32 i = 0; i < 4; ++i)
                {
                    slim.state->sources[i] = std::shared_ptr<Texture>(Native<Texture*>(30 + i), [](Texture*){});
                    desc.format = formats[i];
                    sources[i] = {graph.ImportResource(desc, (void*)(uintptr_t)(40 + i * 2), (void*)(uintptr_t)(41 + i * 2),
                        RG::ResourceState::ShaderResource), {slim.state->sources[i].get()}};
                }
            }
            const auto optionalSlim = [&](auto key, u32 index) {
                return slimActive ? RenderInputBinding::Present(key, sources[index]) : RenderInputBinding::Absent(key);
            };
            const SlimVizBindingRef slimRef{&slim}; const ClusterVizBindingRef ref{packetPresent ? &bindings : nullptr};
            const std::array resources{RenderInputBinding::Present(RenderResources::TonemappedLDR, color),
                RenderInputBinding::Present(SlimVizResources::Bindings, slimRef), RenderInputBinding::Present(ClusterVizResources::Bindings, ref),
                depthPresent ? RenderInputBinding::Present(RenderResources::SurfaceDepth, depth) : RenderInputBinding::Absent(RenderResources::SurfaceDepth),
                gridPresent ? RenderInputBinding::Present(RenderResources::ClusterGrid, grid) : RenderInputBinding::Absent(RenderResources::ClusterGrid),
                optionalSlim(RenderResources::Normal, 0), optionalSlim(RenderResources::Roughness, 1),
                optionalSlim(RenderResources::MotionVectors, 2), optionalSlim(RenderResources::MaterialID, 3)};
            FrameRenderInputs frame; frame.resources = resources; ViewRenderInputs view; view.id = {1}; view.width = 640; view.height = 480;
            const std::array exports{RenderOutputBinding::Capture(RenderResources::VisualizedLDR, output)};
            return pipeline->Build(graph, frame, view, scratch, exports);
        }
    };
}
TEST_CASE("ClusterVizFeature: composed visualization reads depth and tagged grid without duplicate imports [renderfeatures]")
{
    Fixture f; Memory::LinearAllocator scratch(64 * 1024); RG::RenderGraph graph(scratch); GraphTextureRef output;
    REQUIRE(f.Build(graph, scratch, output).success); REQUIRE(graph.GetPasses().size() == 1);
    const auto& pass = graph.GetPasses()[0]; CHECK(pass.name == "ClusterVizPass"); CHECK_FALSE(pass.isCompute);
    CHECK(pass.queueFamily == RG::QueueFamily::Graphics); CHECK_FALSE(pass.hasDepth);
    REQUIRE(pass.colorAttachments.size() == 1); CHECK(pass.colorAttachments[0].loadOp == VK_ATTACHMENT_LOAD_OP_LOAD);
    CHECK(pass.colorAttachments[0].storeOp == VK_ATTACHMENT_STORE_OP_STORE);
    REQUIRE(pass.reads.size() == 1); CHECK(pass.reads[0].index == 2);
    REQUIRE(pass.bufferReads.size() == 1); CHECK(pass.bufferReads[0].index == 1);
    CHECK(pass.bufferReadStates[0] == RG::ResourceState::FragmentStorageRead);
    CHECK(graph.GetResources().size() == 2); CHECK(graph.GetBuffers().size() == 1);
    CHECK(output.handle.index == 1); CHECK(output.handle.version == 1); CHECK(output.binding.texture == Native<const Texture*>(9));
    graph.Compile(); CHECK_FALSE(graph.GetPasses()[0].culled);
}
TEST_CASE("ClusterVizFeature: disabled PT cold and absent provider publish fresh aliases [renderfeatures]")
{
    Fixture f; bool grid = true, depth = true;
    SUBCASE("disabled") { f.bindings.enabled = false; }
    SUBCASE("PT") { f.bindings.enabled = false; grid = depth = false; }
    SUBCASE("cold") { f.bindings.pipeline = VK_NULL_HANDLE; }
    SUBCASE("no provider") { grid = false; }
    for (u32 i = 0; i < 2; ++i)
    {
        Memory::LinearAllocator scratch(64 * 1024); RG::RenderGraph graph(scratch); GraphTextureRef output{{55, 1}, {}};
        REQUIRE(f.Build(graph, scratch, output, grid, depth).success); CHECK(graph.GetPasses().empty());
        CHECK(output.handle.index == 1); CHECK(output.handle.version == 0);
    }
}
TEST_CASE("ClusterVizFeature: rejects inconsistent frozen bindings before registration [renderfeatures]")
{
    Fixture f; bool depth = true, packet = true; u32 width = 640;
    SUBCASE("packet") { packet = false; }
    SUBCASE("depth absent") { depth = false; }
    SUBCASE("layout") { f.bindings.layout = VK_NULL_HANDLE; }
    SUBCASE("depth set") { f.bindings.sets[0] = VK_NULL_HANDLE; }
    SUBCASE("lighting set") { f.bindings.sets[1] = VK_NULL_HANDLE; }
    SUBCASE("state") { f.bindings.state.reset(); }
    SUBCASE("descriptor ownership") { f.bindings.state->set = VK_NULL_HANDLE; }
    SUBCASE("owner") { f.bindings.state->depth.reset(); }
    SUBCASE("buffer") { f.bindings.grid.buffer = Native<VkBuffer>(6); }
    SUBCASE("offset") { ++f.bindings.grid.offset; }
    SUBCASE("size") { --f.bindings.grid.range; }
    SUBCASE("viewport") { f.bindings.parameters.viewport.x++; }
    SUBCASE("extent") { width++; }
    SUBCASE("near") { f.bindings.parameters.nearZ = 0; }
    SUBCASE("far") { f.bindings.parameters.farZ = f.bindings.parameters.nearZ; }
    SUBCASE("finite") { f.bindings.parameters.farZ = std::numeric_limits<float>::quiet_NaN(); }
    Memory::LinearAllocator scratch(64 * 1024); RG::RenderGraph graph(scratch); GraphTextureRef output{{55, 1}, {}};
    CHECK_FALSE(f.Build(graph, scratch, output, true, depth, packet, false, width).success);
    CHECK_FALSE(output.handle.IsValid()); CHECK(graph.GetPasses().empty());
}
TEST_CASE("ClusterVizFeature: frozen jobs retain depth and declare async grid handoff [renderfeatures]")
{
    Fixture f; std::weak_ptr<Texture> retained = f.bindings.state->depth;
    std::weak_ptr<ClusterVizViewState> retainedState = f.bindings.state;
    {
        Memory::LinearAllocator scratch(64 * 1024); RG::RenderGraph graph(scratch); GraphTextureRef output;
        REQUIRE(f.Build(graph, scratch, output, true, true, true, true).success); f.bindings.state.reset(); CHECK_FALSE(retained.expired()); CHECK_FALSE(retainedState.expired());
        REQUIRE(graph.GetPasses().size() == 2); CHECK(graph.GetPasses()[1].bufferReads[0].version == 1);
        graph.Compile(); CHECK_FALSE(graph.GetPasses()[0].culled);
        REQUIRE(graph.GetPasses()[1].bufferPreBarriers.size() == 1);
        CHECK(graph.GetPasses()[1].bufferPreBarriers[0].before == RG::ResourceState::StorageBufferWrite);
        CHECK(graph.GetPasses()[1].bufferPreBarriers[0].after == RG::ResourceState::FragmentStorageRead);
        CHECK(graph.GetPasses()[1].bufferPreBarriers[0].crossQueueSrc);
    }
    CHECK(retained.expired()); CHECK(retainedState.expired());
}
TEST_CASE("ClusterVizFeature: cold and disabled preparation need no device [renderfeatures]")
{
    LightingSubsystem native;
    CHECK_FALSE(native.PrepareClusterVizBindings({}, VK_NULL_HANDLE, {}, 0, 0, 0, 0, false).enabled);
    const auto cold = native.PrepareClusterVizBindings({}, VK_NULL_HANDLE, {}, 640, 480, .1f, 1000, true);
    CHECK(cold.enabled); CHECK_FALSE(cold.pipeline); CHECK_FALSE(cold.state);
}
TEST_CASE("ClusterVizFeature: shared composition preserves slim output and ordered alias versions [renderfeatures]")
{
    Fixture f;
    SUBCASE("slim mode passes through cluster stage") { f.bindings.enabled = false; }
    SUBCASE("both contributions keep semantic order") {}
    Memory::LinearAllocator scratch(64 * 1024); RG::RenderGraph graph(scratch); GraphTextureRef output;
    REQUIRE(f.Build(graph, scratch, output, true, true, true, false, 640, true).success);
    REQUIRE(graph.GetPasses().size() == (f.bindings.enabled ? 2 : 1));
    CHECK(graph.GetPasses()[0].name == "SlimVizPass"); CHECK(output.handle.index == 1);
    CHECK(output.handle.version == (f.bindings.enabled ? 2 : 1));
    if (f.bindings.enabled) CHECK(graph.GetPasses()[1].name == "ClusterVizPass");
    CHECK(graph.GetResources().size() == 6); CHECK(graph.GetBuffers().size() == 1);
}
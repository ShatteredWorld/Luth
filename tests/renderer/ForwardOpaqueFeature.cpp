#include <doctest/doctest.h>
#include "luth/core/diagnostics/Log.h"
#include "luth/renderer/features/ForwardOpaqueCompatibility.h"
#include "luth/renderer/features/RenderPipelineCompiler.h"
#include "luth/renderer/subsystems/GeometrySubsystem.h"
#include "luth/renderer/draw/DrawList.h"
#include "luth/core/RenderSnapshot.h"
#include "luth/memory/LinearAllocator.h"

using namespace Luth;
namespace
{
    template<class T> T Native(u64 value) { return reinterpret_cast<T>(static_cast<uintptr_t>(value)); }
    struct Fixture
    {
        GeometrySubsystem native;
        ForwardOpaqueBindings bindings;
        Memory::GPUSubRegion indirect{Native<VkBuffer>(1), 8192, 819200};
        Memory::GPUSubRegion lights{Native<VkBuffer>(1), 16384, 256};
        std::unique_ptr<CompiledRenderPipeline> pipeline;
        GraphTextureRef color, depth, picking;
        bool hybrid;
        Fixture(bool hybridComposition = false) : hybrid(hybridComposition)
        {
            bindings.initialLayout = Native<VkPipelineLayout>(2); bindings.sets.fill(Native<VkDescriptorSet>(3));
            RenderPipelineDefinition definition;
            if (hybrid) definition.AddFeature<HybridForwardOpaqueFeature>(native);
            else definition.AddFeature<ForwardOpaqueFeature>(native);
            const auto info = hybrid ? HybridForwardOpaqueFeature(native).Describe() : ForwardOpaqueFeature(native).Describe();
            PipelineInputContract inputs;
            for (auto read : info.resources.reads) inputs.resources.push_back({read.key,
                read.requirement == ResourceReadRequirement::Required ? ResourceOutputPresence::Required : ResourceOutputPresence::Optional});
            inputs.capabilities = {&DeformationResources::DeformedGeometry};
            auto compiled = RenderPipelineCompiler{}.Compile(std::move(definition), {}, inputs);
            REQUIRE(compiled.pipeline); pipeline = std::move(compiled.pipeline);
        }
        PipelineBuildResult Build(RG::RenderGraph& graph, Memory::LinearAllocator& scratch, u32 viewIndex = 0,
            bool auxiliary = false, bool packetPresent = true, u32 width = 640)
        {
            const auto target = [&](const char* name, RG::TextureFormat format, u32 extent) {
                RG::TextureDesc desc; desc.name = name; desc.width = extent; desc.height = 480; desc.format = format;
                return GraphTextureRef{graph.ImportResource(desc, (void*)4, (void*)5, RG::ResourceState::Undefined),
                    {Native<const Texture*>(6)}};
            };
            const auto colorTarget = target("SceneColor", RG::TextureFormat::RGBA16_Float, width);
            const auto depthTarget = target("SceneDepth", RG::TextureFormat::D32_Float, 640);
            const auto pickingTarget = target("EntityID", RG::TextureFormat::R32_Uint, 640);
            const GraphBufferRef indirectRef{graph.ImportBuffer({"Indirect", indirect.size, VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT},
                (void*)indirect.buffer, RG::ResourceState::Undefined), {&indirect, indirect.offset, indirect.size}};
            const VisibleDrawRange visible{indirectRef, viewIndex * 5 * 4096, 23};
            const ForwardOpaqueBindingRef binding{packetPresent ? &bindings : nullptr};
            std::vector<RenderInputBinding> resources{RenderInputBinding::Present(ForwardOpaqueResources::Bindings, binding),
                RenderInputBinding::Present(ForwardOpaqueResources::ColorTarget, colorTarget),
                RenderInputBinding::Present(ForwardOpaqueResources::PickingTarget, pickingTarget),
                RenderInputBinding::Present(RenderResources::SurfaceDepth, depthTarget),
                RenderInputBinding::Present(RenderResources::CameraVisibleDraws, visible)};
            if (auxiliary)
            {
                ShadowCascadeRefs shadows;
                for (auto& cascade : shadows.cascades) cascade = target("Shadow", RG::TextureFormat::D32_Float, 640);
                // Keep borrowed values alive until Build returns.
                auto* retained = scratch.New<ShadowCascadeRefs>(shadows);
                resources.push_back(RenderInputBinding::Present(RenderResources::ShadowCascades, *retained));
                auto* ao = scratch.New<GraphTextureRef>(target("AO", RG::TextureFormat::R8_Unorm, 640));
                resources.push_back(RenderInputBinding::Present(RenderResources::AmbientOcclusion, *ao));
                for (auto key : {RenderResources::LightData, RenderResources::ClusterGrid, RenderResources::LightIndices})
                {
                    auto* buffer = scratch.New<GraphBufferRef>(GraphBufferRef{graph.ImportBuffer({"Lighting", lights.size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT},
                        (void*)lights.buffer, RG::ResourceState::Undefined), {&lights, lights.offset, lights.size}});
                    resources.push_back(RenderInputBinding::Present(key, *buffer));
                }
                if (hybrid)
                    for (auto key : {ForwardCompatibilityResources::SunShadowMask, ForwardCompatibilityResources::DenoisedDiffuseDI,
                        ForwardCompatibilityResources::DenoisedDiffuseGI, ForwardCompatibilityResources::DenoisedReflectionRadiance,
                        ForwardCompatibilityResources::DenoisedSpecularDI})
                    {
                        auto* image = scratch.New<GraphTextureRef>(target("RTSignal", RG::TextureFormat::RGBA16_Float, 640));
                        resources.push_back(RenderInputBinding::Present(key, *image));
                    }
            }
            const std::array capabilities{&DeformationResources::DeformedGeometry};
            FrameRenderInputs frame; frame.resources = resources; frame.capabilities = capabilities;
            ViewRenderInputs view; view.id = {u64(viewIndex) + 1}; view.width = 640; view.height = 480;
            const std::array exports{RenderOutputBinding::Capture(RenderResources::OpaqueHDR, color),
                RenderOutputBinding::Capture(RenderResources::LitDepth, depth),
                RenderOutputBinding::Capture(RenderResources::OpaquePickingIDs, picking)};
            return pipeline->Build(graph, frame, view, scratch, exports);
        }
    };
}
TEST_CASE("ForwardOpaqueFeature: native attachment and typed output contracts preserve formats and states [renderfeatures]")
{
    Fixture fixture;
    Memory::LinearAllocator scratch(128 * 1024); RG::RenderGraph graph(scratch);
    REQUIRE(fixture.Build(graph, scratch, 0, true).success);
    REQUIRE(graph.GetPasses().size() == 1);
    const auto& pass = graph.GetPasses()[0];
    CHECK(pass.name == "GeometryPass"); CHECK_FALSE(pass.isCompute); CHECK(pass.queueFamily == RG::QueueFamily::Graphics);
    REQUIRE(pass.colorAttachments.size() == 2);
    CHECK(pass.colorAttachments[0].handle == fixture.color.handle);
    CHECK(pass.colorAttachments[0].loadOp == VK_ATTACHMENT_LOAD_OP_CLEAR);
    CHECK(pass.colorAttachments[0].storeOp == VK_ATTACHMENT_STORE_OP_STORE);
    CHECK(pass.colorAttachments[1].handle == fixture.picking.handle);
    CHECK(pass.colorAttachments[1].clearValue.color.uint32[0] == 0);
    CHECK(pass.depthAttachment.handle == fixture.depth.handle);
    CHECK(pass.depthAttachment.loadOp == VK_ATTACHMENT_LOAD_OP_LOAD);
    CHECK(pass.depthAttachment.storeOp == VK_ATTACHMENT_STORE_OP_STORE);
    CHECK(pass.reads.size() == 5); CHECK(pass.bufferReads.size() == 4);
    for (size_t i = 0; i < 3; ++i) CHECK(pass.bufferReadStates[i] == RG::ResourceState::FragmentStorageRead);
    CHECK(pass.bufferReadStates.back() == RG::ResourceState::IndirectRead);
    CHECK(fixture.color.handle.index == 1); CHECK(fixture.depth.handle.index == 2); CHECK(fixture.picking.handle.index == 3);
    CHECK(fixture.color.handle.version == 1); CHECK(fixture.depth.handle.version == 1);
    graph.Compile(); CHECK_FALSE(graph.GetPasses()[0].culled);
}
TEST_CASE("ForwardOpaqueFeature: hybrid compatibility signals are declared separately from raster [renderfeatures]")
{
    Fixture fixture(true);
    Memory::LinearAllocator scratch(128 * 1024); RG::RenderGraph graph(scratch);
    REQUIRE(fixture.Build(graph, scratch, 0, true).success);
    CHECK(graph.GetPasses()[0].reads.size() == 10);
    const auto raster = ForwardOpaqueFeature(fixture.native).Describe();
    CHECK(raster.resources.reads.size() == 10);
    CHECK(HybridForwardOpaqueFeature(fixture.native).Describe().resources.reads.size() == 15);
}
TEST_CASE("ForwardOpaqueFeature: incomplete inputs reject before registration and reset all exports [renderfeatures]")
{
    Fixture fixture; bool packet = true; u32 width = 640;
    SUBCASE("native packet absent") { packet = false; }
    SUBCASE("descriptor absent") { fixture.bindings.sets[5] = VK_NULL_HANDLE; }
    SUBCASE("extent mismatch") { width = 641; }
    SUBCASE("slice too small") { fixture.indirect.size = 16; }
    Memory::LinearAllocator scratch(128 * 1024); RG::RenderGraph graph(scratch);
    CHECK_FALSE(fixture.Build(graph, scratch, 0, false, packet, width).success);
    CHECK(graph.GetPasses().empty()); CHECK_FALSE(fixture.color.handle.IsValid());
    CHECK_FALSE(fixture.depth.handle.IsValid()); CHECK_FALSE(fixture.picking.handle.IsValid());
}
TEST_CASE("ForwardOpaqueFeature: cold resources and two-view graphs preserve attachment clears [renderfeatures]")
{
    Fixture fixture; fixture.bindings = {};
    for (u32 view : {0u, 1u})
    {
        Memory::LinearAllocator scratch(128 * 1024); RG::RenderGraph graph(scratch);
        REQUIRE(fixture.Build(graph, scratch, view).success);
        CHECK(graph.GetResources().size() == 3); CHECK(fixture.color.handle.IsValid());
    }
}
TEST_CASE("ForwardOpaqueFeature: tagged slice offsets include the selected camera region [renderfeatures]")
{
    Memory::GPUSubRegion slice{Native<VkBuffer>(1), 8192, 819200};
    VisibleDrawRange range{{{1, 0}, {&slice, slice.offset, slice.size}}, 5 * 4096, 23};
    CHECK(GeometrySubsystem::ForwardDrawOffset(range, 7) == 8192 + u64(5 * 4096 + 7) * sizeof(VkDrawIndexedIndirectCommand));
    CHECK_THROWS_AS(GeometrySubsystem::ForwardDrawOffset(range, 23), std::invalid_argument);
    range.indirect.binding.offset = 0;
    CHECK_THROWS_AS(GeometrySubsystem::ForwardDrawOffset(range, 0), std::invalid_argument);
}
TEST_CASE("ForwardOpaqueFeature: native cold preparation freezes polygon mode and descriptors [renderfeatures]")
{
    GeometrySubsystem native; DrawList draws; RenderSnapshot snapshot;
    std::array<VkDescriptorSet, 6> sets; sets.fill(Native<VkDescriptorSet>(1));
    const auto packet = native.PrepareForwardOpaqueBindings(sets, true, false, true, {}, draws, snapshot);
    sets.fill(Native<VkDescriptorSet>(2));
    for (auto set : packet.sets) CHECK(set == Native<VkDescriptorSet>(1));
    CHECK(packet.polygon == VK_POLYGON_MODE_LINE); CHECK(packet.captureDraws);
    CHECK(packet.draws.empty()); CHECK(packet.overlays.empty());
}

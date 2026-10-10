#include <doctest/doctest.h>
#include "luth/core/diagnostics/Log.h"
#include "luth/renderer/features/VisibilityFeature.h"
#include "luth/renderer/features/RenderPipelineCompiler.h"
#include "luth/renderer/subsystems/GeometrySubsystem.h"
#include "luth/renderer/subsystems/SkinningSubsystem.h"
#include "luth/memory/LinearAllocator.h"
#include "luth/core/RenderSnapshot.h"

using namespace Luth;
using namespace Luth::RenderResources;

namespace
{
    template<class T> T Native(u64 value) { return reinterpret_cast<T>(static_cast<uintptr_t>(value)); }
    struct InitializedReader final : IRenderFeature
    {
        FeatureInfo Describe() const override
        {
            FeatureInfo info;
            info.name = "InitializedReader";
            info.resources.reads = {{InitializedIndirectData}};
            return info;
        }
        FeatureFrameDecision Evaluate(const FeaturePrepareContext&) const override { return {}; }
        void Build(RG::RenderGraph&, RenderFeatureContext&) override {}
    };
    struct VisibilityFixture
    {
        GeometrySubsystem native;
        std::unique_ptr<CompiledRenderPipeline> pipeline;
        Memory::GPUSubRegion objectSlice{Native<VkBuffer>(1), 128, 65536};
        Memory::GPUSubRegion indirectSlice{Native<VkBuffer>(1), 8192, 10 * 4096 * sizeof(VkDrawIndexedIndirectCommand)};
        VisibilityFixture()
        {
            RenderPipelineDefinition definition;
            definition.AddFeature<VisibilityFeature>(native);
            PipelineInputContract inputs;
            inputs.resources = {{VisibilityResources::Parameters}, {ObjectData}, {InitializedIndirectData}};
            inputs.capabilities = {&DeformationResources::DeformedGeometry};
            auto compiled = RenderPipelineCompiler{}.Compile(std::move(definition), {}, inputs);
            REQUIRE(compiled.pipeline);
            pipeline = std::move(compiled.pipeline);
        }
        PipelineBuildResult Build(RG::RenderGraph& graph, Memory::LinearAllocator& scratch,
            VisibilityParameters parameters, VisibleDrawRange& camera, CascadeDrawRanges& cascades)
        {
            const GraphBufferRef objects{graph.ImportBuffer({"Objects", objectSlice.size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT},
                (void*)objectSlice.buffer, RG::ResourceState::Undefined), {&objectSlice, objectSlice.offset, objectSlice.size}};
            const GraphBufferRef indirect{graph.ImportBuffer({"Indirect", indirectSlice.size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT},
                (void*)indirectSlice.buffer, RG::ResourceState::Undefined), {&indirectSlice, indirectSlice.offset, indirectSlice.size}};
            const std::array bindings{RenderInputBinding::Present(VisibilityResources::Parameters, parameters),
                RenderInputBinding::Present(ObjectData, objects), RenderInputBinding::Present(InitializedIndirectData, indirect)};
            const std::array capabilities{&DeformationResources::DeformedGeometry};
            FrameRenderInputs frame;
            frame.renderFrameIndex = 17;
            frame.resources = bindings;
            frame.capabilities = capabilities;
            const std::array outputs{RenderOutputBinding::Capture(CameraVisibleDraws, camera),
                RenderOutputBinding::Capture(CascadeVisibleDraws, cascades)};
            return pipeline->Build(graph, frame, {}, scratch, outputs);
        }
    };
}

TEST_CASE("VisibilityFeature: view and cascade ranges retain physical slice offsets [renderfeatures]")
{
    VisibilityFixture fixture;
    for (const u32 viewIndex : {0u, 1u})
    {
        Memory::LinearAllocator scratch(64 * 1024);
        RG::RenderGraph graph(scratch);
        VisibilityParameters parameters;
        parameters.viewIndex = viewIndex;
        parameters.objectCount = 23;
        parameters.cullCascades = true;
        VisibleDrawRange camera;
        CascadeDrawRanges cascades;
        REQUIRE(fixture.Build(graph, scratch, parameters, camera, cascades).success);
        CHECK(camera.firstDraw == viewIndex * 5 * 4096);
        CHECK(camera.maxDrawCount == 23);
        CHECK(camera.indirect.binding.slice == &fixture.indirectSlice);
        CHECK(camera.indirect.binding.offset == 8192);
        CHECK(camera.indirect.binding.size == fixture.indirectSlice.size);
        for (u32 i = 0; i < 4; ++i)
        {
            CHECK(cascades.cascades[i].firstDraw == camera.firstDraw + (i + 1) * 4096);
            CHECK(cascades.cascades[i].maxDrawCount == 23);
            CHECK(cascades.cascades[i].indirect.handle == camera.indirect.handle);
        }
        CHECK(graph.GetPasses().empty()); // Native PSO absent: initialized commands pass through.
        CHECK(graph.GetBuffers().size() == 2); // Same VkBuffer, distinct tagged-heap slices.
    }
}

TEST_CASE("VisibilityFeature: PT omission and disabled cascades reset graph-local exports [renderfeatures]")
{
    VisibilityFixture fixture;
    VisibleDrawRange camera;
    CascadeDrawRanges cascades;
    for (const bool realtime : {true, false, true})
    {
        Memory::LinearAllocator scratch(64 * 1024);
        RG::RenderGraph graph(scratch);
        VisibilityParameters parameters;
        parameters.realtime = realtime;
        REQUIRE(fixture.Build(graph, scratch, parameters, camera, cascades).success);
        CHECK(camera.indirect.handle.IsValid() == realtime);
        CHECK(camera.maxDrawCount == 0); // Empty scene remains valid.
        for (const auto& cascade : cascades.cascades) CHECK_FALSE(cascade.indirect.handle.IsValid());
        CHECK(graph.GetPasses().empty());
    }
}

TEST_CASE("VisibilityFeature: invalid region layout rejects before native registration [renderfeatures]")
{
    VisibilityFixture fixture;
    Memory::LinearAllocator scratch(64 * 1024);
    RG::RenderGraph graph(scratch);
    VisibilityParameters parameters;
    SUBCASE("slot outside layout") { parameters.viewIndex = 2; }
    SUBCASE("objects exceed stride") { parameters.objectCount = 4097; }
    SUBCASE("zero stride") { parameters.regionStride = 0; }
    SUBCASE("zero view capacity") { parameters.maxViews = 0; }
    SUBCASE("undersized slice") { fixture.indirectSlice.size = 16; }
    SUBCASE("undersized object slice") { parameters.objectCount = 23; fixture.objectSlice.size = 16; }
    SUBCASE("overflow") { parameters.maxViews = 0xffffffffu; }
    VisibleDrawRange camera;
    CascadeDrawRanges cascades;
    const auto result = fixture.Build(graph, scratch, parameters, camera, cascades);
    CHECK_FALSE(result.success);
    CHECK_FALSE(result.diagnostics.empty());
    CHECK_FALSE(camera.indirect.handle.IsValid());
    CHECK(graph.GetPasses().empty());
}

TEST_CASE("VisibilityFeature: native cull chain exposes actual buffer versions and graphics routing [renderfeatures]")
{
    Memory::LinearAllocator scratch(64 * 1024);
    RG::RenderGraph graph(scratch);
    const auto objects = graph.ImportBuffer({"Objects", 4096, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT},
        (void*)1, RG::ResourceState::Undefined);
    auto indirect = graph.ImportBuffer({"Indirect", 819200, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT},
        (void*)2, RG::ResourceState::Undefined);
    const CullBindings bindings{Native<VkPipeline>(3), Native<VkPipelineLayout>(4), Native<VkDescriptorSet>(5), 23};
    for (u32 i = 0; i < 5; ++i)
    {
        const auto name = i ? "FrustumCull.C" + std::to_string(i - 1) : std::string("FrustumCull.Cam");
        const auto previous = indirect;
        indirect = GeometrySubsystem::AddCullPass(graph, objects, indirect, {}, i * 4096, name.c_str(), bindings, nullptr);
        REQUIRE(graph.GetPasses().size() == i + 1);
        const auto& pass = graph.GetPasses().back();
        CHECK(pass.name == name);
        CHECK(pass.queueFamily == RG::QueueFamily::Graphics);
        REQUIRE(pass.bufferReads.size() == 1);
        REQUIRE(pass.bufferWrites.size() == 1);
        CHECK(pass.bufferReads[0] == objects);
        CHECK(pass.bufferWrites[0] == indirect);
        CHECK(indirect.index == previous.index);
        CHECK(indirect.version == previous.version + 1);
    }
    graph.Compile();
    for (const auto& pass : graph.GetPasses()) CHECK_FALSE(pass.culled);
    CHECK(graph.GetBuffers().size() == 2);
}

TEST_CASE("VisibilityFeature: native omission and incomplete bindings are explicit [renderfeatures]")
{
    Memory::LinearAllocator scratch(64 * 1024);
    RG::RenderGraph graph(scratch);
    const auto objects = graph.ImportBuffer({"Objects", 4096, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT}, (void*)1, RG::ResourceState::Undefined);
    const auto indirect = graph.ImportBuffer({"Indirect", 819200, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT}, (void*)2, RG::ResourceState::Undefined);
    CHECK(GeometrySubsystem::AddCullPass(graph, objects, indirect, {}, 0, nullptr, {}, nullptr) == indirect);
    const CullBindings incomplete{Native<VkPipeline>(3), VK_NULL_HANDLE, VK_NULL_HANDLE, 1};
    CHECK_THROWS_AS(GeometrySubsystem::AddCullPass(graph, objects, indirect, {}, 0, nullptr, incomplete, nullptr), std::invalid_argument);
    CHECK(graph.GetPasses().empty());
}

TEST_CASE("VisibilityFeature: semantic dependency orders deformation before visibility [renderfeatures]")
{
    GeometrySubsystem geometry;
    SkinningSubsystem skinning;
    RenderPipelineDefinition definition;
    const auto visibility = definition.AddFeature<VisibilityFeature>(geometry);
    const auto deformation = definition.AddFeature<DeformationFeature>(skinning);
    PipelineInputContract inputs;
    inputs.resources = {{VisibilityResources::Parameters}, {DeformationResources::Parameters}, {ObjectData}, {InitializedIndirectData}};
    auto compiled = RenderPipelineCompiler{}.Compile(std::move(definition), {}, inputs);
    REQUIRE(compiled.pipeline);
    REQUIRE(compiled.pipeline->FeatureOrder().size() == 2);
    CHECK(compiled.pipeline->FeatureOrder()[0] == deformation);
    CHECK(compiled.pipeline->FeatureOrder()[1] == visibility);
}

TEST_CASE("VisibilityFeature: initialized-command readers precede in-place culling [renderfeatures]")
{
    GeometrySubsystem geometry;
    RenderPipelineDefinition definition;
    const auto visibility = definition.AddFeature<VisibilityFeature>(geometry);
    const auto reader = definition.AddFeature<InitializedReader>();
    PipelineInputContract inputs;
    inputs.resources = {{VisibilityResources::Parameters}, {ObjectData}, {InitializedIndirectData}};
    inputs.capabilities = {&DeformationResources::DeformedGeometry};
    SUBCASE("alias mutation orders the earlier-stage reader")
    {
        auto compiled = RenderPipelineCompiler{}.Compile(std::move(definition), {}, inputs);
        REQUIRE(compiled.pipeline);
        CHECK(compiled.pipeline->FeatureOrder()[0] == reader);
        CHECK(compiled.pipeline->FeatureOrder()[1] == visibility);
    }
    SUBCASE("contradictory explicit order rejects the definition")
    {
        definition.Before(visibility, reader);
        auto compiled = RenderPipelineCompiler{}.Compile(std::move(definition), {}, inputs);
        CHECK_FALSE(compiled.pipeline);
        REQUIRE_FALSE(compiled.diagnostics.empty());
        CHECK(compiled.diagnostics[0].code == PipelineDiagnosticCode::OrderingCycle);
    }
}

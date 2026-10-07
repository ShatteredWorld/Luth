#include <doctest/doctest.h>
#include "luth/renderer/features/RenderPipelineCompiler.h"
#include "luth/renderer/rendergraph/RenderGraph.h"
#include "luth/memory/LinearAllocator.h"
#include <array>
#include <functional>

using namespace Luth;
using namespace Luth::RenderResources;

static_assert(!std::is_copy_constructible_v<RenderPipelineDefinition>);
static_assert(std::is_move_constructible_v<RenderPipelineDefinition>);

namespace
{
    constexpr RenderCapabilityIdentity Scene{"Scene"};
    constexpr RenderCapabilityIdentity Geometry{"Geometry"};
    constexpr DeviceCapabilityIdentity RayQuery{"RayQuery"};
    constexpr DeviceCapabilityIdentity RayPipeline{"RayPipeline"};
    constexpr FeatureTypeIdentity TypeA{"A"}, TypeB{"B"};

    struct Probe
    {
        int described = 0, evaluated = 0, prepared = 0, built = 0, released = 0;
        FeatureFrameDecision decision;
        RenderViewId releasedView;
        std::vector<std::string>* trace = nullptr;
        std::function<void(RG::RenderGraph&, RenderFeatureContext&)> build;
    };
    class TestFeature final : public IRenderFeature
    {
    public:
        TestFeature(FeatureInfo info, Probe& probe) : m_Info(std::move(info)), m_Probe(probe) {}
        FeatureInfo Describe() const override { ++m_Probe.described; return m_Info; }
        FeatureFrameDecision Evaluate(const FeaturePrepareContext&) const override
        {
            ++m_Probe.evaluated; return m_Probe.decision;
        }
        void Prepare(FeaturePrepareContext&) override
        {
            ++m_Probe.prepared;
            if (m_Probe.trace) m_Probe.trace->push_back("prepare:" + m_Info.name);
        }
        void Build(RG::RenderGraph& graph, RenderFeatureContext& context) override
        {
            ++m_Probe.built;
            if (m_Probe.trace) m_Probe.trace->push_back("build:" + m_Info.name);
            if (m_Probe.build) m_Probe.build(graph, context);
        }
        void ReleaseView(RenderViewId view) override { ++m_Probe.released; m_Probe.releasedView = view; }
    private:
        FeatureInfo m_Info;
        Probe& m_Probe;
    };
    FeatureInfo Info(std::string name, std::vector<ResourceRead> reads = {}, std::vector<ResourceWrite> writes = {},
        FeaturePhase phase = FeaturePhase::BeforeAsync)
    {
        FeatureInfo info;
        info.name = std::move(name);
        info.allowMultipleInstances = true; // TestFeature emulates several native adapters.
        info.phase = phase;
        info.resources = {std::move(reads), std::move(writes)};
        return info;
    }
    template<class Result>
    const PipelineDiagnostic* Find(const Result& result, PipelineDiagnosticCode code)
    {
        for (const auto& diagnostic : result.diagnostics) if (diagnostic.code == code) return &diagnostic;
        return nullptr;
    }
    std::vector<FeatureInstanceId> Order(const PipelineCompileResult& result)
    {
        REQUIRE(result.pipeline);
        return {result.pipeline->FeatureOrder().begin(), result.pipeline->FeatureOrder().end()};
    }
    PipelineBuildResult BuildOnce(CompiledRenderPipeline& pipeline, const FrameRenderInputs& frame = {},
        const ViewRenderInputs& view = {})
    {
        Memory::LinearAllocator scratch(64 * 1024);
        RG::RenderGraph graph(scratch);
        return pipeline.Build(graph, frame, view, scratch);
    }
    struct PassData {};
    void AddPass(RG::RenderGraph& graph, RG::QueueFamily queue)
    {
        graph.AddComputePass<PassData>("NativePass", queue,
            [](PassData&, RG::RenderPassBuilder& builder) { builder.SetHasSideEffect(); },
            [](PassData&, RG::RenderPassContext&) {});
    }
}

TEST_CASE("FeatureCompiler: missing producers identify feature key and expected type [renderfeatures]")
{
    Probe probe;
    RenderPipelineDefinition definition;
    definition.AddFeature<TestFeature>(Info("TAA", {{MotionVectors}}), probe);
    auto result = RenderPipelineCompiler{}.Compile(std::move(definition), {});
    CHECK_FALSE(result.pipeline);
    const auto* error = Find(result, PipelineDiagnosticCode::MissingProducer);
    REQUIRE(error);
    CHECK(error->feature == 1);
    CHECK(error->resource.identity == MotionVectors.identity);
    CHECK(error->resource.type == &RenderResourceType<GraphTextureRef>);
    CHECK(error->message.find("TAA") != std::string::npos);
    CHECK(error->message.find("MotionVectors") != std::string::npos);
    CHECK(probe.described == 1);
    CHECK(probe.evaluated == 0);
    CHECK(probe.prepared == 0);
    CHECK(probe.built == 0);
}

TEST_CASE("FeatureCompiler: input type mismatches and producer conflicts [renderfeatures]")
{
    Probe a, b;
    RenderPipelineDefinition definition;
    const RenderResourceKey<GraphBufferRef> wrong{SurfaceDepth.identity};
    SUBCASE("wrong imported type")
    {
        definition.AddFeature<TestFeature>(Info("Reader", {{SurfaceDepth}}), a);
        auto result = RenderPipelineCompiler{}.Compile(std::move(definition), {}, {{{wrong}}, {}});
        CHECK_FALSE(result.pipeline);
        CHECK(Find(result, PipelineDiagnosticCode::TypeMismatch));
    }
    SUBCASE("duplicate outputs")
    {
        definition.AddFeature<TestFeature>(Info("First", {}, {{OpaqueHDR}}), a);
        definition.AddFeature<TestFeature>(Info("Second", {}, {{OpaqueHDR}}), b);
        auto result = RenderPipelineCompiler{}.Compile(std::move(definition), {});
        CHECK(Find(result, PipelineDiagnosticCode::DuplicateProducer));
        CHECK_FALSE(result.pipeline);
    }
    SUBCASE("exclusive providers")
    {
        ResourceWrite exclusive{OpaqueHDR}; exclusive.exclusive = true;
        definition.AddFeature<TestFeature>(Info("ForwardOpaque", {}, {exclusive}), a);
        definition.AddFeature<TestFeature>(Info("PathTrace", {}, {exclusive}), b);
        auto result = RenderPipelineCompiler{}.Compile(std::move(definition), {});
        const auto* diagnostic = Find(result, PipelineDiagnosticCode::ExclusiveProducers);
        REQUIRE(diagnostic);
        CHECK(diagnostic->feature == 2);
        CHECK(diagnostic->relatedFeature == 1);
        CHECK(diagnostic->message.find("ForwardOpaque") != std::string::npos);
        CHECK(diagnostic->message.find("PathTrace") != std::string::npos);
    }
    SUBCASE("external producer is explicit")
    {
        definition.AddFeature<TestFeature>(Info("Writer", {}, {{SurfaceDepth}}), a);
        auto result = RenderPipelineCompiler{}.Compile(std::move(definition), {}, {{{SurfaceDepth}}, {}});
        CHECK(Find(result, PipelineDiagnosticCode::DuplicateProducer));
    }
}

TEST_CASE("FeatureCompiler: duplicate instances and explicitly distinct outputs [renderfeatures]")
{
    Probe a, b;
    auto first = Info("First", {}, {{OpaqueHDR}});
    auto second = Info("Second", {}, {{SkyHDR}});
    SUBCASE("not allowed") { first.allowMultipleInstances = false; }
    SUBCASE("allowed with distinct outputs") {}
    RenderPipelineDefinition definition;
    definition.AddFeature<TestFeature>(first, a);
    definition.AddFeature<TestFeature>(second, b);
    auto result = RenderPipelineCompiler{}.Compile(std::move(definition), {});
    if (!first.allowMultipleInstances) CHECK(Find(result, PipelineDiagnosticCode::DuplicateInstance));
    else { REQUIRE(result.pipeline); CHECK(result.pipeline->ResourceSlots().SlotCount() == 2); }
}

TEST_CASE("FeatureCompiler: stable ordering respects dependencies and submission phases [renderfeatures]")
{
    Probe a, b, c;
    RenderPipelineDefinition definition;
    SUBCASE("required provider declared last")
    {
        definition.AddFeature<TestFeature>(Info("Consumer", {{SurfaceDepth}}), a);
        definition.AddFeature<TestFeature>(Info("Independent"), b);
        definition.AddFeature<TestFeature>(Info("Producer", {}, {{SurfaceDepth}}), c);
        auto result = RenderPipelineCompiler{}.Compile(std::move(definition), {});
        CHECK(Order(result) == std::vector<FeatureInstanceId>{2, 3, 1});
    }
    SUBCASE("declaration order breaks ready ties")
    {
        definition.AddFeature<TestFeature>(Info("Consumer", {{SurfaceDepth}}), a);
        definition.AddFeature<TestFeature>(Info("Producer", {}, {{SurfaceDepth}}), b);
        definition.AddFeature<TestFeature>(Info("Independent"), c);
        auto result = RenderPipelineCompiler{}.Compile(std::move(definition), {});
        CHECK(Order(result) == std::vector<FeatureInstanceId>{2, 1, 3});
    }
    SUBCASE("independent phases")
    {
        definition.AddFeature<TestFeature>(Info("GraphicsB", {}, {}, FeaturePhase::AfterAsync), a);
        definition.AddFeature<TestFeature>(Info("Compute", {}, {}, FeaturePhase::Async), b);
        definition.AddFeature<TestFeature>(Info("GraphicsA"), c);
        auto result = RenderPipelineCompiler{}.Compile(std::move(definition), {});
        CHECK(Order(result) == std::vector<FeatureInstanceId>{3, 2, 1});
    }
}

TEST_CASE("FeatureCompiler: optional providers are ordered or publish graph-local absence [renderfeatures]")
{
    Probe reader, producer;
    RenderPipelineDefinition definition;
    bool sawAbsent = false;
    reader.build = [&](RG::RenderGraph&, RenderFeatureContext& context) {
        sawAbsent = context.resources.TryGet(AmbientOcclusion) == nullptr;
    };
    definition.AddFeature<TestFeature>(Info("Reader", {{AmbientOcclusion, ResourceReadRequirement::Optional}}), reader);
    SUBCASE("no provider")
    {
        auto result = RenderPipelineCompiler{}.Compile(std::move(definition), {});
        REQUIRE(result.pipeline);
        CHECK(BuildOnce(*result.pipeline).success);
        CHECK(sawAbsent);
    }
    SUBCASE("conditional provider")
    {
        auto info = Info("AO", {}, {{AmbientOcclusion, ResourceOutputPresence::Optional}});
        info.activation = FeatureActivation::Conditional;
        definition.AddFeature<TestFeature>(info, producer);
        producer.decision.active = false;
        producer.build = [](RG::RenderGraph&, RenderFeatureContext& context) {
            context.resources.Publish(AmbientOcclusion, GraphTextureRef{{7, 2}, {}});
        };
        auto result = RenderPipelineCompiler{}.Compile(std::move(definition), {});
        CHECK(Order(result) == std::vector<FeatureInstanceId>{2, 1});
        CHECK(BuildOnce(*result.pipeline).success);
        CHECK(sawAbsent);
        CHECK(producer.prepared == 0);
        CHECK(producer.built == 0);
        producer.decision.active = true;
        CHECK(BuildOnce(*result.pipeline).success);
        CHECK_FALSE(sawAbsent);
        producer.decision.active = false;
        CHECK(BuildOnce(*result.pipeline).success);
        CHECK(sawAbsent);
        CHECK(producer.built == 1);
        CHECK(producer.described == 1); // No topology rediscovery across builds.
    }
    SUBCASE("optional output cannot satisfy a required read")
    {
        definition.AddFeature<TestFeature>(Info("RequiredReader", {{AmbientOcclusion}}), producer);
        auto result = RenderPipelineCompiler{}.Compile(std::move(definition), {},
            {{{AmbientOcclusion, ResourceOutputPresence::Optional}}, {}});
        CHECK(Find(result, PipelineDiagnosticCode::MissingProducer));
    }
}

TEST_CASE("FeatureCompiler: resource capability and explicit cycles report actual paths [renderfeatures]")
{
    Probe a, b, unrelated;
    RenderPipelineDefinition definition;
    SUBCASE("resources")
    {
        definition.AddFeature<TestFeature>(Info("A", {{SkyHDR}}, {{OpaqueHDR}}), a);
        definition.AddFeature<TestFeature>(Info("B", {{OpaqueHDR}}, {{SkyHDR}}), b);
        definition.AddFeature<TestFeature>(Info("BlockedButNotCyclic", {{OpaqueHDR}}), unrelated);
    }
    SUBCASE("explicit")
    {
        auto first = definition.AddFeature<TestFeature>(Info("A"), a);
        auto second = definition.AddFeature<TestFeature>(Info("B"), b);
        definition.Before(first, second); definition.Before(second, first);
    }
    SUBCASE("capabilities")
    {
        auto first = Info("A"); first.capabilities = {{&Scene}, {{&Geometry}}, {}};
        auto second = Info("B"); second.capabilities = {{&Geometry}, {{&Scene}}, {}};
        definition.AddFeature<TestFeature>(first, a);
        definition.AddFeature<TestFeature>(second, b);
    }
    auto result = RenderPipelineCompiler{}.Compile(std::move(definition), {});
    const auto* error = Find(result, PipelineDiagnosticCode::OrderingCycle);
    REQUIRE(error);
    CHECK_FALSE(result.pipeline);
    REQUIRE(error->cycle.size() == 3);
    CHECK(error->cycle.front() == error->cycle.back());
    CHECK(error->message.find("A") != std::string::npos);
    CHECK(error->message.find("B") != std::string::npos);
    CHECK(error->message.find("BlockedButNotCyclic") == std::string::npos);
}

TEST_CASE("FeatureCompiler: capabilities stay explicit and distinct [renderfeatures]")
{
    Probe a, b;
    auto info = Info("QueryConsumer");
    RenderPipelineDefinition definition;
    SUBCASE("unsupported device requirement")
    {
        info.capabilities.deviceRequirements = {&RayQuery};
        definition.AddFeature<TestFeature>(info, a);
        auto result = RenderPipelineCompiler{}.Compile(std::move(definition), {{&RayPipeline}});
        CHECK(Find(result, PipelineDiagnosticCode::UnsupportedCapability));
    }
    SUBCASE("supported requirement")
    {
        info.capabilities.deviceRequirements = {&RayQuery};
        definition.AddFeature<TestFeature>(info, a);
        auto result = RenderPipelineCompiler{}.Compile(std::move(definition), {{&RayQuery}});
        CHECK(result.pipeline);
    }
    SUBCASE("no hidden provider insertion")
    {
        info.capabilities.consumes = {{&Scene, true}};
        definition.AddFeature<TestFeature>(info, a);
        auto result = RenderPipelineCompiler{}.Compile(std::move(definition), {});
        CHECK(Find(result, PipelineDiagnosticCode::MissingCapability));
        CHECK_FALSE(result.pipeline);
    }
    SUBCASE("duplicate providers")
    {
        info.capabilities.provides = {&Scene};
        definition.AddFeature<TestFeature>(info, a);
        definition.AddFeature<TestFeature>(Info("OtherProvider"), b);
        auto result = RenderPipelineCompiler{}.Compile(std::move(definition), {}, {{}, {&Scene}});
        CHECK(Find(result, PipelineDiagnosticCode::DuplicateCapability));
    }
    SUBCASE("incompatible techniques")
    {
        info.type = &TypeA; info.ordering.incompatibleTypes = {&TypeB};
        auto other = Info("Other"); other.type = &TypeB;
        definition.AddFeature<TestFeature>(info, a);
        definition.AddFeature<TestFeature>(other, b);
        auto result = RenderPipelineCompiler{}.Compile(std::move(definition), {});
        CHECK(Find(result, PipelineDiagnosticCode::IncompatibleFeatures));
    }
}

TEST_CASE("FeatureCompiler: reverse phase dependencies and invalid instance references fail [renderfeatures]")
{
    Probe a, b;
    RenderPipelineDefinition definition;
    SUBCASE("graphics B feeds async")
    {
        definition.AddFeature<TestFeature>(Info("LateProducer", {}, {{SurfaceDepth}}, FeaturePhase::AfterAsync), a);
        definition.AddFeature<TestFeature>(Info("ComputeConsumer", {{SurfaceDepth}}, {}, FeaturePhase::Async), b);
        auto result = RenderPipelineCompiler{}.Compile(std::move(definition), {});
        const auto* diagnostic = Find(result, PipelineDiagnosticCode::PhaseViolation);
        REQUIRE(diagnostic);
        CHECK(diagnostic->resource.identity == SurfaceDepth.identity);
    }
    SUBCASE("explicit phase reversal")
    {
        auto late = definition.AddFeature<TestFeature>(Info("Late", {}, {}, FeaturePhase::AfterAsync), a);
        auto early = definition.AddFeature<TestFeature>(Info("Early"), b);
        definition.Before(late, early);
        auto result = RenderPipelineCompiler{}.Compile(std::move(definition), {});
        CHECK(Find(result, PipelineDiagnosticCode::PhaseViolation));
    }
    SUBCASE("unknown ID")
    {
        definition.AddFeature<TestFeature>(Info("A"), a);
        definition.Before(1, 9);
        auto result = RenderPipelineCompiler{}.Compile(std::move(definition), {});
        CHECK(Find(result, PipelineDiagnosticCode::InvalidOrdering));
    }
}

TEST_CASE("FeatureCompiler: bloom precedes grid mutation despite declaration order [renderfeatures]")
{
    Probe grid, bloom, tonemap;
    RenderPipelineDefinition definition;
    definition.AddFeature<TestFeature>(Info("Grid", {{ResolvedHDR}},
        {{GridHDR, ResourceOutputPresence::Required, ResourceKeyRef{ResolvedHDR}, true}}, FeaturePhase::AfterAsync), grid);
    definition.AddFeature<TestFeature>(Info("Tonemap", {{GridHDR}}, {}, FeaturePhase::AfterAsync), tonemap);
    definition.AddFeature<TestFeature>(Info("Bloom", {{ResolvedHDR}}, {}, FeaturePhase::AfterAsync), bloom);
    SUBCASE("conflicting explicit order") { definition.Before(1, 3); }
    auto result = RenderPipelineCompiler{}.Compile(std::move(definition), {}, {{{ResolvedHDR}}, {}});
    if (!result.pipeline)
    {
        const auto* error = Find(result, PipelineDiagnosticCode::OrderingCycle);
        REQUIRE(error);
        CHECK(error->message.find("ResolvedHDR") != std::string::npos);
        CHECK(error->message.find("Bloom") != std::string::npos);
        CHECK(error->message.find("Grid") != std::string::npos);
    }
    else CHECK(Order(result) == std::vector<FeatureInstanceId>{3, 1, 2});
}

TEST_CASE("FeatureCompiler: alias branches preserve old consumers and reject sibling mutations [renderfeatures]")
{
    Probe alias, mutate, read;
    RenderPipelineDefinition definition;
    SUBCASE("non-mutating alias branch")
    {
        definition.AddFeature<TestFeature>(Info("Mutation", {{OpaqueHDR}},
            {{SkyHDR, ResourceOutputPresence::Required, ResourceKeyRef{OpaqueHDR}, true}}), mutate);
        definition.AddFeature<TestFeature>(Info("Alias", {{OpaqueHDR}},
            {{ResolvedHDR, ResourceOutputPresence::Required, ResourceKeyRef{OpaqueHDR}, false}}), alias);
        definition.AddFeature<TestFeature>(Info("OldConsumer", {{ResolvedHDR}}), read);
        auto result = RenderPipelineCompiler{}.Compile(std::move(definition), {}, {{{OpaqueHDR}}, {}});
        CHECK(Order(result) == std::vector<FeatureInstanceId>{2, 3, 1});
    }
    SUBCASE("two mutations of the same stage")
    {
        definition.AddFeature<TestFeature>(Info("Sky", {{OpaqueHDR}},
            {{SkyHDR, ResourceOutputPresence::Required, ResourceKeyRef{OpaqueHDR}, true}}), alias);
        definition.AddFeature<TestFeature>(Info("Fog", {{OpaqueHDR}},
            {{FoggedHDR, ResourceOutputPresence::Required, ResourceKeyRef{OpaqueHDR}, true}}), mutate);
        auto result = RenderPipelineCompiler{}.Compile(std::move(definition), {}, {{{OpaqueHDR}}, {}});
        CHECK(Find(result, PipelineDiagnosticCode::OrderingCycle));
    }
    SUBCASE("mutation of an ancestor stage")
    {
        definition.AddFeature<TestFeature>(Info("Sky", {{OpaqueHDR}},
            {{SkyHDR, ResourceOutputPresence::Required, ResourceKeyRef{OpaqueHDR}, true}}), alias);
        definition.AddFeature<TestFeature>(Info("Fog", {{SkyHDR}},
            {{FoggedHDR, ResourceOutputPresence::Required, ResourceKeyRef{SkyHDR}, true}}), mutate);
        definition.AddFeature<TestFeature>(Info("OldConsumer", {{OpaqueHDR}}), read);
        auto result = RenderPipelineCompiler{}.Compile(std::move(definition), {}, {{{OpaqueHDR}}, {}});
        CHECK(Order(result) == std::vector<FeatureInstanceId>{3, 1, 2});
    }
}

TEST_CASE("FeatureCompiler: invalid aliases and disabled outputs are rejected [renderfeatures]")
{
    Probe probe;
    RenderPipelineDefinition definition;
    auto info = Info("Transform", {}, {{SkyHDR}});
    auto expected = PipelineDiagnosticCode::InvalidContract;
    SUBCASE("mutation without source") { info.resources.writes[0].mutatesInput = true; }
    SUBCASE("source without declared read") { info.resources.writes[0].aliasesInput = ResourceKeyRef{OpaqueHDR}; }
    SUBCASE("conditional required output without fallback")
    {
        info.activation = FeatureActivation::Conditional;
        expected = PipelineDiagnosticCode::InvalidDisabledOutput;
    }
    definition.AddFeature<TestFeature>(info, probe);
    auto result = RenderPipelineCompiler{}.Compile(std::move(definition), {});
    CHECK(Find(result, expected));
    CHECK_FALSE(result.pipeline);
}

TEST_CASE("FeatureCompiler: disabled allocating transforms pass through without registration [renderfeatures]")
{
    Probe transform, consumer;
    RenderPipelineDefinition definition;
    auto info = Info("TAA", {{TransparentHDR}}, {{ResolvedHDR}});
    info.activation = FeatureActivation::Conditional;
    info.resources.writes[0].disabledPassthrough = ResourceKeyRef{TransparentHDR};
    transform.decision.active = false;
    definition.AddFeature<TestFeature>(info, transform);
    RG::ResourceHandle observed;
    consumer.build = [&](RG::RenderGraph&, RenderFeatureContext& context) { observed = context.resources.Get(ResolvedHDR).handle; };
    definition.AddFeature<TestFeature>(Info("Consumer", {{ResolvedHDR}}), consumer);
    auto result = RenderPipelineCompiler{}.Compile(std::move(definition), {}, {{{TransparentHDR}}, {}});
    REQUIRE(result.pipeline);
    GraphTextureRef input{{11, 3}, {}};
    const std::array bindings{RenderInputBinding::Present(TransparentHDR, input)};
    FrameRenderInputs frame; frame.resources = bindings;
    CHECK(BuildOnce(*result.pipeline, frame).success);
    CHECK(observed == input.handle);
    CHECK(transform.built == 0);
    CHECK(transform.prepared == 0);
}

TEST_CASE("FeatureCompiler: external publications validate before feature preparation [renderfeatures]")
{
    Probe probe;
    probe.build = [](RG::RenderGraph&, RenderFeatureContext& context) { context.resources.Get(SurfaceDepth); };
    RenderPipelineDefinition definition;
    definition.AddFeature<TestFeature>(Info("Consumer", {{SurfaceDepth}}), probe);
    auto result = RenderPipelineCompiler{}.Compile(std::move(definition), {}, {{{SurfaceDepth}}, {}});
    REQUIRE(result.pipeline);
    GraphTextureRef value{{1, 2}, {}};
    FrameRenderInputs frame; ViewRenderInputs view;
    std::vector<RenderInputBinding> bindings;
    bool valid = false;
    SUBCASE("missing") {}
    SUBCASE("explicitly absent") { bindings.push_back(RenderInputBinding::Absent(SurfaceDepth)); }
    SUBCASE("wrong type")
    {
        const RenderResourceKey<GraphBufferRef> wrong{SurfaceDepth.identity};
        // Wrong-type absence still carries its expected C++ type.
        bindings.push_back(RenderInputBinding::Absent(wrong));
    }
    SUBCASE("undeclared") { bindings.push_back(RenderInputBinding::Present(Normal, value)); }
    SUBCASE("duplicate across frame and view")
    {
        bindings.push_back(RenderInputBinding::Present(SurfaceDepth, value)); view.resources = bindings;
    }
    SUBCASE("valid view input") { bindings.push_back(RenderInputBinding::Present(SurfaceDepth, value)); valid = true; }
    frame.resources = bindings;
    const auto build = BuildOnce(*result.pipeline, frame, view);
    CHECK(build.success == valid);
    CHECK(probe.prepared == (valid ? 1 : 0));
    if (!valid) CHECK(Find(build, PipelineDiagnosticCode::InvalidInput));
}

TEST_CASE("FeatureCompiler: native contributions prepare completely before building in compiled order [renderfeatures]")
{
    std::vector<std::string> trace;
    Probe a, b; a.trace = &trace; b.trace = &trace;
    a.build = [](RG::RenderGraph& graph, RenderFeatureContext&) { AddPass(graph, RG::QueueFamily::Graphics); };
    b.build = [](RG::RenderGraph& graph, RenderFeatureContext&) { AddPass(graph, RG::QueueFamily::AsyncCompute); };
    RenderPipelineDefinition definition;
    definition.AddFeature<TestFeature>(Info("Compute", {}, {}, FeaturePhase::Async), b);
    definition.AddFeature<TestFeature>(Info("Graphics"), a);
    auto result = RenderPipelineCompiler{}.Compile(std::move(definition), {});
    REQUIRE(result.pipeline);
    Memory::LinearAllocator scratch(4096); RG::RenderGraph graph(scratch);
    CHECK(result.pipeline->Build(graph, {}, {}, scratch).success);
    CHECK(trace == std::vector<std::string>{"prepare:Graphics", "prepare:Compute", "build:Graphics", "build:Compute"});
    REQUIRE(graph.GetPasses().size() == 2);
    CHECK(graph.GetPasses()[0].queueFamily == RG::QueueFamily::Graphics);
    CHECK(graph.GetPasses()[1].queueFamily == RG::QueueFamily::AsyncCompute);
    graph.Compile(); // Caller, not a feature or compiled composition, compiles the native RG.
    result.pipeline->ReleaseView({73});
    CHECK(a.releasedView.value == 73);
    CHECK(b.released == 1);
}

TEST_CASE("FeatureCompiler: build-time contracts and actual routing are checked [renderfeatures]")
{
    Probe probe;
    auto info = Info("Adapter");
    auto expected = PipelineDiagnosticCode::BuildContractViolation;
    SUBCASE("missing output") { info.resources.writes = {{SkyHDR}}; }
    SUBCASE("undeclared write")
    {
        info.resources.reads = {{SkyHDR, ResourceReadRequirement::Optional}};
        probe.build = [](RG::RenderGraph&, RenderFeatureContext& context) { context.resources.Publish(SkyHDR, GraphTextureRef{}); };
    }
    SUBCASE("undeclared read")
    {
        info.resources.writes = {{SkyHDR, ResourceOutputPresence::Optional}};
        probe.build = [](RG::RenderGraph&, RenderFeatureContext& context) { context.resources.TryGet(SkyHDR); };
    }
    SUBCASE("always-active feature disables itself")
    {
        probe.decision.active = false; expected = PipelineDiagnosticCode::InvalidActivation;
    }
    SUBCASE("async declaration registers graphics")
    {
        info.phase = FeaturePhase::Async; expected = PipelineDiagnosticCode::PhaseViolation;
        probe.build = [](RG::RenderGraph& graph, RenderFeatureContext&) { AddPass(graph, RG::QueueFamily::Graphics); };
    }
    RenderPipelineDefinition definition;
    definition.AddFeature<TestFeature>(info, probe);
    auto result = RenderPipelineCompiler{}.Compile(std::move(definition), {});
    REQUIRE(result.pipeline);
    const auto build = BuildOnce(*result.pipeline);
    CHECK_FALSE(build.success);
    const auto* diagnostic = Find(build, expected);
    REQUIRE(diagnostic);
    CHECK(diagnostic->feature == 1);
}

TEST_CASE("FeatureCompiler: legacy host routing and graphics-only export are respected [renderfeatures]")
{
    Probe probe;
    auto phase = FeaturePhase::BeforeAsync;
    bool expected = false;
    auto queue = RG::QueueFamily::Graphics;
    SUBCASE("before-async cannot follow a host async pass") {}
    SUBCASE("async cannot follow host graphics B") { phase = FeaturePhase::Async; queue = RG::QueueFamily::AsyncCompute; }
    SUBCASE("graphics-only after phase is legal") { phase = FeaturePhase::AfterAsync; expected = true; }
    probe.build = [queue](RG::RenderGraph& graph, RenderFeatureContext&) { AddPass(graph, queue); };
    RenderPipelineDefinition definition;
    definition.AddFeature<TestFeature>(Info("Adapter", {}, {}, phase), probe);
    auto result = RenderPipelineCompiler{}.Compile(std::move(definition), {});
    REQUIRE(result.pipeline);
    Memory::LinearAllocator scratch(4096); RG::RenderGraph graph(scratch);
    if (!expected) AddPass(graph, RG::QueueFamily::AsyncCompute);
    if (phase == FeaturePhase::Async) AddPass(graph, RG::QueueFamily::Graphics);
    const auto build = result.pipeline->Build(graph, {}, {}, scratch);
    CHECK(build.success == expected);
    if (!expected) CHECK(Find(build, PipelineDiagnosticCode::PhaseViolation));
}

TEST_CASE("FeatureCompiler: transitive demand activates shared providers once per build [renderfeatures]")
{
    Probe base, provider, first, second;
    auto scene = Info("SceneProvider"); scene.activation = FeatureActivation::DemandDriven;
    scene.capabilities = {{&Scene}, {{&Geometry}}, {}};
    auto geometry = Info("GeometryProvider"); geometry.activation = FeatureActivation::DemandDriven;
    geometry.capabilities.provides = {&Geometry};
    auto consumer = Info("Consumer"); consumer.capabilities.consumes = {{&Scene, true}};
    const std::array<const RenderCapabilityIdentity*, 1> requests{&Scene};
    RenderPipelineDefinition definition;
    definition.AddFeature<TestFeature>(consumer, first);
    definition.AddFeature<TestFeature>(scene, provider);
    definition.AddFeature<TestFeature>(consumer, second);
    definition.AddFeature<TestFeature>(geometry, base);
    auto result = RenderPipelineCompiler{}.Compile(std::move(definition), {});
    REQUIRE(result.pipeline);
    CHECK(BuildOnce(*result.pipeline).success);
    CHECK(provider.prepared == 0);
    CHECK(base.built == 0);
    first.decision.requests = requests;
    second.decision.requests = requests;
    provider.decision.active = false; // Demand, rather than the provider's own toggle, controls it.
    CHECK(BuildOnce(*result.pipeline).success);
    CHECK(provider.prepared == 1);
    CHECK(base.built == 1);
    first.decision.requests = {}; second.decision.requests = {};
    CHECK(BuildOnce(*result.pipeline).success);
    CHECK(provider.built == 1);
    CHECK(base.built == 1);
    CHECK(base.described == 1);
    result.pipeline->ReleaseView({99});
    CHECK(base.released == 1); // Release reaches dormant providers too.
}

TEST_CASE("FeatureCompiler: dynamic capability requests and external availability are validated [renderfeatures]")
{
    Probe probe;
    auto info = Info("Consumer");
    const std::array<const RenderCapabilityIdentity*, 1> requests{&Scene};
    PipelineInputContract inputs;
    bool declared = false, available = false;
    SUBCASE("undeclared dynamic request") { probe.decision.requests = requests; }
    SUBCASE("external capability unavailable") { declared = true; }
    SUBCASE("external capability available") { declared = true; available = true; }
    if (declared) { info.capabilities.consumes = {{&Scene}}; inputs.capabilities = {&Scene}; }
    RenderPipelineDefinition definition;
    definition.AddFeature<TestFeature>(info, probe);
    auto result = RenderPipelineCompiler{}.Compile(std::move(definition), {}, inputs);
    REQUIRE(result.pipeline);
    FrameRenderInputs frame;
    if (available) frame.capabilities = requests;
    const auto build = BuildOnce(*result.pipeline, frame);
    if (!declared) CHECK(Find(build, PipelineDiagnosticCode::UndeclaredCapabilityRequest));
    else if (frame.capabilities.empty()) CHECK(Find(build, PipelineDiagnosticCode::MissingCapability));
    else CHECK(build.success);
}

TEST_CASE("FeatureCompiler: failed replacement retains the previous valid composition [renderfeatures]")
{
    Probe good, bad;
    RenderPipelineDefinition first;
    first.AddFeature<TestFeature>(Info("Good"), good);
    auto result = RenderPipelineCompiler{}.Compile(std::move(first), {});
    std::unique_ptr<CompiledRenderPipeline> current;
    CHECK(result.ReplaceIfValid(current));
    auto* previous = current.get();
    RenderPipelineDefinition replacement;
    replacement.AddFeature<TestFeature>(Info("Bad", {{MotionVectors}}), bad);
    auto failed = RenderPipelineCompiler{}.Compile(std::move(replacement), {});
    CHECK_FALSE(failed.ReplaceIfValid(current));
    CHECK(current.get() == previous);
    CHECK(BuildOnce(*current).success);
    CHECK(bad.prepared == 0);
}

TEST_CASE("FeatureCompiler: failed diagnostics retain instance-owned keys [renderfeatures]")
{
    class OwnedKeyFeature final : public IRenderFeature
    {
        ResourceKeyIdentity m_Key{"InstanceOwnedMissingKey"};
    public:
        FeatureInfo Describe() const override
        {
            return Info("Owned", {{RenderResourceKey<GraphTextureRef>{&m_Key}}});
        }
        FeatureFrameDecision Evaluate(const FeaturePrepareContext&) const override { return {}; }
        void Build(RG::RenderGraph&, RenderFeatureContext&) override {}
    };
    RenderPipelineDefinition definition;
    definition.AddFeature<OwnedKeyFeature>();
    auto failed = RenderPipelineCompiler{}.Compile(std::move(definition), {});
    const auto* diagnostic = Find(failed, PipelineDiagnosticCode::MissingProducer);
    REQUIRE(diagnostic);
    CHECK(diagnostic->resource.identity->name == "InstanceOwnedMissingKey");
}

TEST_CASE("FeatureCompiler: empty compositions build without graph contributions [renderfeatures]")
{
    auto result = RenderPipelineCompiler{}.Compile({}, {});
    REQUIRE(result.pipeline);
    CHECK(result.pipeline->FeatureOrder().empty());
    Memory::LinearAllocator scratch(4096); RG::RenderGraph graph(scratch);
    CHECK(result.pipeline->Build(graph, {}, {}, scratch).success);
    CHECK(graph.GetPasses().empty());
    CHECK(scratch.GetUsedMemory() == 0);
}

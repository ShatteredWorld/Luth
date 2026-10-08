#include <doctest/doctest.h>
#include "luth/renderer/debug/ProfilingProvenance.h"
#include "luth/renderer/backend/vulkan/VulkanBackend.h"
#include "luth/memory/LinearAllocator.h"
using namespace Luth;
namespace { struct Data {}; }

TEST_CASE("ProfilingProvenance: completion requires the same timeline generation and both queues")
{
    SubmissionCompletionToken token{7, 12, 5, 20, 1, true};
    CHECK_FALSE(SubmissionCompletionProgress{7, 11, 5}.Contains(token));
    CHECK_FALSE(SubmissionCompletionProgress{7, 12, 4}.Contains(token));
    CHECK(SubmissionCompletionProgress{7, 12, 5}.Contains(token));
    CHECK(SubmissionCompletionProgress{7, 100, 100}.Contains(token));
    CHECK_FALSE(SubmissionCompletionProgress{8, 100, 100}.Contains(token));
    token.computeValue = 0;
    CHECK(SubmissionCompletionProgress{7, 12, 0}.Contains(token));
    token.valid = false;
    CHECK_FALSE(SubmissionCompletionProgress{7, 100, 100}.Contains(token));
    token.valid = true; token.graphicsValue = 0;
    CHECK_FALSE(SubmissionCompletionProgress{7, 100, 100}.Contains(token));
    // A cold native backend rejects completion without dereferencing VulkanContext.
    VulkanBackend cold;
    CHECK_FALSE(cold.IsSubmissionComplete(token));
    CHECK_FALSE(cold.IsSubmissionComplete({}));
}

TEST_CASE("ProfilingProvenance: dense pass identities preserve routing and culled graph gaps")
{
    Memory::LinearAllocator scratch(65536); RG::RenderGraph graph(scratch);
    graph.AddPass<Data>("Dead", [](Data&, RG::RenderPassBuilder&) {}, [](Data&, RG::RenderPassContext&) {});
    graph.AddPass<Data>("Depth", [](Data&, RG::RenderPassBuilder& b) { b.SetHasSideEffect(); }, [](Data&, RG::RenderPassContext&) {});
    graph.AddComputePass<Data>("Clusters", RG::QueueFamily::AsyncCompute,
        [](Data&, RG::RenderPassBuilder& b) { b.SetHasSideEffect(); }, [](Data&, RG::RenderPassContext&) {});
    graph.AddComputePass<Data>("Bloom", [](Data&, RG::RenderPassBuilder& b) { b.SetHasSideEffect(); }, [](Data&, RG::RenderPassContext&) {});
    graph.Compile();
    ProfilingProvenance store; store.Prepare({1}, 2, 33, graph);
    const auto* record = store.Find({1}); REQUIRE(record);
    REQUIRE(record->passes.size() == 3);
    CHECK(record->passes[0].graphIndex == 1);
    CHECK(record->passes[0].phase == ProfileSubmissionPhase::GraphicsA);
    CHECK(record->passes[1].phase == ProfileSubmissionPhase::AsyncCompute);
    CHECK(record->passes[2].phase == ProfileSubmissionPhase::GraphicsB);
    CHECK(record->passes[2].compute);
    CHECK(record->renderFrameIndex == 33);
    CHECK(record->resourceGeneration == 2);
    CHECK(record->topologyGeneration == 1);
    CHECK_FALSE(record->completion.valid);
    CHECK(store.Submit({1}, 33, {8, 22, 3, 34, 0, true}));
    const auto frozen = *store.Find({1});
    CHECK(frozen.completion.frameIndex == 34);
    CHECK(frozen.completion.viewSlot == 0);
    store.Prepare({1}, 2, 34, graph);
    CHECK(store.Find({1})->topologyGeneration == 1);
    CHECK_FALSE(store.Find({1})->completion.valid);
    CHECK(frozen.renderFrameIndex == 33);
    CHECK(frozen.completion.valid);
    store.Prepare({2}, 1, 34, graph);
    CHECK(store.Find({2})->topologyGeneration == 1);
    store.Prepare({1}, 3, 35, graph);
    CHECK(store.Find({1})->topologyGeneration == 2);
    store.Release({1}); CHECK(store.Find({1}) == nullptr);
    CHECK(store.Find({2}) != nullptr);
}

TEST_CASE("ProfilingProvenance: topology changes and invalid submissions cannot reuse positional samples")
{
    Memory::LinearAllocator scratch(65536); RG::RenderGraph graph(scratch), changed(scratch);
    graph.AddPass<Data>("Draw", [](Data&, RG::RenderPassBuilder& b) { b.SetHasSideEffect(); }, [](Data&, RG::RenderPassContext&) {});
    changed.AddComputePass<Data>("Draw", [](Data&, RG::RenderPassBuilder& b) { b.SetHasSideEffect(); }, [](Data&, RG::RenderPassContext&) {});
    graph.Compile(); changed.Compile();
    ProfilingProvenance store;
    CHECK_THROWS_AS(store.Prepare({}, 1, 0, graph), std::invalid_argument);
    CHECK_THROWS_AS(store.Prepare({1}, 0, 0, graph), std::invalid_argument);
    CHECK_FALSE(store.Submit({1}, 0, {}));
    store.Prepare({1}, 1, 0, graph);
    store.Prepare({1}, 1, 1, changed);
    CHECK(store.Find({1})->topologyGeneration == 2);
    CHECK_FALSE(store.Submit({1}, 2, {9, 100, 100, 2, 0, true}));
    CHECK_FALSE(store.Find({1})->completion.valid);
    CHECK(store.Submit({1}, 1, {}));
    CHECK_FALSE(SubmissionCompletionProgress{9, 100, 100}.Contains(store.Find({1})->completion));
    store.Prepare({1}, 1, 2, graph);
    CHECK(store.Find({1})->topologyGeneration == 3);
}

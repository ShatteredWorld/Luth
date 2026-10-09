#include <doctest/doctest.h>
#include "luth/renderer/backend/vulkan/VulkanMeshDeformation.h"
#include "luth/renderer/backend/vulkan/VulkanAccelerationStructure.h"
#include "luth/renderer/resources/Mesh.h"
#include <limits>
#include "luth/renderer/backend/vulkan/TlasBuilder.h"
#include "luth/renderer/subsystems/RtSubsystem.h"
#include "luth/renderer/backend/vulkan/VulkanRtMeshResources.h"
using namespace Luth;

TEST_CASE("MeshDeformation: interleaved current and previous slices preserve absolute frame parity")
{
    const auto layout = MeshDeformationLayout::ForVertices(37);
    CHECK(layout.regionBytes == 37 * 72);
    CHECK(layout.TotalBytes() == 2 * 37 * 72);
    for (const u64 frame : {u64(0), u64(1), u64(2), u64(3), u64(0xffffffff),
        u64(0x100000000), std::numeric_limits<u64>::max()})
    {
        const auto current = layout.CurrentOffset(frame), previous = layout.PreviousOffset(frame);
        CHECK(current != previous);
        CHECK(current + previous == layout.regionBytes);
        CHECK(current == (frame & 1u) * layout.regionBytes);
        CHECK(current + layout.regionBytes <= layout.TotalBytes());
        CHECK(previous + layout.regionBytes <= layout.TotalBytes());
    }
    CHECK_THROWS_AS(MeshDeformationLayout::ForVertices(0), std::invalid_argument);
    const auto large = MeshDeformationLayout::ForVertices(std::numeric_limits<u32>::max());
    CHECK(large.regionBytes == u64(std::numeric_limits<u32>::max()) * 72);
    CHECK(large.TotalBytes() == large.regionBytes * 2);
}

TEST_CASE("MeshDeformation: cold resources never produce nonzero current or previous addresses")
{
    VKMeshDeformation resource;
    VKAccelerationStructure rigid;
    CHECK_FALSE(resource.GetBuffer()); CHECK(resource.GetVertexCount() == 0);
    CHECK(resource.GetLayout().TotalBytes() == 0);
    for (const u64 frame : {u64(0), u64(1), std::numeric_limits<u64>::max()})
    {
        CHECK(resource.GetCurrentAddress(frame) == 0); CHECK(resource.GetPreviousAddress(frame) == 0);
        CHECK(rigid.GetDeformedBdaCurr(static_cast<u32>(frame)) == 0);
        CHECK(rigid.GetDeformedBdaPrev(static_cast<u32>(frame)) == 0);
    }
    CHECK_FALSE(rigid.GetDeformation());
    Mesh empty({}, {}, 0, false);
    CHECK_FALSE(empty.EnsureDeformation()); CHECK_FALSE(empty.GetBlas());
    Mesh noNativeSource({}, {}, 1, true);
    CHECK_FALSE(noNativeSource.EnsureDeformation());
}

TEST_CASE("MeshDeformation: mesh owns one resource and BLAS borrows the same lifetime")
{
    auto resource = std::make_shared<VKMeshDeformation>();
    std::weak_ptr<VKMeshDeformation> lifetime = resource;
    std::shared_ptr<VKAccelerationStructure> borrower;
    {
        Mesh mesh({}, {}, 0, false);
        mesh.SetDeformation(resource);
        CHECK(mesh.EnsureDeformation() == resource); // No second allocation, even without BLAS.
        CHECK_FALSE(mesh.GetBlas());
        borrower = std::make_shared<VKAccelerationStructure>(mesh.GetDeformation());
        mesh.SetBlas(borrower);
        CHECK(borrower->GetDeformation() == mesh.GetDeformation());
        CHECK(resource.use_count() == 3);
        resource.reset();
    }
    CHECK_FALSE(lifetime.expired()); // Pending RT work can outlive the model owner.
    borrower.reset();
    CHECK(lifetime.expired());
}

TEST_CASE("MeshDeformation: dropping RT ownership preserves the mesh resource and rejects replacement")
{
    Mesh mesh({}, {}, 0, false);
    auto resource = std::make_shared<VKMeshDeformation>();
    mesh.SetDeformation(resource);
    mesh.SetBlas(std::make_shared<VKAccelerationStructure>(resource));
    mesh.SetBlas({});
    CHECK(mesh.GetDeformation() == resource); CHECK(mesh.EnsureDeformation() == resource);
    CHECK_FALSE(mesh.GetBlas());
    CHECK_NOTHROW(mesh.SetDeformation(resource));
    CHECK_THROWS_AS(mesh.SetDeformation({}), std::invalid_argument);
    CHECK_THROWS_AS(mesh.SetDeformation(std::make_shared<VKMeshDeformation>()), std::invalid_argument);
    CHECK(mesh.GetDeformation() == resource);
    Mesh other({}, {}, 1, true);
    CHECK_THROWS_AS(other.SetDeformation(resource), std::invalid_argument);
}

TEST_CASE("MeshDeformation: upload readiness rejects incomplete and missing native bindings")
{
    const MeshDeformationBindings bindings{0x1000, 0x2000, 0x3000, 37, 9};
    CHECK_FALSE(bindings.IsReady(0));
    CHECK_FALSE(bindings.IsReady(8));
    CHECK(bindings.IsReady(9));
    CHECK(bindings.IsReady(std::numeric_limits<u64>::max()));
    auto updated = bindings;
    updated.sourceUploadFence = 10; // A new upload invalidates the next CPU snapshot.
    CHECK_FALSE(updated.IsReady(9));
    CHECK(bindings.IsReady(9)); // Already-frozen input is immutable.
    CHECK(updated.IsReady(10));
    updated.sourceUploadFence = 0;
    CHECK(updated.IsReady(0));
    for (u32 missing = 0; missing < 4; ++missing)
    {
        auto incomplete = bindings;
        if (missing == 0) incomplete.source = 0;
        if (missing == 1) incomplete.current = 0;
        if (missing == 2) incomplete.previous = 0;
        if (missing == 3) incomplete.vertexCount = 0;
        CHECK_FALSE(incomplete.IsReady(std::numeric_limits<u64>::max()));
    }
    CHECK_FALSE(MeshDeformationBindings{}.IsReady(std::numeric_limits<u64>::max()));
}

TEST_CASE("MeshDeformation: raster bootstrap aliases current and cold preparation needs no RT or device")
{
    const MeshDeformationBindings bindings{0x1000, 0x2000, 0x3000, 37, 0};
    CHECK(bindings.PreviousForRaster(true) == bindings.current);
    CHECK(bindings.PreviousForRaster(false) == bindings.previous);
    Mesh mesh({}, {}, 0, true);
    mesh.SetDeformation(std::make_shared<VKMeshDeformation>());
    CHECK_FALSE(mesh.GetBlas());
    for (const u64 frame : {u64(0), u64(1), u64(0x100000001)})
    {
        const auto cold = mesh.GetDeformation()->PrepareBindings(frame);
        CHECK(cold.source == 0);
        CHECK(cold.current == 0);
        CHECK(cold.previous == 0);
        CHECK(cold.vertexCount == 0);
        CHECK_FALSE(cold.IsReady(std::numeric_limits<u64>::max()));
        CHECK(cold.PreviousForRaster(true) == 0);
    }
}

TEST_CASE("MeshDeformation: prepared TLAS empty and reuse preserve paired bindings without recording")
{
    const std::unordered_map<UUID, u32, UUIDHash> slots;
    const auto empty = TlasBuilder::PrepareTlas({}, 0, {}, slots, 7, false);
    CHECK(empty.result.tlas == VK_NULL_HANDLE);
    CHECK(empty.result.geomTableBDA == 0);
    CHECK(empty.result.instanceCount == 0);
    CHECK(empty.result.blasReadyGen == 7);
    CHECK_FALSE(empty.command);
    CHECK_NOTHROW(empty.Record(VK_NULL_HANDLE));
    auto prior = empty.result;
    prior.tlas = reinterpret_cast<VkAccelerationStructureKHR>(uintptr_t(17));
    prior.geomTableBDA = 29;
    const auto reused = TlasBuilder::PrepareTlas({}, 1, prior, slots, 7, false);
    CHECK(reused.result.reused);
    CHECK(reused.result.tlas == prior.tlas);
    CHECK(reused.result.geomTableBDA == prior.geomTableBDA);
    CHECK_FALSE(reused.command);
    CHECK_NOTHROW(reused.Record(VK_NULL_HANDLE));
    const auto changed = TlasBuilder::PrepareTlas({}, 2, prior, slots, 8, false);
    CHECK_FALSE(changed.result.reused);
    CHECK(changed.result.tlas == VK_NULL_HANDLE);
    CHECK(changed.result.geomTableBDA == 0);
    CHECK(changed.result.blasReadyGen == 8);
}
TEST_CASE("MeshDeformation: prepared TLAS copies retain native build input addresses")
{
    static const TlasBuildCommand* expected;
    static u32 calls;
    calls = 0;
    auto command = std::make_shared<TlasBuildCommand>();
    expected = command.get();
    command->buildInfo.pGeometries = &command->geom;
    command->range.primitiveCount = 3;
    command->record = [](VkCommandBuffer, uint32_t count,
        const VkAccelerationStructureBuildGeometryInfoKHR* info,
        const VkAccelerationStructureBuildRangeInfoKHR* const* ranges) {
        ++calls;
        CHECK(count == 1);
        CHECK(info == &expected->buildInfo);
        CHECK(info->pGeometries == &expected->geom);
        CHECK(ranges == &expected->rangePtr);
        CHECK(*ranges == &expected->range);
        CHECK((*ranges)->primitiveCount == 3);
    };
    PreparedTlasBuild prepared{{}, command};
    const std::weak_ptr<const TlasBuildCommand> lifetime = command;
    auto retained = prepared;
    command.reset(); prepared = {};
    CHECK_FALSE(lifetime.expired());
    retained.Record(VK_NULL_HANDLE);
    CHECK(calls == 1);
    retained = {};
    CHECK(lifetime.expired());
}

TEST_CASE("MeshDeformation: scheduled BLAS readiness is local and commits only at recording")
{
    auto first = std::make_shared<VKAccelerationStructure>();
    auto update = std::make_shared<VKAccelerationStructure>();
    update->MarkBuildRecorded(8);
    auto unrelated = std::make_shared<VKAccelerationStructure>();
    auto command = std::make_shared<BlasBuildCommand>();
    command->frameAbs = 9;
    command->infos.resize(2); command->geoms.resize(2);
    command->ranges.resize(2); command->rangePtrs.resize(2);
    command->targets = {first, update}; command->firstBuilds = {true, false};
    for (size_t i = 0; i < 2; ++i)
    {
        command->infos[i].pGeometries = &command->geoms[i];
        command->rangePtrs[i] = &command->ranges[i];
    }
    static const BlasBuildCommand* expected;
    static u32 calls;
    expected = command.get(); calls = 0;
    command->record = [](VkCommandBuffer, uint32_t count,
        const VkAccelerationStructureBuildGeometryInfoKHR* infos,
        const VkAccelerationStructureBuildRangeInfoKHR* const* ranges) {
        ++calls;
        CHECK(count == 2);
        CHECK(infos == expected->infos.data());
        CHECK(ranges == expected->rangePtrs.data());
        CHECK_FALSE(expected->targets[0]->IsBuildRecorded());
        for (size_t i = 0; i < count; ++i)
        {
            CHECK(infos[i].pGeometries == &expected->geoms[i]);
            CHECK(ranges[i] == &expected->ranges[i]);
        }
    };
    PreparedBlasBuild batch{{command}};
    const PreparedBlasBuild scheduled[]{batch};
    CHECK(batch.FirstBuildCount() == 1);
    CHECK_FALSE(first->IsBuildRecorded());
    CHECK_FALSE(IsBlasReadyForTlas(first.get(), {}, 9));
    CHECK(IsBlasReadyForTlas(first.get(), scheduled, 9));
    CHECK_FALSE(IsBlasReadyForTlas(first.get(), scheduled, 10));
    CHECK(IsBlasReadyForTlas(update.get(), {}, 9));
    CHECK_FALSE(IsBlasReadyForTlas(unrelated.get(), scheduled, 9));
    CHECK_FALSE(IsBlasReadyForTlas(nullptr, scheduled, 9));
    const std::weak_ptr<const BlasBuildCommand> lifetime = command;
    auto retained = batch;
    command.reset(); batch = {};
    CHECK_FALSE(lifetime.expired());
    retained.Record(VK_NULL_HANDLE);
    CHECK(first->IsBuildRecorded());
    CHECK(first->GetBuildFrameAbs() == 9);
    CHECK(update->GetBuildFrameAbs() == 8);
    retained.Record(VK_NULL_HANDLE); // Copies share recording state; no duplicate native build.
    scheduled[0].Record(VK_NULL_HANDLE);
    CHECK(calls == 1);
}

TEST_CASE("MeshDeformation: abandoned BLAS batches retry without publishing recorded readiness")
{
    auto target = std::make_shared<VKAccelerationStructure>();
    u32 retries = 0;
    {
        auto command = std::make_shared<BlasBuildCommand>();
        command->targets = {target}; command->firstBuilds = {true};
        command->abandon = [&]() { ++retries; };
        PreparedBlasBuild prepared{{command}};
        command.reset();
        CHECK(prepared.IsScheduled(target.get(), 0));
        CHECK_FALSE(target->IsBuildRecorded());
    }
    CHECK(retries == 1);
    CHECK_FALSE(target->IsBuildRecorded());
    const auto empty = TlasBuilder::PrepareSkinnedBLASes({}, 0);
    CHECK(empty.commands.empty());
    CHECK(empty.FirstBuildCount() == 0);
    CHECK_FALSE(empty.IsScheduled(target.get(), 0));
    CHECK_NOTHROW(empty.Record(VK_NULL_HANDLE));
}

TEST_CASE("MeshDeformation: RT scene preparation shares paired bindings across views and refreshes next frame")
{
    RtSubsystem rt;
    const std::unordered_map<UUID, u32, UUIDHash> slots;
    CHECK_FALSE(rt.IsPreparedFor(3));
    const auto first = rt.PrepareScene({}, 3, slots, false);
    CHECK(rt.IsPreparedFor(3));
    CHECK_FALSE(first->recorded);
    CHECK_FALSE(first->HasSceneData());
    CHECK(first->GetTlas() == rt.GetTlas());
    CHECK(first->GetGeometryTableBDA() == rt.GetGeometryTableBDA());
    CHECK(rt.PrepareScene({}, 3, slots, false) == first);
    const auto next = rt.PrepareScene({}, 4, slots, false);
    CHECK(next != first);
    CHECK(rt.IsPreparedFor(4));
    CHECK_FALSE(rt.IsPreparedFor(3));
    CHECK(first->frameIndex == 3);
    CHECK(next->frameIndex == 4);
}

TEST_CASE("MeshDeformation: RT scene records frozen BLAS barrier TLAS exactly once")
{
    static std::vector<u32> events;
    events.clear();
    auto blas = std::make_shared<BlasBuildCommand>();
    blas->infos.resize(1); blas->ranges.resize(1); blas->rangePtrs = {&blas->ranges[0]};
    blas->record = [](VkCommandBuffer, uint32_t,
        const VkAccelerationStructureBuildGeometryInfoKHR*,
        const VkAccelerationStructureBuildRangeInfoKHR* const*) { events.push_back(1); };
    auto tlas = std::make_shared<TlasBuildCommand>();
    tlas->record = [](VkCommandBuffer, uint32_t,
        const VkAccelerationStructureBuildGeometryInfoKHR*,
        const VkAccelerationStructureBuildRangeInfoKHR* const*) { events.push_back(3); };
    auto scene = std::make_shared<PreparedRtScene>();
    scene->blas[0].commands = {blas};
    scene->tlas.command = tlas;
    scene->emptyFallback = reinterpret_cast<VkAccelerationStructureKHR>(uintptr_t(7));
    CHECK(scene->GetTlas() == scene->emptyFallback);
    CHECK_FALSE(scene->HasSceneData());
    scene->tlas.result.tlas = reinterpret_cast<VkAccelerationStructureKHR>(uintptr_t(11));
    scene->tlas.result.geomTableBDA = 29;
    scene->tlas.result.instanceCount = 1;
    CHECK(scene->HasSceneData());
    CHECK(scene->GetTlas() == scene->tlas.result.tlas);
    CHECK(scene->GetGeometryTableBDA() == 29);
    CHECK(scene->ReuseCandidate().tlas == VK_NULL_HANDLE);
    CHECK(scene->ReuseCandidate().geomTableBDA == 0);
    scene->barrier = [](VkCommandBuffer, const VkDependencyInfo* dependency) {
        events.push_back(2);
        REQUIRE(dependency->memoryBarrierCount == 1);
        const auto& memory = *dependency->pMemoryBarriers;
        CHECK(memory.srcStageMask == VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR);
        CHECK(memory.srcAccessMask == VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR);
        CHECK(memory.dstStageMask == VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR);
        CHECK(memory.dstAccessMask == VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR);
    };
    std::shared_ptr<const PreparedRtScene> retained = scene;
    blas.reset(); tlas.reset(); scene.reset();
    retained->Record(VK_NULL_HANDLE);
    CHECK(events == std::vector<u32>{1, 2, 3});
    CHECK(retained->recorded);
    CHECK(retained->ReuseCandidate().tlas == retained->GetTlas());
    CHECK(retained->ReuseCandidate().geomTableBDA == retained->GetGeometryTableBDA());
    const auto secondView = retained;
    secondView->Record(VK_NULL_HANDLE);
    CHECK(events == std::vector<u32>{1, 2, 3});
}

TEST_CASE("MeshDeformation: RT mesh resources are lazy and shared across scene requests")
{
    static u32 rigidCalls, deformableCalls;
    rigidCalls = deformableCalls = 0;
    VulkanRtMeshResources resources(
        [](const Mesh&) { ++rigidCalls; return std::make_shared<VKAccelerationStructure>(); },
        [](const Mesh& mesh) { ++deformableCalls; return std::make_shared<VKAccelerationStructure>(mesh.GetDeformation()); });
    Mesh rigid({}, {}, 0, false);
    Mesh wind({}, {}, 0, false);
    Mesh skinned({}, {}, 0, true);
    auto deformation = std::make_shared<VKMeshDeformation>();
    wind.SetDeformation(deformation);
    skinned.SetDeformation(std::make_shared<VKMeshDeformation>());
    CHECK_FALSE(rigid.GetBlas()); CHECK_FALSE(wind.GetBlas()); CHECK_FALSE(skinned.GetBlas());
    CHECK(rigidCalls == 0); CHECK(deformableCalls == 0);
    auto first = resources.Ensure(rigid);
    CHECK(first);
    CHECK(resources.Ensure(rigid) == first);
    CHECK(rigidCalls == 1);
    const auto windBlas = resources.Ensure(wind);
    CHECK(windBlas);
    CHECK(windBlas->GetDeformation() == deformation);
    CHECK(resources.Ensure(wind) == windBlas);
    CHECK(resources.Ensure(skinned));
    CHECK(resources.Ensure(skinned) == skinned.GetBlas());
    CHECK(deformableCalls == 2);
    wind.SetBlas({});
    CHECK(wind.GetDeformation() == deformation); // RT release never discards raster deformation.
}

TEST_CASE("MeshDeformation: failed lazy RT creation retries and existing resources bypass factories")
{
    static u32 attempts;
    attempts = 0;
    VulkanRtMeshResources resources(
        [](const Mesh&) -> std::shared_ptr<VKAccelerationStructure> {
            if (++attempts == 1) return {};
            return std::make_shared<VKAccelerationStructure>();
        });
    Mesh mesh({}, {}, 0, false);
    CHECK_FALSE(resources.Ensure(mesh));
    CHECK_FALSE(mesh.GetBlas());
    CHECK(attempts == 1);
    auto result = resources.Ensure(mesh);
    CHECK(result);
    CHECK(attempts == 2);
    CHECK(resources.Ensure(mesh) == result);
    CHECK(attempts == 2);
    const auto replacement = std::make_shared<VKAccelerationStructure>();
    mesh.SetBlas(replacement);
    CHECK(resources.Ensure(mesh) == replacement);
    CHECK(attempts == 2);
}

TEST_CASE("MeshDeformation: missing deformation never falls back to rigid RT geometry")
{
    static u32 rigidCalls, deformableCalls;
    rigidCalls = deformableCalls = 0;
    VulkanRtMeshResources resources(
        [](const Mesh&) { ++rigidCalls; return std::make_shared<VKAccelerationStructure>(); },
        [](const Mesh& mesh) -> std::shared_ptr<VKAccelerationStructure> {
            ++deformableCalls;
            if (!mesh.GetDeformation()) return {};
            return std::make_shared<VKAccelerationStructure>(mesh.GetDeformation());
        });
    Mesh wind({}, {}, 0, false);
    CHECK_FALSE(resources.Ensure(wind, true));
    CHECK_FALSE(wind.GetBlas());
    CHECK(rigidCalls == 0);
    CHECK(deformableCalls == 1);
    const auto deformation = std::make_shared<VKMeshDeformation>();
    wind.SetDeformation(deformation); // Independent raster allocation becomes available.
    const auto result = resources.Ensure(wind, true);
    REQUIRE(result);
    CHECK(result->GetDeformation() == deformation);
    CHECK(resources.Ensure(wind) == result);
    CHECK(rigidCalls == 0);
    CHECK(deformableCalls == 2);
}

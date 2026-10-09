#include <doctest/doctest.h>
#include "luth/renderer/backend/vulkan/VulkanMeshDeformation.h"
#include "luth/renderer/backend/vulkan/VulkanAccelerationStructure.h"
#include "luth/renderer/resources/Mesh.h"
#include <limits>
#include "luth/renderer/backend/vulkan/TlasBuilder.h"
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
#include <doctest/doctest.h>
#include "luth/renderer/backend/vulkan/VulkanMeshDeformation.h"
#include "luth/renderer/backend/vulkan/VulkanAccelerationStructure.h"
#include "luth/renderer/resources/Mesh.h"
#include <limits>
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

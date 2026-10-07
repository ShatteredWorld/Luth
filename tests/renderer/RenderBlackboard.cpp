#include <doctest/doctest.h>

#include "luth/renderer/features/RenderBlackboard.h"
#include "luth/renderer/rendergraph/RenderGraph.h"
#include "luth/memory/GPUTaggedPageAllocator.h"
#include "luth/memory/LinearAllocator.h"
#include <array>
#include <string_view>

using namespace Luth;
using namespace Luth::RenderResources;

namespace
{
    template<class F>
    void ExpectError(BlackboardErrorCode expected, F&& operation)
    {
        try
        {
            operation();
            FAIL("Expected a blackboard contract error");
        }
        catch (const BlackboardError& error)
        {
            CHECK(error.Code() == expected);
        }
    }

    GraphTextureRef TextureRef(u32 index = 1, u32 version = 0)
    {
        return { {index, version}, {} };
    }
}

TEST_CASE("RenderBlackboard: present and explicit absent resources [renderfeatures]")
{
    const std::array<ResourceKeyRef, 2> keys{SurfaceDepth, AmbientOcclusion};
    ResourceSlotLayout layout(keys);
    Memory::LinearAllocator scratch(4096);
    RenderBlackboard board(layout, scratch);

    ExpectError(BlackboardErrorCode::UnpublishedResource, [&] { board.TryGet(SurfaceDepth); });
    board.Publish(SurfaceDepth, TextureRef(17, 3));
    board.PublishAbsent(AmbientOcclusion);
    CHECK(board.Get(SurfaceDepth).handle == RG::ResourceHandle{17, 3});
    CHECK(board.TryGet(AmbientOcclusion) == nullptr);
    ExpectError(BlackboardErrorCode::AbsentResource, [&] { board.Get(AmbientOcclusion); });
}

TEST_CASE("RenderBlackboard: all forms of duplicate publication fail [renderfeatures]")
{
    const std::array<ResourceKeyRef, 2> keys{SurfaceDepth, AmbientOcclusion};
    ResourceSlotLayout layout(keys);
    Memory::LinearAllocator scratch(4096);
    RenderBlackboard board(layout, scratch);
    board.Publish(SurfaceDepth, TextureRef());
    board.PublishAbsent(AmbientOcclusion);

    ExpectError(BlackboardErrorCode::DuplicatePublication, [&] { board.Publish(SurfaceDepth, TextureRef(2)); });
    ExpectError(BlackboardErrorCode::DuplicatePublication, [&] { board.PublishAbsent(SurfaceDepth); });
    ExpectError(BlackboardErrorCode::DuplicatePublication, [&] { board.Publish(AmbientOcclusion, TextureRef()); });
    ExpectError(BlackboardErrorCode::DuplicatePublication, [&] { board.PublishAbsent(AmbientOcclusion); });
    CHECK(board.Get(SurfaceDepth).handle.index == 1);
    CHECK(board.TryGet(AmbientOcclusion) == nullptr);
}

TEST_CASE("RenderBlackboard: canonical identity is distinct from names and types [renderfeatures]")
{
    const ResourceKeyIdentity first{"SameName"};
    const ResourceKeyIdentity second{"SameName"};
    const RenderResourceKey<GraphTextureRef> a{&first};
    const RenderResourceKey<GraphTextureRef> b{&second};
    const RenderResourceKey<GraphBufferRef> wrongType{&first};
    const std::array<ResourceKeyRef, 3> keys{a, a, b};
    ResourceSlotLayout layout(keys);
    CHECK(layout.SlotCount() == 2);
    CHECK(layout.Find(a) == 0);
    CHECK(layout.Find(b) == 1);

    Memory::LinearAllocator scratch(4096);
    RenderBlackboard board(layout, scratch);
    board.Publish(a, TextureRef(11));
    board.Publish(b, TextureRef(12));
    CHECK(board.Get(a).handle.index == 11);
    CHECK(board.Get(b).handle.index == 12);
    ExpectError(BlackboardErrorCode::TypeMismatch, [&] { board.TryGet(wrongType); });
    ExpectError(BlackboardErrorCode::TypeMismatch, [&] { board.Publish(wrongType, GraphBufferRef{}); });
    const std::array<ResourceKeyRef, 2> incompatible{a, wrongType};
    ExpectError(BlackboardErrorCode::TypeMismatch, [&] { ResourceSlotLayout invalid(incompatible); });
    ExpectError(BlackboardErrorCode::UnknownKey, [&] { board.TryGet(Normal); });
    ExpectError(BlackboardErrorCode::InvalidKey, [&] { board.TryGet(RenderResourceKey<GraphTextureRef>{nullptr}); });
}

TEST_CASE("RenderBlackboard: feature contracts validate accesses and advertised outputs [renderfeatures]")
{
    const std::array<ResourceKeyRef, 4> keys{SurfaceDepth, Normal, AmbientOcclusion, SkyHDR};
    ResourceSlotLayout layout(keys);
    Memory::LinearAllocator scratch(4096);
    RenderBlackboard board(layout, scratch);
    board.Publish(SurfaceDepth, TextureRef(5));
    board.Publish(Normal, TextureRef(6));
    const ResourceContract ao{
        { {SurfaceDepth} },
        { {AmbientOcclusion, ResourceOutputPresence::Optional} }
    };
    {
        RenderBlackboard::FeatureScope scope(board, ao, "GTAO");
        CHECK(board.Get(SurfaceDepth).handle.index == 5);
        ExpectError(BlackboardErrorCode::UndeclaredRead, [&] { board.TryGet(Normal); });
        ExpectError(BlackboardErrorCode::UndeclaredWrite, [&] { board.Publish(SkyHDR, TextureRef()); });
        ExpectError(BlackboardErrorCode::MissingOutput, [&] { scope.ValidateOutputs(); });
        ExpectError(BlackboardErrorCode::NestedFeatureScope, [&] {
            RenderBlackboard::FeatureScope nested(board, ao, "Nested");
        });
        board.PublishAbsent(AmbientOcclusion);
        scope.ValidateOutputs();
    }
    const ResourceContract sky{ {}, { {SkyHDR} } };
    {
        RenderBlackboard::FeatureScope scope(board, sky, "Sky");
        ExpectError(BlackboardErrorCode::RequiredOutputAbsent, [&] { board.PublishAbsent(SkyHDR); });
        board.Publish(SkyHDR, TextureRef(7));
        scope.ValidateOutputs();
    }
    const ResourceContract consumer{
        { {AmbientOcclusion, ResourceReadRequirement::Optional}, {SkyHDR} }, {}
    };
    {
        RenderBlackboard::FeatureScope scope(board, consumer, "Forward");
        CHECK(board.TryGet(AmbientOcclusion) == nullptr);
        CHECK(board.Get(SkyHDR).handle.index == 7);
        scope.ValidateOutputs();
    }
}

TEST_CASE("RenderBlackboard: feature scope unwinds on failure [renderfeatures]")
{
    const std::array<ResourceKeyRef, 1> keys{SurfaceDepth};
    ResourceSlotLayout layout(keys);
    Memory::LinearAllocator scratch(4096);
    RenderBlackboard board(layout, scratch);
    const ResourceContract empty{};
    try
    {
        RenderBlackboard::FeatureScope scope(board, empty, "BrokenFeature");
        board.Publish(SurfaceDepth, TextureRef());
        FAIL("Undeclared write must fail");
    }
    catch (const BlackboardError& error)
    {
        CHECK(std::string_view(error.what()).find("BrokenFeature") != std::string_view::npos);
        CHECK(std::string_view(error.what()).find("SurfaceDepth") != std::string_view::npos);
        CHECK(error.Key().type == &RenderResourceType<GraphTextureRef>);
    }
    // The failed feature did not publish and its scoped restrictions are no longer active.
    board.Publish(SurfaceDepth, TextureRef(2));
    CHECK(board.Get(SurfaceDepth).handle.index == 2);
}

TEST_CASE("RenderBlackboard: independent graphs and arena reuse reset absence [renderfeatures]")
{
    const std::array<ResourceKeyRef, 1> keys{AmbientOcclusion};
    ResourceSlotLayout layout(keys);
    Memory::LinearAllocator scratch(4096);
    {
        RenderBlackboard first(layout, scratch);
        RenderBlackboard second(layout, scratch);
        first.Publish(AmbientOcclusion, TextureRef(8));
        ExpectError(BlackboardErrorCode::UnpublishedResource, [&] { second.TryGet(AmbientOcclusion); });
        second.PublishAbsent(AmbientOcclusion);
        CHECK(first.Get(AmbientOcclusion).handle.index == 8);
        CHECK(second.TryGet(AmbientOcclusion) == nullptr);
    }
    scratch.Reset();
    RenderBlackboard next(layout, scratch);
    ExpectError(BlackboardErrorCode::UnpublishedResource, [&] { next.TryGet(AmbientOcclusion); });
    next.PublishAbsent(AmbientOcclusion);
    CHECK(next.TryGet(AmbientOcclusion) == nullptr);
}

TEST_CASE("RenderBlackboard: physical buffer slices retain offsets and ranges [renderfeatures]")
{
    Memory::GPUSubRegion objects{}, indirect{};
    // Distinct tagged slices may share a backing buffer. Their logical RG identity and
    // descriptor offsets must survive publication without deduplication by VkBuffer.
    objects.offset = 256; objects.size = 128;
    indirect.buffer = objects.buffer;
    indirect.offset = 1024; indirect.size = 512;
    const std::array<ResourceKeyRef, 2> keys{ObjectData, InitializedIndirectData};
    ResourceSlotLayout layout(keys);
    Memory::LinearAllocator scratch(4096);
    RenderBlackboard board(layout, scratch);
    board.Publish(ObjectData, GraphBufferRef{ {1, 2}, {&objects, objects.offset, objects.size} });
    board.Publish(InitializedIndirectData, GraphBufferRef{ {2, 4}, {&indirect, indirect.offset, indirect.size} });
    CHECK(board.Get(ObjectData).binding.slice == &objects);
    CHECK(board.Get(ObjectData).binding.offset == 256);
    CHECK(board.Get(ObjectData).binding.size == 128);
    CHECK(board.Get(InitializedIndirectData).binding.slice == &indirect);
    CHECK(board.Get(InitializedIndirectData).binding.offset == 1024);
    CHECK(board.Get(InitializedIndirectData).binding.size == 512);
    CHECK(board.Get(ObjectData).handle.index != board.Get(InitializedIndirectData).handle.index);
}

TEST_CASE("RenderBlackboard: stage aliases forward the producer handle without imports [renderfeatures]")
{
    const std::array<ResourceKeyRef, 2> keys{OpaqueHDR, SkyHDR};
    ResourceSlotLayout layout(keys);
    Memory::LinearAllocator scratch(4096);
    RG::RenderGraph graph(scratch);
    RG::TextureDesc desc{"HDR", 16, 16, RG::TextureFormat::RGBA16_Float};
    const RG::ResourceHandle handle = graph.RegisterResource(desc);
    const auto resourcesBefore = graph.GetResources().size();
    RenderBlackboard board(layout, scratch);
    board.Publish(OpaqueHDR, GraphTextureRef{handle, {nullptr, 2, 1, 3, 1}});
    const ResourceContract sky{
        { {OpaqueHDR} }, { {SkyHDR, ResourceOutputPresence::Required, ResourceKeyRef{OpaqueHDR}, true} }
    };
    {
        RenderBlackboard::FeatureScope scope(board, sky, "DisabledSky");
        board.Publish(SkyHDR, board.Get(OpaqueHDR));
        scope.ValidateOutputs();
    }
    CHECK(board.Get(SkyHDR).handle == handle);
    CHECK(board.Get(SkyHDR).binding.baseMip == 2);
    CHECK(board.Get(SkyHDR).binding.baseLayer == 3);
    CHECK(graph.GetResources().size() == resourcesBefore);
}

TEST_CASE("RenderBlackboard: storage is bounded and aligned in the frame arena [renderfeatures]")
{
    constexpr ResourceKeyIdentity tinyIdentity{"Tiny"};
    constexpr ResourceKeyIdentity alignedIdentity{"Aligned"};
    struct alignas(std::max_align_t) AlignedValue { u64 first; u64 second; };
    const RenderResourceKey<u8> tiny{&tinyIdentity};
    const RenderResourceKey<AlignedValue> aligned{&alignedIdentity};
    const std::array<ResourceKeyRef, 4> keys{tiny, aligned, SurfaceDepth, ShadowCascades};
    ResourceSlotLayout layout(keys);
    Memory::LinearAllocator scratch(4096);
    // Exercise allocation on a new page, including after an unaligned previous allocation.
    scratch.Allocate(4081, 1);
    RenderBlackboard board(layout, scratch);
    const auto bytesAfterConstruction = scratch.GetUsedMemory();
    CHECK(board.StorageBytes() == layout.ValueBytes() + layout.SlotCount());
    CHECK(board.StorageBytes() <= sizeof(u8) + sizeof(AlignedValue) + sizeof(GraphTextureRef) +
        sizeof(ShadowCascadeRefs) + keys.size() * alignof(std::max_align_t));
    board.Publish(tiny, u8{7});
    board.Publish(aligned, AlignedValue{11, 13});
    CHECK(reinterpret_cast<uintptr_t>(&board.Get(aligned)) % alignof(AlignedValue) == 0);
    u64 sum = 0;
    for (int i = 0; i < 1000; ++i)
        sum += board.Get(tiny) + board.Get(aligned).second;
    CHECK(sum == 20000);
    CHECK(scratch.GetUsedMemory() == bytesAfterConstruction);
}

TEST_CASE("RenderBlackboard: empty layouts require no frame storage [renderfeatures]")
{
    ResourceSlotLayout layout(std::span<const ResourceKeyRef>{});
    Memory::LinearAllocator scratch(4096);
    RenderBlackboard board(layout, scratch);
    CHECK(board.StorageBytes() == 0);
    CHECK(scratch.GetUsedMemory() == 0);
    ExpectError(BlackboardErrorCode::UnknownKey, [&] { board.TryGet(SurfaceDepth); });
}

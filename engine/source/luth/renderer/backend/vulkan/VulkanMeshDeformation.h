#pragma once
#include "VulkanAllocator.h"
#include <memory>
#include <stdexcept>

namespace Luth
{
    class Mesh;
    class VKVertexBuffer;
    // CPU-frozen source/output bindings. Readiness is independent of AS build state.
    struct MeshDeformationBindings
    {
        VkDeviceAddress source = 0;
        VkDeviceAddress current = 0;
        VkDeviceAddress previous = 0;
        u32 vertexCount = 0;
        u64 sourceUploadFence = 0;
        bool HasStorage() const
        {
            return source && current && previous && vertexCount;
        }
        bool IsReady(u64 completedUploadValue) const
        { return HasStorage() && sourceUploadFence <= completedUploadValue; }
        VkDeviceAddress PreviousForRaster(bool firstFrame) const
        { return firstFrame ? current : previous; }
    };
    // Interleaved Vertex regions. Absolute render frame parity matches the existing shaders/refits.
    struct MeshDeformationLayout
    {
        static constexpr u64 kVertexStride = 72;
        u64 regionBytes = 0;
        static MeshDeformationLayout ForVertices(u32 count)
        {
            if (!count) throw std::invalid_argument("Deformation requires vertices");
            return {u64(count) * kVertexStride};
        }
        u64 TotalBytes() const { return 2 * regionBytes; }
        u64 CurrentOffset(u64 frame) const { return (frame & 1u) * regionBytes; }
        u64 PreviousOffset(u64 frame) const { return ((frame ^ 1u) & 1u) * regionBytes; }
    };

    // Owns the persistent compute output independently of acceleration structures.
    // Mesh and optional RT borrowers share this object, never a second allocation.
    class VKMeshDeformation
    {
    public:
        VKMeshDeformation() = default;
        ~VKMeshDeformation();
        VKMeshDeformation(const VKMeshDeformation&) = delete;
        VKMeshDeformation& operator=(const VKMeshDeformation&) = delete;
        static std::shared_ptr<VKMeshDeformation> Create(const Mesh&);
        VkBuffer GetBuffer() const { return m_Buffer; }
        u32 GetVertexCount() const { return m_VertexCount; }
        const MeshDeformationLayout& GetLayout() const { return m_Layout; }
        MeshDeformationBindings PrepareBindings(u64 renderFrameIndex) const;
        VkDeviceAddress GetCurrentAddress(u64 frame) const
        { return m_Address ? m_Address + m_Layout.CurrentOffset(frame) : 0; }
        VkDeviceAddress GetPreviousAddress(u64 frame) const
        { return m_Address ? m_Address + m_Layout.PreviousOffset(frame) : 0; }
    private:
        // Keep source storage and its latest upload fence with the deformation owner.
        std::shared_ptr<VKVertexBuffer> m_Source;
        VkBuffer m_Buffer = VK_NULL_HANDLE;
        VmaAllocation m_Allocation = nullptr;
        VkDeviceAddress m_Address = 0;
        u32 m_VertexCount = 0;
        MeshDeformationLayout m_Layout;
    };
}

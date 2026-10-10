#include "luthpch.h"
#include "VulkanMeshDeformation.h"
#include "VulkanMeshBufferUsage.h"
#include "VulkanContext.h"
#include "VulkanBuffer.h"
#include "luth/renderer/resources/Mesh.h"
#include "luth/renderer/resources/Model.h"
#include <vk_mem_alloc.h>

namespace Luth
{
    static_assert(sizeof(Vertex) == MeshDeformationLayout::kVertexStride, "deformed vertex ABI");
    static_assert(offsetof(Vertex, Position) == 0 && offsetof(Vertex, Normal) == 12,
        "deformed position/normal ABI");
    static_assert(offsetof(Vertex, TexCoord0) == 24 && offsetof(Vertex, TexCoord1) == 32,
        "deformed UV ABI");
    static_assert(offsetof(Vertex, Tangent) == 40 && offsetof(Vertex, Color) == 56,
        "deformed tangent/color ABI");
    static_assert(sizeof(SkinnedVertex) == 104, "skin input ABI");
    static_assert(offsetof(SkinnedVertex, Position) == 0 && offsetof(SkinnedVertex, Normal) == 12,
        "skin position/normal ABI");
    static_assert(offsetof(SkinnedVertex, TexCoord0) == 24 && offsetof(SkinnedVertex, TexCoord1) == 32,
        "skin UV ABI");
    static_assert(offsetof(SkinnedVertex, Tangent) == 40 && offsetof(SkinnedVertex, Color) == 56,
        "skin tangent/color ABI");
    static_assert(offsetof(SkinnedVertex, BoneIDs) == 72 && offsetof(SkinnedVertex, BoneWeights) == 88,
        "skin bone ABI");

    VKMeshDeformation::~VKMeshDeformation()
    {
        if (!m_Buffer) return; // Also permits allocation-free headless ownership tests.
        VulkanContext::Get().PushDeletion([buffer = m_Buffer, allocation = m_Allocation] {
            VulkanAllocator::FreeBuffer(buffer, allocation);
        });
    }

    MeshDeformationBindings VKMeshDeformation::PrepareBindings(u64 renderFrameIndex) const
    {
        if (!m_Source) return {};
        return {m_Source->GetDeviceAddress(), GetCurrentAddress(renderFrameIndex),
            GetPreviousAddress(renderFrameIndex), m_VertexCount, m_Source->GetUploadFence()};
    }

    std::shared_ptr<VKMeshDeformation> VKMeshDeformation::Create(const Mesh& mesh)
    {
        auto source = std::dynamic_pointer_cast<VKVertexBuffer>(mesh.GetVertexBuffer());
        if (!mesh.GetVertexCount() || !source) return {};
        auto result = std::make_shared<VKMeshDeformation>();
        result->m_Source = std::move(source);
        result->m_VertexCount = mesh.GetVertexCount();
        result->m_Layout = MeshDeformationLayout::ForVertices(result->m_VertexCount);
        auto& context = VulkanContext::Get();
        VkBufferCreateInfo info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        info.size = result->m_Layout.TotalBytes();
        info.usage = VulkanMeshBufferUsage::Deformation(context.SupportsRayTracing());
        info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        context.ApplyConcurrentSharing(info);
        result->m_Allocation = VulkanAllocator::AllocateBuffer(info,
            VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE, result->m_Buffer);
        VkBufferDeviceAddressInfo address{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};
        address.buffer = result->m_Buffer;
        result->m_Address = vkGetBufferDeviceAddress(context.GetDevice(), &address);
        return result;
    }
}

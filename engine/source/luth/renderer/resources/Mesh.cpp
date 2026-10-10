#include "luthpch.h"

#include "luth/renderer/resources/Mesh.h"
#include "luth/renderer/backend/vulkan/VulkanAccelerationStructure.h"
#include "luth/renderer/backend/vulkan/VulkanMeshDeformation.h"
#include "luth/renderer/Renderer.h"
#include "luth/renderer/RenderBackend.h"

namespace Luth
{
    // Out-of-line for native resources with incomplete types in the public header.
    Mesh::~Mesh() = default;

    void Mesh::SetDeformation(const std::shared_ptr<VKMeshDeformation>& resource)
    {
        if ((m_Deformation && resource != m_Deformation) ||
            (resource && resource->GetVertexCount() != m_VertexCount))
            throw std::invalid_argument("Mesh deformation is immutable and must match its vertex count");
        m_Deformation = resource;
    }

    const std::shared_ptr<VKMeshDeformation>& Mesh::EnsureDeformation()
    {
        if (!m_Deformation) m_Deformation = VKMeshDeformation::Create(*this);
        return m_Deformation;
    }

    std::shared_ptr<Mesh> Mesh::Create(
        const std::shared_ptr<VertexBuffer>& vb,
        const std::shared_ptr<IndexBuffer>& ib,
        uint32_t vertexCount,
        bool isSkinned)
    {
        // Model loading prepares raster resources; the RT domain requests BLAS independently.
        return std::make_shared<Mesh>(vb, ib, vertexCount, isSkinned);
    }
}

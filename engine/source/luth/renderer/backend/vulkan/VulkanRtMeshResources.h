#pragma once

#include "luth/renderer/backend/vulkan/VulkanAccelerationStructure.h"
#include "luth/renderer/resources/Mesh.h"

namespace Luth
{
    // Optional RT-domain service. Model loading prepares raster geometry/deformation only.
    // Called during serialized scene preparation, never during GPU command recording.
    class VulkanRtMeshResources
    {
    public:
        using Factory = std::shared_ptr<VKAccelerationStructure> (*)(const Mesh&);
        VulkanRtMeshResources(Factory rigid = VKAccelerationStructure::CreateStaticBLAS,
            Factory deformable = VKAccelerationStructure::CreateDeformableBLAS)
            : m_Rigid(rigid), m_Deformable(deformable) {}

        const std::shared_ptr<VKAccelerationStructure>& Ensure(Mesh& mesh, bool requiresDeformation = false) const
        {
            if (!mesh.GetBlas())
            {
                // Snapshot classification survives a failed deformation allocation; never downgrade
                // wind/skinned geometry to a rigid BLAS merely because the resource is missing.
                const auto create = requiresDeformation || mesh.IsSkinned() || mesh.GetDeformation() ? m_Deformable : m_Rigid;
                mesh.SetBlas(create(mesh));
            }
            return mesh.GetBlas();
        }

    private:
        Factory m_Rigid;
        Factory m_Deformable;
    };
}

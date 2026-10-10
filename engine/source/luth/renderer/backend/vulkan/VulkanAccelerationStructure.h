#pragma once

#include "luth/core/types/LuthTypes.h"
#include "VulkanAllocator.h"
#include "VulkanMeshDeformation.h"

#include <memory>
#include <utility>
#include <vector>
#include <functional>
#include <span>
#include <vulkan/vulkan.h>

namespace Luth
{
    class Mesh;
    class VKVertexBuffer;
    class VKIndexBuffer;
    class VKAccelerationStructure;

    // Vectors are sized before establishing native self-references; packets share stable storage.
    struct BlasBuildCommand
    {
        std::vector<VkAccelerationStructureBuildGeometryInfoKHR> infos;
        std::vector<VkAccelerationStructureGeometryKHR> geoms;
        std::vector<VkAccelerationStructureBuildRangeInfoKHR> ranges;
        std::vector<const VkAccelerationStructureBuildRangeInfoKHR*> rangePtrs;
        std::vector<std::shared_ptr<VKAccelerationStructure>> targets;
        std::vector<std::shared_ptr<Mesh>> meshes;
        std::vector<bool> firstBuilds;
        PFN_vkCmdBuildAccelerationStructuresKHR record = nullptr;
        u32 frameAbs = 0;
        mutable bool recorded = false;
        std::function<void()> abandon;
        BlasBuildCommand() = default;
        BlasBuildCommand(const BlasBuildCommand&) = delete;
        BlasBuildCommand& operator=(const BlasBuildCommand&) = delete;
        ~BlasBuildCommand() { if (!recorded && abandon) abandon(); }
        void Record(VkCommandBuffer) const;
    };

    struct PreparedBlasBuild
    {
        std::vector<std::shared_ptr<const BlasBuildCommand>> commands;
        u32 FirstBuildCount() const;
        bool IsScheduled(const VKAccelerationStructure*, u32 frameAbs) const;
        void Record(VkCommandBuffer cmd) const
        { for (const auto& command : commands) command->Record(cmd); }
    };

    // RAII wrapper for a single VkAccelerationStructureKHR + its persistent backing VkBuffer.
    // Used for both BLAS (per-mesh, owned by Mesh) and TLAS (per-frame, owned by RtSubsystem).
    // Deformable BLAS borrows the mesh-owned compute output; it never allocates or frees it.
    // Dtor pushes every owned VkBuffer + AS handle into VulkanContext::PushDeletion so they retire N+2 frames
    // out, safe against in-flight cmd buffers referencing the AS in a build / traceRays call.
    class VKAccelerationStructure
    {
    public:
        VKAccelerationStructure() = default;
        explicit VKAccelerationStructure(std::shared_ptr<VKMeshDeformation> deformation)
            : m_Deformation(std::move(deformation)) {}
        ~VKAccelerationStructure();

        VKAccelerationStructure(const VKAccelerationStructure&) = delete;
        VKAccelerationStructure& operator=(const VKAccelerationStructure&) = delete;

        VkAccelerationStructureKHR GetHandle()        const { return m_Handle; }
        VkDeviceAddress            GetDeviceAddress() const { return m_DeviceAddress; }
        bool                       IsDeformable()     const { return m_IsDeformable; }

        // Compatibility bridge until raster/skinning consumers move to Mesh::GetDeformation.
        const std::shared_ptr<VKMeshDeformation>& GetDeformation() const { return m_Deformation; }
        VkDeviceAddress GetDeformedBdaCurr(u32 frameAbs) const
            { return m_Deformation ? m_Deformation->GetCurrentAddress(frameAbs) : 0; }
        VkDeviceAddress GetDeformedBdaPrev(u32 frameAbs) const
            { return m_Deformation ? m_Deformation->GetPreviousAddress(frameAbs) : 0; }
        u32             GetVertexCount()     const { return m_VertexCount; }
        u64             GetUpdateScratchSize() const { return m_UpdateScratchSize; }

        // Deferred-build readiness. The AS object exists (valid device address) before its build is
        // recorded, so IsBuildRecorded() (not GetDeviceAddress) is the TLAS-inclusion predicate.
        bool IsBuildRecorded()     const { return m_BuildRecorded; }
        u32  GetBuildFrameAbs()    const { return m_BuildFrameAbs; }
        u64  GetBuildScratchSize() const { return m_BuildScratchSize; }
        // Marks a deferred BLAS built at `frameAbs`. Used by the batched RefitSkinnedBLASes first-build path
        // (which packs its own build info); RecordBuild sets these directly for the static-drain path.
        void MarkBuildRecorded(u32 frameAbs) { m_BuildRecorded = true; m_BuildFrameAbs = frameAbs; }

        // Per-mesh static BLAS factory. Creates the AS object only + enqueues a deferred build (never blocks):
        // DrainPendingStaticBuilds records the MODE_BUILD on the async-compute AS pass once the VB/IB upload
        // retires. PREFER_FAST_TRACE per NVIDIA RTX best practices (static BLAS optimizes for trace, built once).
        // Until the build is recorded the TLAS gather + the raster draw list skip the mesh (progressive load).
        static std::shared_ptr<VKAccelerationStructure> CreateStaticBLAS(const Mesh& mesh);

        // Borrows an already prepared mesh deformation resource. Initial build/refits
        // preserve ALLOW_UPDATE and current-region parity; no deformation allocation here.
        static std::shared_ptr<VKAccelerationStructure> CreateDeformableBLAS(const Mesh& mesh);

        // In-place refit (MODE_UPDATE_KHR). Requires the BLAS was originally built with
        // ALLOW_UPDATE_BIT_KHR + same primitiveCount/geometry layout/vertex format/index format.
        // The deformed positions buffer must already be populated by the deform compute on `cmd`
        // (or a prior submission); caller is responsible for the ComputeWrite -> AS-build barrier
        // (render graph emits this when both passes share resource handles).
        // scratchBda must be at least GetUpdateScratchSize() bytes, aligned to the
        // minAccelerationStructureScratchOffsetAlignment from VulkanContext::GetAsProperties().
        void Refit(VkCommandBuffer cmd, VkDeviceAddress scratchBda) const;

        // Records the initial MODE_BUILD onto `cmd`, reconstructed from the recipe captured at creation
        // (no Mesh needed, so a deferred pass can drive it). scratchBda must be >= GetBuildScratchSize()
        // and aligned to minAccelerationStructureScratchOffsetAlignment. Deformable reads its CURR
        // deformed region for `frameAbs`; static reads its fixed source VB. Sets IsBuildRecorded().
        void RecordBuild(VkCommandBuffer cmd, VkDeviceAddress scratchBda, u32 frameAbs);
        std::shared_ptr<BlasBuildCommand> PrepareBuild(VkDeviceAddress scratchBda, u32 frameAbs) const;

        // Drains pending static BLAS builds onto `cmd` (the async-compute AS pass): records the MODE_BUILD
        // for each queued static BLAS whose VB/IB upload fence has retired (non-blocking poll), batching one
        // device-local scratch. A mesh evicted before its build cancels via weak_ptr expiry. Returns the
        // number of builds recorded (a null->ready transition the TLAS must fold in). see arch/rendering-pipeline.md
        static u32 DrainPendingStaticBuilds(VkCommandBuffer cmd, u32 frameAbs);
        static PreparedBlasBuild PreparePendingStaticBuilds(u32 frameAbs);

    private:
        VkAccelerationStructureKHR m_Handle          = VK_NULL_HANDLE;
        VkBuffer                   m_StorageBuffer   = VK_NULL_HANDLE;
        VmaAllocation              m_StorageAlloc    = nullptr;
        VkDeviceAddress            m_DeviceAddress   = 0;

        // Deformable-only.
        std::shared_ptr<VKMeshDeformation> m_Deformation;
        u32             m_VertexCount      = 0;
        u32             m_PrimitiveCount   = 0;
        u64             m_UpdateScratchSize = 0;
        bool            m_IsDeformable     = false;

        // Deferred initial build. The native recipe retains source buffers without retaining Mesh.
        bool                                 m_BuildRecorded     = false;
        u32                                  m_BuildFrameAbs     = ~0u;   // frameAbs of RecordBuild; ~0u = built at load
        VkDeviceAddress                      m_BuildVbBda        = 0;     // static source VB (deformable uses CURR region)
        VkDeviceAddress                      m_BuildIbBda        = 0;
        // Retain the buffers behind the native recipe while a prepared build holds this AS.
        std::shared_ptr<VKVertexBuffer>       m_BuildVertexBuffer;
        std::shared_ptr<VKIndexBuffer>        m_BuildIndexBuffer;
        u32                                  m_BuildVertexStride = 0;
        u32                                  m_BuildMaxVertex    = 0;
        u64                                  m_BuildScratchSize  = 0;     // MODE_BUILD scratch (aligned)
        VkGeometryFlagsKHR                   m_GeomFlags         = 0;
        VkBuildAccelerationStructureFlagsKHR m_BuildFlags        = 0;
    };

    // Scheduled readiness requires these retained batches to record before the matching TLAS.
    inline bool IsBlasReadyForTlas(const VKAccelerationStructure* blas,
        std::span<const PreparedBlasBuild> scheduled, u32 frameAbs)
    {
        if (!blas) return false;
        if (blas->IsBuildRecorded()) return true;
        for (const auto& batch : scheduled)
            if (batch.IsScheduled(blas, frameAbs)) return true;
        return false;
    }
}

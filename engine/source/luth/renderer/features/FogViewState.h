#pragma once

#include "luth/core/FrameData.h"
#include "luth/renderer/features/RenderViewState.h"
#include "luth/renderer/resources/Texture.h"
#include "luth/renderer/settings/VolumetricSettings.h"
#include <array>
#include <vulkan/vulkan.h>

namespace Luth
{
    // Physical resources only. Graph imports and handles belong to each execution.
    struct FogViewState
    {
        FogViewState() = default;
        ~FogViewState();
        FogViewState(const FogViewState&) = delete;
        FogViewState& operator=(const FogViewState&) = delete;

        static ViewStateConfig Config(u32 width, u32 height, VolumetricSettings::Quality quality);
        static std::shared_ptr<FogViewState> Create(RenderViewId id, const ViewStateConfig& config,
            const std::array<VkDescriptorSetLayout, 6>& layouts);
        VkDevice device = VK_NULL_HANDLE;
        VkDescriptorPool pool = VK_NULL_HANDLE;
        // Retain the physical source until descriptors retire. An old view handle
        // cannot be recycled while it still identifies this configuration.
        std::shared_ptr<Texture> depthSource;

        // Volumetric fog atlases (RGBA16F). View-frustum-aligned; persistent across frames so the
        // resolve pass can reproject + blend with prev frame's resolved output. Allocated via the
        // 3D VKTexture ctor (STORAGE + SAMPLED, null internal sampler; VolumetricSubsystem owns
        // the shared linear-clamp sampler). Dims pulled from VolumetricSettings::quality preset.
        //
        // volInScatter is the scratch atlas: inject writes pre-integrate per-voxel scatter, then
        // integrate reads + writes the post-integrate cumulative in-place. volInScatterHistA/B
        // ping-pong as the temporal-resolve I/O pair: each frame the resolve pass reads one as
        // "prev resolved" and writes the other as "current resolved". Composite + viz sample the
        // "current resolved" atlas of the active frame parity.
        std::shared_ptr<Texture> volDensity;
        std::shared_ptr<Texture> volInScatter;
        std::shared_ptr<Texture> volInScatterHistA;
        std::shared_ptr<Texture> volInScatterHistB;

        // Volumetric inject density pass. Cycled: b1 (FogVolume SSBO) rewrites per frame against
        // a fresh tagged-heap region; b0 + b2 are stable per-view.
        std::array<VkDescriptorSet, MAX_FRAMES_IN_FLIGHT> volInjectDensityDescSet{};

        // Volumetric inject scatter pass. Cycled: b2-b4 (Light, ClusterGrid, LightIndex SSBOs)
        // rewrite per frame; b0/b1/b5 stable. Reads volDensity written by the density pass via the
        // shared ResourceNode (RG inserts the barrier).
        std::array<VkDescriptorSet, MAX_FRAMES_IN_FLIGHT> volInjectScatterDescSet{};

        // Volumetric integrate pass. Cycled; reads + writes volInScatter (scratch) in-place.
        std::array<VkDescriptorSet, MAX_FRAMES_IN_FLIGHT> volIntegrateDescSet{};

        // Volumetric resolve pass. Cycled; reads scratch + prev history (parity), writes curr
        // history (parity). Temporal accumulation happens here, post-integrate.
        std::array<VkDescriptorSet, MAX_FRAMES_IN_FLIGHT> volResolveDescSet{};

        // Volumetric composite. Cycled: b1 (in-scatter sampler) parity-picks the resolved history
        // atlas (HistA or HistB). b0 (sceneDepth sampler) is stable across slots.
        std::array<VkDescriptorSet, MAX_FRAMES_IN_FLIGHT> volCompositeDescSet{};

        // Volumetric debug viz. Cycled: b2 follows the same ping-pong parity as composite.
        std::array<VkDescriptorSet, MAX_FRAMES_IN_FLIGHT> volVizDescSet{};

        // Volumetric atlas dimensions live here so changing VolumetricSettings::quality at runtime
        // re-allocates the atlases (mirrors width/height resize handling). volQualityCached tracks
        // the value at last allocation; the native domain compares configurations before replacement.
        u32 volDimX = 0, volDimY = 0, volDimZ = 0;
        u32 volQualityCached = ~0u;

    };
    using FogViewStateStore = FeatureViewStates<std::shared_ptr<FogViewState>>;
}

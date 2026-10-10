#pragma once

#include "luth/renderer/rendergraph/RenderGraphResources.h"
#include <array>
#include <string_view>
#include <type_traits>

namespace Luth
{
    class Texture;
    namespace Memory { struct GPUSubRegion; }

    // Borrowed native bindings. Only native adapters resolve these references. Their owners
    // must outlive graph recording; neither a binding nor an RG handle owns a GPU resource.
    struct TextureBindingRef
    {
        const Texture* texture = nullptr;
        u32 baseMip = 0;
        u32 mipCount = 1;
        u32 baseLayer = 0;
        u32 layerCount = 1;
    };

    struct BufferBindingRef
    {
        const Memory::GPUSubRegion* slice = nullptr;
        u64 offset = 0; // Absolute offset in the backing buffer, not an offset within the slice.
        u64 size = 0;
    };

    struct GraphTextureRef
    {
        RG::ResourceHandle handle;
        TextureBindingRef binding;
    };

    struct GraphBufferRef
    {
        RG::BufferHandle handle;
        BufferBindingRef binding;
    };

    struct VisibleDrawRange
    {
        GraphBufferRef indirect;
        u32 firstDraw = 0;
        u32 maxDrawCount = 0;
    };

    struct ShadowCascadeRefs
    {
        std::array<GraphTextureRef, 4> cascades;
    };

    struct CascadeDrawRanges
    {
        std::array<VisibleDrawRange, 4> cascades;
    };

    struct ResourceTypeIdentity
    {
        std::string_view name;
        size_t size;
        size_t alignment;
    };

    template<class T>
    constexpr std::string_view RenderResourceTypeName()
    {
#ifdef _MSC_VER
        return __FUNCSIG__;
#else
        return __PRETTY_FUNCTION__;
#endif
    }

    template<class T>
    inline constexpr ResourceTypeIdentity RenderResourceType{
        RenderResourceTypeName<T>(), sizeof(T), alignof(T)
    };

    // Identity is its address, never its diagnostic name. Define identities once as inline
    // variables (or retain instance-specific identities for the compiled pipeline's lifetime).
    struct ResourceKeyIdentity
    {
        std::string_view name;
    };

    template<class T>
    struct RenderResourceKey
    {
        static_assert(std::is_trivially_copyable_v<T>, "Blackboard values must be borrowed, trivially copyable references");
        static_assert(alignof(T) <= alignof(std::max_align_t), "Frame arena does not support over-aligned values");
        const ResourceKeyIdentity* identity;
    };

    // Internal contract/layout representation retains the expected type, including when a
    // caller accidentally constructs a differently typed key with the same identity.
    struct ResourceKeyRef
    {
        const ResourceKeyIdentity* identity = nullptr;
        const ResourceTypeIdentity* type = nullptr;

        template<class T>
        constexpr ResourceKeyRef(RenderResourceKey<T> key)
            : identity(key.identity), type(&RenderResourceType<T>) {}

        constexpr ResourceKeyRef() = default;
    };

    namespace RenderResources
    {
        // Stage keys identify semantic values; they do not promise distinct physical images.
#define LUTH_RENDER_RESOURCE(Type, Name) \
        inline constexpr ResourceKeyIdentity Name##Identity{#Name}; \
        inline constexpr RenderResourceKey<Type> Name{&Name##Identity}

        LUTH_RENDER_RESOURCE(GraphBufferRef, ObjectData);
        LUTH_RENDER_RESOURCE(GraphBufferRef, InitializedIndirectData);
        LUTH_RENDER_RESOURCE(VisibleDrawRange, CameraVisibleDraws);
        LUTH_RENDER_RESOURCE(CascadeDrawRanges, CascadeVisibleDraws);
        LUTH_RENDER_RESOURCE(GraphTextureRef, PrepassDepth);
        LUTH_RENDER_RESOURCE(GraphTextureRef, SurfaceDepth);
        LUTH_RENDER_RESOURCE(GraphTextureRef, LitDepth);
        LUTH_RENDER_RESOURCE(GraphTextureRef, Normal);
        LUTH_RENDER_RESOURCE(GraphTextureRef, Roughness);
        LUTH_RENDER_RESOURCE(GraphTextureRef, MotionVectors);
        LUTH_RENDER_RESOURCE(GraphTextureRef, MaterialID);
        LUTH_RENDER_RESOURCE(ShadowCascadeRefs, ShadowCascades);
        LUTH_RENDER_RESOURCE(GraphBufferRef, LightData);
        LUTH_RENDER_RESOURCE(GraphBufferRef, ClusterGrid);
        LUTH_RENDER_RESOURCE(GraphBufferRef, LightIndices);
        LUTH_RENDER_RESOURCE(GraphTextureRef, AmbientOcclusion);
        LUTH_RENDER_RESOURCE(GraphTextureRef, FogDensity);
        LUTH_RENDER_RESOURCE(GraphTextureRef, ResolvedFog);
        LUTH_RENDER_RESOURCE(GraphTextureRef, RefractionBackdrop);
        LUTH_RENDER_RESOURCE(GraphTextureRef, OpaquePickingIDs);
        LUTH_RENDER_RESOURCE(GraphTextureRef, FinalPickingIDs);
        LUTH_RENDER_RESOURCE(GraphTextureRef, SelectionMask);
        LUTH_RENDER_RESOURCE(GraphTextureRef, SelectionDepth);
        LUTH_RENDER_RESOURCE(GraphTextureRef, BloomOutput);
        LUTH_RENDER_RESOURCE(GraphTextureRef, OpaqueHDR);
        LUTH_RENDER_RESOURCE(GraphTextureRef, SkyHDR);
        LUTH_RENDER_RESOURCE(GraphTextureRef, FoggedHDR);
        LUTH_RENDER_RESOURCE(GraphTextureRef, TransparentHDR);
        LUTH_RENDER_RESOURCE(GraphTextureRef, ResolvedHDR);
        LUTH_RENDER_RESOURCE(GraphTextureRef, GridHDR);
        LUTH_RENDER_RESOURCE(GraphTextureRef, TonemappedLDR);
        LUTH_RENDER_RESOURCE(GraphTextureRef, VisualizedLDR);
        LUTH_RENDER_RESOURCE(GraphTextureRef, OutlinedLDR);
        LUTH_RENDER_RESOURCE(GraphTextureRef, FinalViewLDR);

#undef LUTH_RENDER_RESOURCE
    }
}

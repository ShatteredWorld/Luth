#pragma once

#include "luth/renderer/features/RenderBlackboard.h"
#include <string>

namespace Luth
{
    struct RenderSnapshot;
    struct DrawList;
    struct CameraParams;
    namespace RG { class RenderGraph; }

    using FeatureInstanceId = u32;
    inline constexpr FeatureInstanceId InvalidFeatureInstance = 0;
    struct RenderViewId { u64 value = 0; bool operator==(const RenderViewId&) const = default; };
    enum class FeaturePhase { BeforeAsync, Async, AfterAsync };
    enum class FeatureActivation { Always, Conditional, DemandDriven };

    struct FeatureTypeIdentity { std::string_view name; };
    using FeatureTypeId = const FeatureTypeIdentity*;
    template<class T>
    inline constexpr FeatureTypeIdentity RenderFeatureType{RenderResourceTypeName<T>()};

    // Extensions declare their own canonical identities. Core raster composition need not
    // include RT feature headers or assume AS, ray queries and RT pipelines are equivalent.
    struct RenderCapabilityIdentity { std::string_view name; };
    struct DeviceCapabilityIdentity { std::string_view name; };
    struct RendererCapabilities
    {
        std::vector<const DeviceCapabilityIdentity*> supported;
    };
    struct CapabilityRead
    {
        const RenderCapabilityIdentity* capability = nullptr;
        bool conditional = false; // Requested by Evaluate only for the relevant native variant.
    };
    struct CapabilityContract
    {
        std::vector<const RenderCapabilityIdentity*> provides;
        std::vector<CapabilityRead> consumes;
        std::vector<const DeviceCapabilityIdentity*> deviceRequirements;
    };
    struct OrderingConstraints
    {
        std::vector<FeatureInstanceId> before;
        std::vector<FeatureInstanceId> after;
        std::vector<FeatureTypeId> incompatibleTypes;
    };
    struct FeatureInfo
    {
        FeatureTypeId type = nullptr; // Definition supplies the concrete C++ type when omitted.
        std::string name;
        FeaturePhase phase = FeaturePhase::BeforeAsync;
        bool allowMultipleInstances = false;
        FeatureActivation activation = FeatureActivation::Always;
        ResourceContract resources;
        CapabilityContract capabilities;
        OrderingConstraints ordering;
    };

    // Typed external publications. Bindings borrow values until Build returns; graph jobs
    // must capture their immutable references by value. No public untyped value dictionary.
    class RenderInputBinding
    {
    public:
        template<class T>
        static RenderInputBinding Present(RenderResourceKey<T> key, const T& value)
        {
            return {key, &value};
        }
        template<class T>
        static RenderInputBinding Present(RenderResourceKey<T>, const T&&) = delete;
        template<class T>
        static RenderInputBinding Absent(RenderResourceKey<T> key) { return {key, nullptr}; }
    private:
        friend class CompiledRenderPipeline;
        RenderInputBinding(ResourceKeyRef key, const void* value) : m_Key(key), m_Value(value) {}
        ResourceKeyRef m_Key;
        const void* m_Value;
    };
    struct PipelineInputContract
    {
        std::vector<ResourceWrite> resources;
        std::vector<const RenderCapabilityIdentity*> capabilities;
    };
    struct FrameRenderInputs
    {
        u64 renderFrameIndex = 0;
        const RenderSnapshot* snapshot = nullptr;
        const DrawList* draws = nullptr;
        std::span<const RenderInputBinding> resources;
        std::span<const RenderCapabilityIdentity* const> capabilities;
    };
    struct ViewRenderInputs
    {
        RenderViewId id;
        u64 resourceGeneration = 0;
        u32 width = 0;
        u32 height = 0;
        const CameraParams* camera = nullptr;
        std::span<const RenderInputBinding> resources;
        std::span<const RenderCapabilityIdentity* const> capabilities;
    };
    struct FeaturePrepareContext
    {
        const FrameRenderInputs& frame;
        const ViewRenderInputs& view;
        Memory::LinearAllocator& scratch;
    };
    struct RenderFeatureContext
    {
        const FrameRenderInputs& frame;
        const ViewRenderInputs& view;
        RenderBlackboard& resources;
        Memory::LinearAllocator& scratch;
    };
    using FrameCapabilityRequests = std::span<const RenderCapabilityIdentity* const>;
    struct FeatureFrameDecision
    {
        bool active = true;
        // Span storage must survive Build. Demand-driven providers are activated only by
        // consumers; their requests are propagated after activation, regardless of active.
        FrameCapabilityRequests requests;
    };
    class IRenderFeature
    {
    public:
        virtual ~IRenderFeature() = default;
        // Metadata only: native device initialization belongs to domain owners.
        virtual FeatureInfo Describe() const = 0;
        virtual FeatureFrameDecision Evaluate(const FeaturePrepareContext&) const = 0;
        virtual void Prepare(FeaturePrepareContext&) {}
        virtual void Build(RG::RenderGraph&, RenderFeatureContext&) = 0;
        virtual void ReleaseView(RenderViewId) {}
    };
}

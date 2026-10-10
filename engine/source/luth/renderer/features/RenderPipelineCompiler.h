#pragma once

#include "luth/renderer/features/RenderPipelineDefinition.h"

namespace Luth
{
    enum class PipelineDiagnosticCode
    {
        InvalidMetadata, InvalidContract, TypeMismatch, MissingProducer, DuplicateProducer,
        ExclusiveProducers, DuplicateInstance, UnsupportedCapability, MissingCapability,
        DuplicateCapability, IncompatibleFeatures, InvalidOrdering, OrderingCycle,
        PhaseViolation, InvalidDisabledOutput, InvalidInput, InvalidActivation,
        UndeclaredCapabilityRequest, BuildContractViolation
    };
    struct PipelineDiagnostic
    {
        PipelineDiagnosticCode code;
        FeatureInstanceId feature = InvalidFeatureInstance;
        FeatureInstanceId relatedFeature = InvalidFeatureInstance;
        ResourceKeyRef resource;
        const RenderCapabilityIdentity* capability = nullptr;
        std::vector<FeatureInstanceId> cycle;
        std::string message;
    };
    struct PipelineBuildResult
    {
        bool success = false;
        std::vector<PipelineDiagnostic> diagnostics;
    };

    class CompiledRenderPipeline;
    // One-use activation/preparation result. Bindings supplied to Prepare and Build must
    // remain frozen and scratch must survive Build. Only the latest preparation is valid;
    // this token contains no graph handles or native resource ownership.
    class PreparedPipelineFrame
    {
    public:
        PreparedPipelineFrame() = default;
        PreparedPipelineFrame(const PreparedPipelineFrame&) = delete;
        PreparedPipelineFrame& operator=(const PreparedPipelineFrame&) = delete;
    private:
        friend class CompiledRenderPipeline;
        const CompiledRenderPipeline* owner = nullptr;
        u64 frame = 0, generation = 0, serial = 0;
        RenderViewId view;
        std::span<u8> active;
        bool consumed = false;
    };
    struct PipelinePrepareResult : PipelineBuildResult
    {
        std::unique_ptr<PreparedPipelineFrame> plan;
    };

    class CompiledRenderPipeline
    {
    public:
        PipelinePrepareResult Prepare(const FrameRenderInputs&, const ViewRenderInputs&, Memory::LinearAllocator&);
        PipelineBuildResult Build(RG::RenderGraph&, const FrameRenderInputs&,
            const ViewRenderInputs&, Memory::LinearAllocator&,
            std::span<const RenderOutputBinding> outputs = {}, PreparedPipelineFrame* prepared = nullptr);
        void ReleaseView(RenderViewId);
        std::span<const FeatureInstanceId> FeatureOrder() const { return m_Order; }
        const ResourceSlotLayout& ResourceSlots() const { return m_Layout; }
        CompiledRenderPipeline(const CompiledRenderPipeline&) = delete;
        CompiledRenderPipeline& operator=(const CompiledRenderPipeline&) = delete;
    private:
        friend class RenderPipelineCompiler;
        struct CapabilityDependency
        {
            const RenderCapabilityIdentity* capability;
            size_t provider; // Index in the immutable compiled execution order, or external.
            bool conditional;
        };
        struct Entry
        {
            FeatureInstanceId id;
            FeatureInfo info;
            std::unique_ptr<IRenderFeature> feature;
            std::vector<CapabilityDependency> capabilities;
        };
        CompiledRenderPipeline(std::vector<Entry>, std::span<const ResourceKeyRef>, PipelineInputContract);
        std::vector<Entry> m_Entries;
        std::vector<FeatureInstanceId> m_Order;
        ResourceSlotLayout m_Layout;
        PipelineInputContract m_Inputs;
        std::unordered_map<const ResourceKeyIdentity*, size_t> m_InputIndices;
        std::vector<ResourceKeyRef> m_AbsentKeys;
        u64 m_PreparationSerial = 0;
    };
    struct PipelineCompileResult
    {
        std::unique_ptr<CompiledRenderPipeline> pipeline;
        std::vector<PipelineDiagnostic> diagnostics;
        // Replacement policy only. Native-resource safe points belong to the caller.
        bool ReplaceIfValid(std::unique_ptr<CompiledRenderPipeline>& current)
        {
            if (!pipeline) return false;
            current = std::move(pipeline);
            return true;
        }
    private:
        friend class RenderPipelineCompiler;
        // Failed diagnostics may refer to instance-owned resource identities.
        std::unique_ptr<RenderPipelineDefinition> m_RejectedDefinition;
    };
    class RenderPipelineCompiler
    {
    public:
        PipelineCompileResult Compile(RenderPipelineDefinition, const RendererCapabilities&,
            const PipelineInputContract& = {}) const;
    };
}

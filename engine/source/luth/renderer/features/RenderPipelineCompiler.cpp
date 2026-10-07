#include "luth/renderer/features/RenderPipelineCompiler.h"
#include "luth/memory/LinearAllocator.h"
#include "luth/renderer/rendergraph/RenderGraph.h"
#include <algorithm>
#include <functional>
#include <queue>
#include <unordered_set>

namespace Luth
{
    namespace
    {
        constexpr size_t External = std::numeric_limits<size_t>::max();
        template<class Range, class T>
        bool Contains(const Range& range, const T& value)
        {
            return std::find(range.begin(), range.end(), value) != range.end();
        }
        struct Producer { size_t feature; bool optional; bool exclusive; };
        struct Alias { ResourceKeyRef source; bool mutates; };
        struct Edge
        {
            size_t to;
            std::string reason;
            ResourceKeyRef key;
            const RenderCapabilityIdentity* capability;
        };
    }

    PipelineCompileResult RenderPipelineCompiler::Compile(RenderPipelineDefinition incoming,
        const RendererCapabilities& capabilities, const PipelineInputContract& inputs) const
    {
        PipelineCompileResult result;
        result.m_RejectedDefinition = std::make_unique<RenderPipelineDefinition>(std::move(incoming));
        auto& definition = *result.m_RejectedDefinition;
        const size_t count = definition.m_Features.size();
        std::vector<FeatureInfo> infos;
        infos.reserve(count);
        auto id = [](size_t index) { return index == External ? InvalidFeatureInstance : static_cast<u32>(index + 1); };
        auto label = [&](size_t index) {
            return index == External ? std::string("external input") : infos[index].name + " #" + std::to_string(id(index));
        };
        auto diagnose = [&](PipelineDiagnosticCode code, size_t feature, size_t related,
            ResourceKeyRef key, std::string message, const RenderCapabilityIdentity* capability = nullptr) {
            result.diagnostics.push_back({code, id(feature), id(related), key, capability, {}, std::move(message)});
        };
        for (auto& entry : definition.m_Features)
        {
            infos.push_back(entry.feature->Describe());
            auto& info = infos.back();
            if (!info.type) info.type = entry.concreteType;
            if (info.name.empty() || !info.type || info.type->name.empty() ||
                info.phase < FeaturePhase::BeforeAsync || info.phase > FeaturePhase::AfterAsync ||
                info.activation < FeatureActivation::Always || info.activation > FeatureActivation::DemandDriven)
                diagnose(PipelineDiagnosticCode::InvalidMetadata, entry.id - 1, External, {}, "Invalid feature metadata");
        }

        std::vector<ResourceKeyRef> keys;
        std::unordered_map<const ResourceKeyIdentity*, ResourceKeyRef> types;
        std::unordered_map<const ResourceKeyIdentity*, Producer> producers;
        std::unordered_map<const RenderCapabilityIdentity*, size_t> providers;
        std::unordered_map<FeatureTypeId, size_t> instances;
        std::unordered_map<const ResourceKeyIdentity*, Alias> aliases;
        std::vector<std::vector<Edge>> edges(count);

        auto registerKey = [&](ResourceKeyRef key, size_t feature) {
            if (!key.identity || !key.type || key.identity->name.empty() || !key.type->size ||
                !key.type->alignment || key.type->alignment > alignof(std::max_align_t) ||
                (key.type->alignment & (key.type->alignment - 1)))
            {
                diagnose(PipelineDiagnosticCode::InvalidContract, feature, External, key, label(feature) + ": invalid resource key");
                return false;
            }
            const auto [it, inserted] = types.emplace(key.identity, key);
            if (!inserted && it->second.type != key.type)
            {
                diagnose(PipelineDiagnosticCode::TypeMismatch, feature, External, key,
                    label(feature) + ": " + std::string(key.identity->name) + " expected " + std::string(key.type->name) +
                    ", declared as " + std::string(it->second.type->name));
                return false;
            }
            if (inserted) keys.push_back(key);
            return true;
        };
        auto registerWrite = [&](const ResourceWrite& write, size_t feature) {
            if (!registerKey(write.key, feature)) return;
            const auto [it, inserted] = producers.emplace(write.key.identity,
                Producer{feature, write.presence == ResourceOutputPresence::Optional, write.exclusive});
            if (!inserted)
                diagnose(write.exclusive || it->second.exclusive ? PipelineDiagnosticCode::ExclusiveProducers :
                    PipelineDiagnosticCode::DuplicateProducer, feature, it->second.feature, write.key,
                    std::string(write.key.identity->name) + ": conflicting producers " + label(it->second.feature) + " and " + label(feature));
        };
        auto registerCapability = [&](const RenderCapabilityIdentity* key, size_t feature) {
            if (!key || key->name.empty())
                diagnose(PipelineDiagnosticCode::InvalidContract, feature, External, {}, label(feature) + ": invalid capability");
            else
            {
                const auto [it, inserted] = providers.emplace(key, feature);
                if (!inserted) diagnose(PipelineDiagnosticCode::DuplicateCapability, feature, it->second, {},
                    std::string(key->name) + ": conflicting capability providers", key);
            }
        };
        for (const auto& input : inputs.resources)
        {
            registerWrite(input, External);
            if (input.aliasesInput || input.mutatesInput || input.disabledPassthrough)
                diagnose(PipelineDiagnosticCode::InvalidContract, External, External, input.key, "Imported resources cannot declare alias transforms");
        }
        for (const auto* capability : inputs.capabilities) registerCapability(capability, External);

        for (size_t i = 0; i < count; ++i)
        {
            const auto& info = infos[i];
            const auto [instance, inserted] = instances.emplace(info.type, i);
            if (!inserted && (!info.allowMultipleInstances || !infos[instance->second].allowMultipleInstances))
                diagnose(PipelineDiagnosticCode::DuplicateInstance, i, instance->second, {}, label(i) + ": duplicate feature type");
            for (const auto* device : info.capabilities.deviceRequirements)
                if (!device || device->name.empty())
                    diagnose(PipelineDiagnosticCode::InvalidContract, i, External, {}, label(i) + ": invalid device capability");
                else if (!Contains(capabilities.supported, device))
                    diagnose(PipelineDiagnosticCode::UnsupportedCapability, i, External, {}, label(i) + ": unsupported device capability " + std::string(device->name));
            std::unordered_set<const ResourceKeyIdentity*> reads;
            for (const auto& read : info.resources.reads)
            {
                registerKey(read.key, i);
                if (!reads.insert(read.key.identity).second)
                    diagnose(PipelineDiagnosticCode::InvalidContract, i, External, read.key, label(i) + ": duplicate read declaration");
            }
            for (const auto& write : info.resources.writes)
            {
                registerWrite(write, i);
                if (reads.contains(write.key.identity))
                    diagnose(PipelineDiagnosticCode::InvalidContract, i, External, write.key, label(i) + ": stages must have distinct input/output keys");
                if (write.mutatesInput && !write.aliasesInput)
                    diagnose(PipelineDiagnosticCode::InvalidContract, i, External, write.key, label(i) + ": mutation requires an alias source");
                if (info.activation != FeatureActivation::Always && write.presence == ResourceOutputPresence::Required &&
                    !write.aliasesInput && !write.disabledPassthrough)
                    diagnose(PipelineDiagnosticCode::InvalidDisabledOutput, i, External, write.key, label(i) + ": disabled required output needs a pass-through source");
                const auto possibleAlias = write.aliasesInput ? write.aliasesInput : write.disabledPassthrough;
                if (possibleAlias)
                    aliases.emplace(write.key.identity, Alias{*possibleAlias, write.mutatesInput});
                if (write.aliasesInput && write.disabledPassthrough &&
                    write.aliasesInput->identity != write.disabledPassthrough->identity)
                    diagnose(PipelineDiagnosticCode::InvalidContract, i, External, write.key, label(i) + ": active alias and disabled pass-through must use the same source");
                for (const auto sourceKey : {write.aliasesInput, write.disabledPassthrough})
                {
                    if (!sourceKey) continue;
                    registerKey(*sourceKey, i);
                    const auto source = std::find_if(info.resources.reads.begin(), info.resources.reads.end(), [&](const ResourceRead& read) {
                        return read.key.identity == sourceKey->identity && read.key.type == sourceKey->type;
                    });
                    if (source == info.resources.reads.end() || write.key.type != sourceKey->type ||
                        (write.presence == ResourceOutputPresence::Required && source->requirement != ResourceReadRequirement::Required))
                        diagnose(PipelineDiagnosticCode::InvalidContract, i, External, write.key, label(i) + ": alias requires a compatible declared read (required for a required output)");
                }
            }
            for (const auto* capability : info.capabilities.provides) registerCapability(capability, i);
        }

        auto addEdge = [&](size_t from, size_t to, std::string reason, ResourceKeyRef key = {}, const RenderCapabilityIdentity* capability = nullptr) {
            if (from == External) return;
            if (infos[from].phase > infos[to].phase)
                diagnose(PipelineDiagnosticCode::PhaseViolation, to, from, key,
                    label(from) + " -> " + label(to) + ": dependency reverses submission phases (" + reason + ")", capability);
            auto& list = edges[from];
            if (std::none_of(list.begin(), list.end(), [&](const Edge& edge) { return edge.to == to; }))
                list.push_back({to, std::move(reason), key, capability});
        };
        auto explicitEdge = [&](FeatureInstanceId first, FeatureInstanceId second, size_t owner) {
            if (!first || !second || first > count || second > count)
                diagnose(PipelineDiagnosticCode::InvalidOrdering, owner, External, {}, "Ordering references an unknown feature instance");
            else addEdge(first - 1, second - 1, "explicit order");
        };
        for (const auto& [first, second] : definition.m_Ordering) explicitEdge(first, second, External);
        std::vector<ResourceKeyRef> absentKeys;
        for (size_t i = 0; i < count; ++i)
        {
            const auto& info = infos[i];
            for (auto other : info.ordering.before) explicitEdge(id(i), other, i);
            for (auto other : info.ordering.after) explicitEdge(other, id(i), i);
            for (auto incompatible : info.ordering.incompatibleTypes)
                if (const auto it = instances.find(incompatible); it != instances.end())
                    diagnose(PipelineDiagnosticCode::IncompatibleFeatures, i, it->second, {}, label(i) + " is incompatible with " + label(it->second));
            for (const auto& read : info.resources.reads)
            {
                if (!read.key.identity || !read.key.type) continue;
                const auto provider = producers.find(read.key.identity);
                if (provider == producers.end())
                {
                    if (read.requirement == ResourceReadRequirement::Required)
                        diagnose(PipelineDiagnosticCode::MissingProducer, i, External, read.key,
                            label(i) + ": required resource " + std::string(read.key.identity->name) + " has no producer");
                    else if (std::none_of(absentKeys.begin(), absentKeys.end(), [&](ResourceKeyRef key) { return key.identity == read.key.identity; }))
                        absentKeys.push_back(read.key);
                }
                else
                {
                    if (provider->second.optional && read.requirement == ResourceReadRequirement::Required)
                        diagnose(PipelineDiagnosticCode::MissingProducer, i, provider->second.feature, read.key,
                            label(i) + ": required resource " + std::string(read.key.identity->name) + " has only an optional producer " + label(provider->second.feature));
                    addEdge(provider->second.feature, i, std::string(read.key.identity->name), read.key);
                }
            }
            std::unordered_set<const RenderCapabilityIdentity*> consumes;
            for (const auto& read : info.capabilities.consumes)
            {
                if (!read.capability || read.capability->name.empty() || !consumes.insert(read.capability).second)
                {
                    diagnose(PipelineDiagnosticCode::InvalidContract, i, External, {}, label(i) + ": invalid or duplicate capability read");
                    continue;
                }
                const auto provider = providers.find(read.capability);
                if (provider == providers.end())
                    diagnose(PipelineDiagnosticCode::MissingCapability, i, External, {}, label(i) + ": requires " + std::string(read.capability->name) + " without a provider", read.capability);
                else addEdge(provider->second, i, std::string(read.capability->name), {}, read.capability);
            }
        }
        // Stop malformed contracts before traversing their alias pointers or sorting edges.
        if (!result.diagnostics.empty()) return result;

        for (size_t mutation = 0; mutation < count; ++mutation)
            for (const auto& write : infos[mutation].resources.writes)
            {
                if (!write.mutatesInput) continue;
                std::unordered_set<const ResourceKeyIdentity*> ancestors;
                auto key = write.aliasesInput->identity;
                while (ancestors.insert(key).second)
                {
                    const auto alias = aliases.find(key);
                    if (alias == aliases.end()) break;
                    key = alias->second.source.identity;
                }
                for (size_t reader = 0; reader < count; ++reader)
                {
                    if (reader == mutation) continue;
                    for (const auto& read : infos[reader].resources.reads)
                    {
                        // Follow non-mutating alias branches too. A later mutation denotes
                        // a newer stage, whose consumers must not move before this one.
                        std::unordered_set<const ResourceKeyIdentity*> visited;
                        auto candidate = read.key.identity;
                        while (visited.insert(candidate).second)
                        {
                            if (ancestors.contains(candidate))
                            {
                                addEdge(reader, mutation, std::string(read.key.identity->name) + " alias mutation", read.key);
                                break;
                            }
                            const auto alias = aliases.find(candidate);
                            if (alias == aliases.end() || alias->second.mutates) break;
                            candidate = alias->second.source.identity;
                        }
                    }
                }
            }
        if (!result.diagnostics.empty()) return result;
        // Phase edges constrain independent features as well as explicit dependencies.
        for (size_t i = 0; i < count; ++i)
            for (size_t j = 0; j < count; ++j)
                if (infos[i].phase < infos[j].phase) addEdge(i, j, "submission phase");
        std::vector<size_t> indegree(count, 0), order;
        for (const auto& list : edges) for (const auto& edge : list) ++indegree[edge.to];
        std::priority_queue<size_t, std::vector<size_t>, std::greater<>> ready;
        for (size_t i = 0; i < count; ++i) if (!indegree[i]) ready.push(i);
        while (!ready.empty())
        {
            const auto current = ready.top(); ready.pop(); order.push_back(current);
            for (const auto& edge : edges[current]) if (!--indegree[edge.to]) ready.push(edge.to);
        }
        if (order.size() != count)
        {
            std::vector<u8> color(count, 0);
            std::vector<size_t> stack;
            std::function<bool(size_t)> visit = [&](size_t current) {
                color[current] = 1; stack.push_back(current);
                for (const auto& edge : edges[current])
                {
                    if (!color[edge.to] && visit(edge.to)) return true;
                    if (color[edge.to] == 1)
                    {
                        PipelineDiagnostic diagnostic{PipelineDiagnosticCode::OrderingCycle};
                        diagnostic.feature = id(edge.to);
                        diagnostic.relatedFeature = id(current);
                        diagnostic.resource = edge.key;
                        diagnostic.capability = edge.capability;
                        const auto start = std::find(stack.begin(), stack.end(), edge.to);
                        for (auto it = start; it != stack.end(); ++it)
                        {
                            diagnostic.cycle.push_back(id(*it));
                            diagnostic.message += label(*it) + " -> ";
                        }
                        diagnostic.cycle.push_back(id(edge.to));
                        diagnostic.message += label(edge.to) + ": ordering cycle (";
                        for (auto it = start; it != stack.end(); ++it)
                        {
                            const auto next = it + 1 == stack.end() ? edge.to : *(it + 1);
                            const auto reason = std::find_if(edges[*it].begin(), edges[*it].end(), [&](const Edge& e) { return e.to == next; });
                            if (it != start) diagnostic.message += ", ";
                            diagnostic.message += reason->reason;
                            if (!diagnostic.resource.identity && reason->key.identity) diagnostic.resource = reason->key;
                            if (!diagnostic.capability) diagnostic.capability = reason->capability;
                        }
                        diagnostic.message += ")";
                        result.diagnostics.push_back(std::move(diagnostic));
                        return true;
                    }
                }
                stack.pop_back(); color[current] = 2; return false;
            };
            for (size_t i = 0; i < count; ++i) if (!color[i] && visit(i)) break;
            return result;
        }

        std::vector<size_t> positions(count);
        for (size_t i = 0; i < count; ++i) positions[order[i]] = i;
        std::vector<CompiledRenderPipeline::Entry> entries;
        entries.reserve(count);
        for (auto original : order)
        {
            std::vector<CompiledRenderPipeline::CapabilityDependency> dependencies;
            for (const auto& read : infos[original].capabilities.consumes)
            {
                const auto provider = providers.at(read.capability);
                dependencies.push_back({read.capability, provider == External ? External : positions[provider], read.conditional});
            }
            entries.push_back({id(original), std::move(infos[original]),
                std::move(definition.m_Features[original].feature), std::move(dependencies)});
        }
        try
        {
            result.pipeline.reset(new CompiledRenderPipeline(std::move(entries), keys, inputs));
            result.pipeline->m_AbsentKeys = std::move(absentKeys);
            result.m_RejectedDefinition.reset();
        }
        catch (const std::exception& error)
        {
            diagnose(PipelineDiagnosticCode::InvalidContract, External, External, {}, error.what());
        }
        return result;
    }

    CompiledRenderPipeline::CompiledRenderPipeline(std::vector<Entry> entries,
        std::span<const ResourceKeyRef> keys, PipelineInputContract inputs)
        : m_Entries(std::move(entries)), m_Layout(keys), m_Inputs(std::move(inputs))
    {
        for (const auto& entry : m_Entries) m_Order.push_back(entry.id);
        for (size_t i = 0; i < m_Inputs.resources.size(); ++i)
            m_InputIndices.emplace(m_Inputs.resources[i].key.identity, i);
    }

    PipelineBuildResult CompiledRenderPipeline::Build(RG::RenderGraph& graph,
        const FrameRenderInputs& frame, const ViewRenderInputs& view, Memory::LinearAllocator& scratch,
        std::span<const RenderOutputBinding> outputs)
    {
        PipelineBuildResult result;
        for (const auto& output : outputs) output.m_Copy(output.m_Destination, nullptr);
        FeatureInstanceId current = InvalidFeatureInstance;
        auto fail = [&](PipelineDiagnosticCode code, std::string message, ResourceKeyRef key = {}, const RenderCapabilityIdentity* capability = nullptr) {
            result.diagnostics.push_back({code, current, InvalidFeatureInstance, key, capability, {}, std::move(message)});
        };
        try
        {
            RenderBlackboard board(m_Layout, scratch);
            for (const auto& output : outputs) (void)m_Layout.Find(output.m_Key);
            for (const auto bindings : {frame.resources, view.resources})
                for (const auto& binding : bindings)
                {
                    const auto input = m_InputIndices.find(binding.m_Key.identity);
                    if (input == m_InputIndices.end() || m_Inputs.resources[input->second].key.type != binding.m_Key.type)
                    {
                        fail(PipelineDiagnosticCode::InvalidInput, "Undeclared or wrongly typed external publication", binding.m_Key);
                        return result;
                    }
                    if (binding.m_Value) board.PublishBorrowed(binding.m_Key, binding.m_Value);
                    else board.PublishAbsentBorrowed(binding.m_Key);
                }
            for (const auto& input : m_Inputs.resources)
            {
                const size_t slot = m_Layout.Find(input.key);
                if (board.m_States[slot] == RenderBlackboard::State::Unpublished && input.presence == ResourceOutputPresence::Optional)
                    board.PublishAbsentBorrowed(input.key);
                if (board.m_States[slot] != RenderBlackboard::State::Present && input.presence == ResourceOutputPresence::Required)
                {
                    fail(PipelineDiagnosticCode::InvalidInput, "Required external input is absent: " + std::string(input.key.identity->name), input.key);
                    return result;
                }
            }
            for (auto key : m_AbsentKeys) board.PublishAbsentBorrowed(key);
            const auto count = m_Entries.size();
            auto* decisions = count ? static_cast<FeatureFrameDecision*>(scratch.Allocate(sizeof(FeatureFrameDecision) * count, alignof(FeatureFrameDecision))) : nullptr;
            auto* active = count ? static_cast<u8*>(scratch.Allocate(count, alignof(u8))) : nullptr;
            FeaturePrepareContext prepare{frame, view, scratch};
            for (size_t i = 0; i < count; ++i)
            {
                const auto& entry = m_Entries[i]; current = entry.id;
                new (decisions + i) FeatureFrameDecision(entry.feature->Evaluate(prepare));
                active[i] = entry.info.activation == FeatureActivation::DemandDriven ? 0 : static_cast<u8>(decisions[i].active);
                if (entry.info.activation == FeatureActivation::Always && !active[i])
                {
                    fail(PipelineDiagnosticCode::InvalidActivation, entry.info.name + ": an always-active feature cannot disable itself");
                    return result;
                }
                for (const auto* request : decisions[i].requests)
                    if (std::none_of(entry.capabilities.begin(), entry.capabilities.end(), [&](const auto& read) { return read.capability == request; }))
                    {
                        fail(PipelineDiagnosticCode::UndeclaredCapabilityRequest, entry.info.name + ": undeclared capability request", {}, request);
                        return result;
                    }
            }
            // Reverse compiled order propagates transitive demand once. Static dependency
            // discovery, provider resolution and topology sorting never run here.
            for (size_t i = count; i-- > 0;)
            {
                if (!active[i]) continue;
                const auto& entry = m_Entries[i]; current = entry.id;
                for (const auto& read : entry.capabilities)
                {
                    if (read.conditional && !Contains(decisions[i].requests, read.capability)) continue;
                    if (read.provider == External)
                    {
                        if (!Contains(frame.capabilities, read.capability) && !Contains(view.capabilities, read.capability))
                        {
                            fail(PipelineDiagnosticCode::MissingCapability, entry.info.name + ": external capability unavailable: " + std::string(read.capability->name), {}, read.capability);
                            return result;
                        }
                    }
                    else if (m_Entries[read.provider].info.activation == FeatureActivation::DemandDriven)
                        active[read.provider] = 1;
                    else if (!active[read.provider])
                    {
                        fail(PipelineDiagnosticCode::MissingCapability, entry.info.name + ": selected capability provider is inactive", {}, read.capability);
                        return result;
                    }
                }
            }
            for (size_t i = 0; i < count; ++i)
                if (active[i]) { current = m_Entries[i].id; m_Entries[i].feature->Prepare(prepare); }
            bool seenAsync = false, seenGraphicsB = false;
            for (const auto& pass : graph.GetPasses())
                if (pass.queueFamily == RG::QueueFamily::AsyncCompute) seenAsync = true;
                else if (seenAsync) seenGraphicsB = true;
            RenderFeatureContext context{frame, view, board, scratch};
            for (size_t i = 0; i < count; ++i)
            {
                auto& entry = m_Entries[i]; current = entry.id;
                const size_t firstPass = graph.GetPasses().size();
                RenderBlackboard::FeatureScope scope(board, entry.info.resources, entry.info.name);
                if (active[i]) entry.feature->Build(graph, context);
                else
                    for (const auto& write : entry.info.resources.writes)
                    {
                        const auto source = write.disabledPassthrough ? write.disabledPassthrough : write.aliasesInput;
                        const void* value = source ? board.TryGetBorrowed(*source) : nullptr;
                        if (value) board.PublishBorrowed(write.key, value);
                        else board.PublishAbsentBorrowed(write.key);
                    }
                scope.ValidateOutputs();
                for (size_t p = firstPass; p < graph.GetPasses().size(); ++p)
                {
                    const auto& pass = graph.GetPasses()[p];
                    const bool async = pass.queueFamily == RG::QueueFamily::AsyncCompute;
                    if ((async != (entry.info.phase == FeaturePhase::Async)) ||
                        (async && seenGraphicsB) || (!async && seenAsync && entry.info.phase == FeaturePhase::BeforeAsync))
                    {
                        fail(PipelineDiagnosticCode::PhaseViolation, entry.info.name + ": pass " + pass.name + " contradicts its submission phase");
                        return result;
                    }
                    if (async) seenAsync = true;
                    else if (seenAsync) seenGraphicsB = true;
                }
            }
            for (const auto& output : outputs)
                output.m_Copy(output.m_Destination, board.TryGetBorrowed(output.m_Key));
            result.success = true;
        }
        catch (const BlackboardError& error)
        {
            fail(current ? PipelineDiagnosticCode::BuildContractViolation : PipelineDiagnosticCode::InvalidInput, error.what(), error.Key());
        }
        catch (const std::exception& error)
        {
            fail(PipelineDiagnosticCode::BuildContractViolation, error.what());
        }
        // A failed build leaves a partial graph. The caller must discard it, never record it.
        return result;
    }

    void CompiledRenderPipeline::ReleaseView(RenderViewId view)
    {
        for (auto& entry : m_Entries) entry.feature->ReleaseView(view);
    }
}

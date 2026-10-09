#pragma once
#include "luth/renderer/features/RenderFeature.h"
#include "luth/renderer/subsystems/RtSubsystem.h"
#include "luth/core/RenderSnapshot.h"
#include <functional>

namespace Luth
{
    enum class RtSceneConsumer : size_t { SunShadow, Fog, DirectLighting, GlobalIllumination, Reflections, PathTrace, Count };
    struct RtSceneParameters
    {
        std::array<bool, static_cast<size_t>(RtSceneConsumer::Count)> active{};
        const std::unordered_map<UUID, u32, UUIDHash>* materialSlots = nullptr;
        bool markEmitters = false;
    };
    struct RaySceneRef { const PreparedRtScene* native = nullptr; };
    namespace RtSceneResources
    {
        inline constexpr RenderCapabilityIdentity RayScene{"RayScene"};
        inline constexpr DeviceCapabilityIdentity AccelerationStructures{"AccelerationStructures"};
        inline constexpr DeviceCapabilityIdentity RayQueries{"RayQueries"};
        inline constexpr ResourceKeyIdentity ParametersIdentity{"RtScene.Parameters"};
        inline constexpr RenderResourceKey<RtSceneParameters> Parameters{&ParametersIdentity};
        inline constexpr ResourceKeyIdentity SceneIdentity{"RtScene.PreparedBindings"};
        inline constexpr RenderResourceKey<RaySceneRef> Scene{&SceneIdentity};
        inline constexpr std::array Requests{&RayScene};
    }
    inline const RtSceneParameters& GetRtSceneParameters(const FeaturePrepareContext& ctx)
    {
        for (const auto bindings : {ctx.view.resources, ctx.frame.resources})
            for (const auto& binding : bindings)
                if (const auto* params = binding.TryGet(RtSceneResources::Parameters)) return *params;
        throw std::invalid_argument("RT scene: missing frozen parameters");
    }
    // Temporary adapters declare each unmigrated native consumer separately. M14–M17
    // replace them with their actual pass adapters without changing provider policy.
    class RtSceneDemandFeature final : public IRenderFeature
    {
    public:
        explicit RtSceneDemandFeature(RtSceneConsumer consumer) : m_Consumer(consumer) {}
        FeatureInfo Describe() const override
        {
            static constexpr std::array names{"RtSunShadowDemand", "RtFogDemand", "RestirDiDemand",
                "RestirGiDemand", "ReflectionsDemand", "PathTraceDemand"};
            FeatureInfo info;
            info.name = names.at(static_cast<size_t>(m_Consumer));
            info.allowMultipleInstances = true;
            info.phase = FeaturePhase::Async;
            info.activation = FeatureActivation::Conditional;
            info.resources.reads = {{RtSceneResources::Parameters},
                {RtSceneResources::Scene, ResourceReadRequirement::Optional}};
            info.capabilities.consumes = {{&RtSceneResources::RayScene}};
            info.capabilities.deviceRequirements = {&RtSceneResources::RayQueries};
            return info;
        }
        FeatureFrameDecision Evaluate(const FeaturePrepareContext& ctx) const override
        { return {GetRtSceneParameters(ctx).active.at(static_cast<size_t>(m_Consumer))}; }
        void Build(RG::RenderGraph&, RenderFeatureContext& ctx) override
        {
            const auto* scene = ctx.resources.TryGet(RtSceneResources::Scene);
            if (!scene || !scene->native || scene->native->frameIndex != ctx.frame.renderFrameIndex)
                throw std::logic_error("RT consumer: prepared scene is missing or stale");
        }
    private:
        RtSceneConsumer m_Consumer;
    };
    class RtSceneFeature final : public IRenderFeature
    {
    public:
        using PrepareFn = std::function<std::shared_ptr<const PreparedRtScene>(const FrameRenderInputs&, const RtSceneParameters&)>;
        using RegisterFn = std::function<void(RG::RenderGraph&, std::shared_ptr<const PreparedRtScene>)>;
        RtSceneFeature(PrepareFn prepare, RegisterFn record) : m_Prepare(std::move(prepare)), m_Register(std::move(record)) {}
        explicit RtSceneFeature(RtSubsystem& native) : RtSceneFeature(
            [&native](const FrameRenderInputs& frame, const RtSceneParameters& params) {
                if (!frame.snapshot || !params.materialSlots) throw std::invalid_argument("RT scene: missing native inputs");
                return native.PrepareScene(frame.snapshot->meshes, frame.renderFrameIndex, *params.materialSlots, params.markEmitters);
            },
            [&native](RG::RenderGraph& graph, std::shared_ptr<const PreparedRtScene> scene) {
                native.AddTlasBuildPass(graph, std::move(scene));
            }) {}
        FeatureInfo Describe() const override
        {
            FeatureInfo info;
            info.name = "RtScene";
            info.phase = FeaturePhase::Async;
            info.activation = FeatureActivation::DemandDriven;
            info.resources.reads = {{RtSceneResources::Parameters}};
            info.resources.writes = {{RtSceneResources::Scene, ResourceOutputPresence::Optional}};
            info.capabilities.provides = {&RtSceneResources::RayScene};
            info.capabilities.deviceRequirements = {&RtSceneResources::AccelerationStructures};
            return info;
        }
        FeatureFrameDecision Evaluate(const FeaturePrepareContext&) const override { return {false}; }
        void Prepare(FeaturePrepareContext& ctx) override
        {
            m_Scene = m_Prepare(ctx.frame, GetRtSceneParameters(ctx));
            if (!m_Scene || m_Scene->frameIndex != ctx.frame.renderFrameIndex)
                throw std::logic_error("RT scene provider returned an invalid frame packet");
        }
        void Build(RG::RenderGraph& graph, RenderFeatureContext& ctx) override
        {
            if (!m_Scene || m_Scene->frameIndex != ctx.frame.renderFrameIndex)
                throw std::logic_error("RT scene provider was not prepared for this frame");
            m_Register(graph, m_Scene);
            ctx.resources.Publish(RtSceneResources::Scene, RaySceneRef{m_Scene.get()});
        }
    private:
        PrepareFn m_Prepare;
        RegisterFn m_Register;
        std::shared_ptr<const PreparedRtScene> m_Scene;
    };
}

#include "luthpch.h"
#include "luth/renderer/debug/NativeDebugOutputs.h"
#include "luth/renderer/RenderPipeline.h"
#include "luth/renderer/FrameTargets.h"

namespace Luth
{
    DebugOutputs CollectSharedDebugOutputs(const LightingSubsystem& lighting)
    {
        return {{"ShadowMap", lighting.GetShadowMap()}, {"IrradianceMap", lighting.GetIrradianceMap()},
            {"PrefilteredMap", lighting.GetPrefilteredMap()}, {"BRDF_LUT", lighting.GetBRDFLut()}};
    }

    DebugOutputs CollectViewDebugOutputs(const FrameTargets& targets, const ViewResources* state, const GtaoViewState* ao)
    {
        DebugOutputs outputs{{"SceneColor", targets.GetSceneColor()}, {"SceneDepth", targets.GetSceneDepth()},
            {"LDROutput", targets.GetLDROutput()}, {"EntityID", targets.GetEntityIDBuffer()},
            {"SlimNormal", targets.GetSlimNormal()}, {"SlimRoughness", targets.GetSlimRoughness()},
            {"SlimMotion", targets.GetSlimMotion()}, {"SlimMaterialID", targets.GetSlimMaterialID()}};
        if (state)
        {
            if (state->bloom)
                for (u32 i = 0; i < BloomViewState::kMipCount; ++i)
                    outputs.push_back({"BloomMip" + std::to_string(i), state->bloom->mips[i]});
            if (state->fog)
            {
                outputs.push_back({"VolDensity", state->fog->volDensity});
                outputs.push_back({"VolInScatter", state->fog->volInScatter});
                outputs.push_back({"VolInScatterHistA", state->fog->volInScatterHistA});
                outputs.push_back({"VolInScatterHistB", state->fog->volInScatterHistB});
            }
            outputs.push_back({"Reflections", state->reflRadiance});
        }
        if (ao)
        {
            outputs.push_back({"GTAOLinearDepth", ao->linearDepth});
            outputs.push_back({"GTAORawAO", ao->rawAO});
            outputs.push_back({"GTAOFinal", ao->finalAO});
        }
        return outputs;
    }
}

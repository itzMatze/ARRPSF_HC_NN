#include "ComputePathTracer.h"
#include "RenderGraph/RenderPassHelpers.h"
#include "RenderGraph/RenderPassStandardFlags.h"

namespace
{
const std::string kShaderFile("RenderPasses/ComputePathTracer/ComputePathTracer.slang");

const char kInputViewDir[] = "viewW";

const ChannelList kInputChannels = {
    // clang-format off
    { "vbuffer",        "gVBuffer",     "Visibility buffer in packed format" },
    { kInputViewDir,    "gViewW",       "World-space view direction (xyz float format)", true /* optional */ },
    // clang-format on
};

const ChannelList kOutputChannels = {
    // clang-format off
    { "color",          "gOutputColor", "Output color (sum of direct and indirect)", false, ResourceFormat::RGBA32Float },
    // clang-format on
};

const char kMaxBounces[] = "maxBounces";
const char kComputeDirect[] = "computeDirect";
const char kUseImportanceSampling[] = "useImportanceSampling";
const char kRRProbStartValue[] = "RRProbStartValue";
const char kRRProbReductionFactor[] = "RRProbReductionFactor";
} // namespace

extern "C" FALCOR_API_EXPORT void registerPlugin(Falcor::PluginRegistry& registry)
{
    registry.registerClass<RenderPass, ComputePathTracer>();
}

ComputePathTracer::ComputePathTracer(ref<Device> pDevice, const Properties& props) : RenderPass(pDevice)
{
    mpSampleGenerator = SampleGenerator::create(mpDevice, SAMPLE_GENERATOR_UNIFORM);
    for (const auto& [key, value] : props)
    {
        if (key == kMaxBounces) mMaxBounces = value;
        else if (key == kComputeDirect) mComputeDirect = value;
        else if (key == kUseImportanceSampling) mUseImportanceSampling = value;
        else logWarning("Unknown property '{}' in ComputePathTracer properties.", key);
    }
}

Properties ComputePathTracer::getProperties() const
{
    Properties props;
    props[kMaxBounces] = mMaxBounces;
    props[kComputeDirect] = mComputeDirect;
    props[kUseImportanceSampling] = mUseImportanceSampling;
    props[kRRProbStartValue] = mRRProbStartValue;
    props[kRRProbReductionFactor] = mRRProbReductionFactor;
    return props;
}

RenderPassReflection ComputePathTracer::reflect(const CompileData& compileData)
{
    RenderPassReflection reflector;

    // Define our input/output channels.
    addRenderPassInputs(reflector, kInputChannels);
    addRenderPassOutputs(reflector, kOutputChannels);

    return reflector;
}

void ComputePathTracer::compile(RenderContext* pRenderContext, const CompileData& compileData) {}

void ComputePathTracer::execute(RenderContext* pRenderContext, const RenderData& renderData)
{
    const auto& pOutput = renderData.getTexture("color");
    uint2 frameDim = {pOutput->getWidth(), pOutput->getHeight()};

    // Update refresh flag if options that affect the output have changed.
    auto& dict = renderData.getDictionary();
    if (mOptionsChanged)
    {
        auto flags = dict.getValue(kRenderPassRefreshFlags, RenderPassRefreshFlags::None);
        dict[Falcor::kRenderPassRefreshFlags] = flags | Falcor::RenderPassRefreshFlags::RenderOptionsChanged;
    }

    // If we have no scene, just clear the outputs and return.
    if (!mpScene)
    {
        for (auto it : kOutputChannels)
        {
            Texture* pDst = renderData.getTexture(it.name).get();
            if (pDst) pRenderContext->clearTexture(pDst);
        }
        return;
    }

    if (is_set(mpScene->getUpdates(), Scene::UpdateFlags::RecompileNeeded) ||
        is_set(mpScene->getUpdates(), Scene::UpdateFlags::GeometryChanged))
    {
        FALCOR_THROW("This render pass does not support scene changes that require shader recompilation.");
    }

    // Request the light collection if emissive lights are enabled.
    if (mpScene->getRenderSettings().useEmissiveLights)
    {
        mpScene->getLightCollection(pRenderContext);
    }

    DefineList defineList = getValidResourceDefines(kInputChannels, renderData);
    defineList.add(getValidResourceDefines(kOutputChannels, renderData));
    defineList.add(mpScene->getSceneDefines());
    defineList.add(mpSampleGenerator->getDefines());
    defineList["MAX_BOUNCES"] = std::to_string(mMaxBounces);
    defineList["RR_PROB_START_VALUE"] = fmt::format("{:.4f}", mRRProbStartValue);
    defineList["RR_PROB_REDUCTION_FACTOR"] = fmt::format("{:.4f}", mRRProbReductionFactor);
    defineList["DEBUG_PATH_LENGTH"] = mShowPathLength ? "1" : "0";
    defineList["PATH_LENGTH_UPPER_LIMIT"] = std::to_string(mPathLengthUpperLimit);
    defineList["COMPUTE_DIRECT"] = mComputeDirect ? "1" : "0";
    defineList["USE_IMPORTANCE_SAMPLING"] = mUseImportanceSampling ? "1" : "0";
    defineList["USE_ANALYTIC_LIGHTS"] = mpScene->useAnalyticLights() ? "1" : "0";
    defineList["USE_EMISSIVE_LIGHTS"] = mpScene->useEmissiveLights() ? "1" : "0";
    defineList["USE_ENV_LIGHT"] = mpScene->useEnvLight() ? "1" : "0";
    defineList["USE_ENV_BACKGROUND"] = mpScene->useEnvBackground() ? "1" : "0";

    if (!mpPass)
    {
        ProgramDesc desc;
        desc.addShaderModules(mpScene->getShaderModules());
        desc.addShaderLibrary(kShaderFile).csEntry("main");
        desc.addTypeConformances(mpScene->getTypeConformances());

        mpPass = ComputePass::create(mpDevice, desc, defineList, true);
        mpVars = ProgramVars::create(mpDevice, mpPass->getProgram()->getReflector());
    }

    if (mOptionsChanged)
    {
        mpPass->getProgram()->addDefines(defineList);
        // Set constants.
        mpVars = ProgramVars::create(mpDevice, mpPass->getProgram()->getReflector());
        mOptionsChanged = false;
    }

    auto var = mpVars->getRootVar();
    var["CB"]["gFrameDim"] = frameDim;
    var["CB"]["gFrameCount"] = mFrameCount;
    mpScene->bindShaderData(var["gScene"]);
    mpSampleGenerator->bindShaderData(var);

    // Bind I/O buffers. These needs to be done per-frame as the buffers may change anytime.
    auto bind = [&](const ChannelDesc& desc)
    {
        if (!desc.texname.empty())
        {
            var[desc.texname] = renderData.getTexture(desc.name);
        }
    };
    for (auto channel : kInputChannels) bind(channel);
    for (auto channel : kOutputChannels) bind(channel);

    mpPass->setVars(mpVars);
    // Get dimensions of ray dispatch.
    const uint2 targetDim = renderData.getDefaultTextureDims();
    FALCOR_ASSERT(targetDim.x > 0 && targetDim.y > 0);

    ref<ComputeState> cs = ComputeState::create(mpDevice);
    // Spawn the rays.
    mpPass->execute(pRenderContext, frameDim.x, frameDim.y);

    mFrameCount++;
}

void ComputePathTracer::renderUI(Gui::Widgets& widget)
{
    widget.var("Max bounces", mMaxBounces, 0u, 1u << 16);
    widget.tooltip("Maximum path length for indirect illumination.\n0 = direct only\n1 = one indirect bounce etc.", true);

    widget.checkbox("Evaluate direct illumination", mComputeDirect);
    widget.tooltip("Compute direct illumination.\nIf disabled only indirect is computed (when max bounces > 0).", true);

    widget.checkbox("Use importance sampling", mUseImportanceSampling);
    widget.tooltip("Use importance sampling for materials", true);

    widget.var("RR start value", mRRProbStartValue, 1.0f, 4.0f);
    widget.tooltip("Starting value of the survival probability", true);
    widget.var("RR reduction factor", mRRProbReductionFactor, 0.1f, 0.99f);
    widget.tooltip("Gets multiplied to the initial survival probability at each interaction", true);
    Gui::Group debug_group = widget.group("Debug");
    debug_group.checkbox("Path length", mShowPathLength);
    debug_group.var("Upper limit", mPathLengthUpperLimit, 1u, 1000u, 1, true);

    // If new rendering options that modify the output are applied, set flag to indicate that.
    // In execute() we will pass the flag to other passes for reset of temporal data etc.
    mOptionsChanged |= widget.button("Apply");
}

void ComputePathTracer::setScene(RenderContext* pRenderContext, const ref<Scene>& pScene)
{
    mpScene = pScene;
}


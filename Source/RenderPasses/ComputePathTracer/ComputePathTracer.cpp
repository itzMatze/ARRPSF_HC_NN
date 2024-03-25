#include "ComputePathTracer.h"
#include "RenderGraph/RenderPassHelpers.h"
#include "RenderGraph/RenderPassStandardFlags.h"
#include "imgui.h"

namespace
{
constexpr bool kUseImGui = true;
const std::string kPTShaderFile("RenderPasses/ComputePathTracer/ComputePathTracer.slang");

const char kInputViewDir[] = "viewW";

const ChannelList kInputChannels = {
    // clang-format off
    { "vbuffer",        "gVBuffer",     "Visibility buffer in packed format" },
    { kInputViewDir,    "gViewW",       "World-space view direction (xyz float format)" },
    // clang-format on
};

const ChannelList kOutputChannels = {
    // clang-format off
    { "color",          "gOutputColor", "Output color (sum of direct and indirect)", false, ResourceFormat::RGBA32Float },
    // clang-format on
};

const std::string kLowerBounceCount = "lowerBounceCount";
const std::string kUpperBounceCount = "upperBounceCount";
const std::string kUseImportanceSampling = "useImportanceSampling";
const std::string kUseNEE = "useNEE";
const std::string kUseMIS = "useMIS";
const std::string kMISUsePowerHeuristic = "MISUsePowerHeuristic";
const std::string kUseRR = "useRR";
const std::string kRRProbStartValue = "RRProbStartValue";
const std::string kRRProbReductionFactor = "RRProbReductionFactor";
const std::string kLightBVHOptions = "lightBVHOptions";
} // namespace

extern "C" FALCOR_API_EXPORT void registerPlugin(Falcor::PluginRegistry& registry)
{
    registry.registerClass<RenderPass, ComputePathTracer>();
}

void ComputePathTracer::parseProperties(const Properties& props)
{
    for (const auto& [key, value] : props)
    {
        if (key == kLowerBounceCount) mLowerBounceCount = value;
        else if (key == kUpperBounceCount) mUpperBounceCount = value;
        else if (key == kUseImportanceSampling) mUseImportanceSampling = value;
        else if (key == kUseNEE) mUseNEE = value;
        else if (key == kUseMIS) mUseMIS = value;
        else if (key == kMISUsePowerHeuristic) mMISUsePowerHeuristic = value;
        else if (key == kUseRR) mUseRR = value;
        else if (key == kLightBVHOptions) mLightBVHOptions = value;
        else logWarning("Unknown property '{}' in ComputePathTracer properties.", key);
    }
}

ComputePathTracer::ComputePathTracer(ref<Device> pDevice, const Properties& props) : RenderPass(pDevice)
{
    mpSampleGenerator = SampleGenerator::create(mpDevice, SAMPLE_GENERATOR_UNIFORM);
    mpPixelDebug = std::make_unique<PixelDebug>(mpDevice);
    parseProperties(props);
}

void ComputePathTracer::reset()
{
    // Retain the options for the emissive sampler.
    if (auto lightBVHSampler = dynamic_cast<LightBVHSampler*>(mpEmissiveSampler.get()))
    {
        mLightBVHOptions = lightBVHSampler->getOptions();
    }
    mpEmissiveSampler = nullptr;
    mpEnvMapSampler = nullptr;
    mpSamplerBlock = nullptr;
    mFrameCount = 0;
    mpFillCachePass = nullptr;
}

void ComputePathTracer::setProperties(const Properties& props)
{
    parseProperties(props);
    mOptionsChanged = true;
    reset();
}

Properties ComputePathTracer::getProperties() const
{
    Properties props;
    props[kLowerBounceCount] = mLowerBounceCount;
    props[kUpperBounceCount] = mUpperBounceCount;
    props[kUseImportanceSampling] = mUseImportanceSampling;
    props[kUseNEE] = mUseNEE;
    props[kUseMIS] = mUseMIS;
    props[kMISUsePowerHeuristic] = mMISUsePowerHeuristic;
    props[kUseRR] = mUseRR;
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

void ComputePathTracer::createPasses(const RenderData& renderData)
{
    DefineList defineList = getValidResourceDefines(kInputChannels, renderData);
    defineList.add(getValidResourceDefines(kOutputChannels, renderData));
    defineList.add(mpScene->getSceneDefines());
    defineList.add(mpSampleGenerator->getDefines());
    if (mpEmissiveSampler) defineList.add(mpEmissiveSampler->getDefines());
    defineList["LOWER_BOUNCE_COUNT"] = std::to_string(mLowerBounceCount);
    defineList["UPPER_BOUNCE_COUNT"] = std::to_string(mUpperBounceCount);
    defineList["USE_NEE"] = mUseNEE ? "1" : "0";
    defineList["USE_MIS"] = mUseMIS ? "1" : "0";
    defineList["MIS_USE_POWER_HEURISTIC"] = mMISUsePowerHeuristic ? "1" : "0";
    defineList["USE_RR"] = mUseRR ? "1" : "0";
    defineList["RR_PROB_START_VALUE"] = fmt::format("{:.4f}", mRRProbStartValue);
    defineList["RR_PROB_REDUCTION_FACTOR"] = fmt::format("{:.4f}", mRRProbReductionFactor);
    defineList["DEBUG_PATH_LENGTH"] = mDebugPathLength ? "1" : "0";
    defineList["USE_IMPORTANCE_SAMPLING"] = mUseImportanceSampling ? "1" : "0";
    defineList["USE_ANALYTIC_LIGHTS"] = mpScene->useAnalyticLights() ? "1" : "0";
    defineList["USE_EMISSIVE_LIGHTS"] = mpScene->useEmissiveLights() ? "1" : "0";
    defineList["USE_ENV_LIGHT"] = mpScene->useEnvLight() ? "1" : "0";
    defineList["USE_ENV_BACKGROUND"] = mpScene->useEnvBackground() ? "1" : "0";

    if (!mpFillCachePass)
    {
        ProgramDesc desc;
        desc.addShaderModules(mpScene->getShaderModules());
        desc.addShaderLibrary(kPTShaderFile).csEntry("main");
        desc.addTypeConformances(mpScene->getTypeConformances());
        mpFillCachePass = ComputePass::create(mpDevice, desc, defineList, true);
    }

void ComputePathTracer::setupData(RenderContext* pRenderContext)
{
    if (mpScene->useEnvLight())
    {
        if (!mpEnvMapSampler)
        {
            mpEnvMapSampler = std::make_unique<EnvMapSampler>(mpDevice, mpScene->getEnvMap());
        }
    }
    if (!mpEmissiveSampler && mpScene->getRenderSettings().useEmissiveLights)
    {
        const auto& pLights = mpScene->getLightCollection(pRenderContext);
        mpEmissiveSampler = std::make_unique<LightBVHSampler>(pRenderContext, mpScene, mLightBVHOptions);
    }
}

void ComputePathTracer::setupBuffers()
{
    if (!mpSamplerBlock)
    {
        mpSamplerBlock = ParameterBlock::create(mpDevice, mpPathTracingPass->getProgram()->getReflector()->getParameterBlock("gSampler"));
    }
}

void ComputePathTracer::bindData(const RenderData& renderData, uint2 frameDim)
{
    auto var = mpFillCachePass->getRootVar();
    var["CB"]["gFrameDim"] = frameDim;
    var["CB"]["gFrameCount"] = mFrameCount;
    mpScene->bindShaderData(var["gScene"]);
    mpSampleGenerator->bindShaderData(var);
    if (mpEnvMapSampler) mpEnvMapSampler->bindShaderData(mpSamplerBlock->getRootVar()["envMapSampler"]);
    if (mpEmissiveSampler) mpEmissiveSampler->bindShaderData(mpSamplerBlock->getRootVar()["emissiveSampler"]);
    var["gSampler"] = mpSamplerBlock;
    for (auto channel : kInputChannels) var[channel.texname] = renderData.getTexture(channel.name);
    for (auto channel : kOutputChannels) var[channel.texname] = renderData.getTexture(channel.name);
    mpPixelDebug->prepareProgram(mpFillCachePass->getProgram(), var);
}

void ComputePathTracer::execute(RenderContext* pRenderContext, const RenderData& renderData)
{
    const auto& pOutput = renderData.getTexture("color");
    uint2 frameDim = {pOutput->getWidth(), pOutput->getHeight()};

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
    mpPixelDebug->beginFrame(pRenderContext, frameDim);

    if (is_set(mpScene->getUpdates(), Scene::UpdateFlags::RecompileNeeded) ||
        is_set(mpScene->getUpdates(), Scene::UpdateFlags::GeometryChanged))
    {
        FALCOR_THROW("This render pass does not support scene changes that require shader recompilation.");
    }

    if (mpEmissiveSampler)
    {
        if (mpEmissiveSampler->update(pRenderContext))
        {
            renderData.getDictionary()[Falcor::kRenderPassRefreshFlags] = Falcor::RenderPassRefreshFlags::LightingChanged;
        }
    }

    if (mOptionsChanged)
    {
        reset();
        renderData.getDictionary()[Falcor::kRenderPassRefreshFlags] = Falcor::RenderPassRefreshFlags::RenderOptionsChanged;
        setupData(pRenderContext);
        createPasses(renderData);
        setupBuffers();
        mOptionsChanged = false;
    }
    bindData(renderData, frameDim);

    // get dimensions of ray dispatch.
    const uint2 targetDim = renderData.getDefaultTextureDims();
    FALCOR_ASSERT(targetDim.x > 0 && targetDim.y > 0);
    mpFillCachePass->execute(pRenderContext, frameDim.x, frameDim.y);
    mpPixelDebug->endFrame(pRenderContext);
    mFrameCount++;
}

void ComputePathTracer::renderUI(Gui::Widgets& widget)
{
    if (kUseImGui)
    {
        ImGui::PushItemWidth(40);
        ImGui::Text("Bounce limits");
        ImGui::SameLine();
        ImGui::InputScalar("min", ImGuiDataType_U32, &mLowerBounceCount);
        ImGui::SameLine();
        ImGui::InputScalar("max", ImGuiDataType_U32, &mUpperBounceCount);
        ImGui::PopItemWidth();
    }
    else
    {
        widget.var("LowerBounceCount", mLowerBounceCount, 0u, 1u << 16);
        widget.var("UpperBounceCount", mUpperBounceCount, 0u, 1u << 16);
    }
    widget.tooltip("Inclusive range of bounces that contribute to final image color", true);

    widget.checkbox("BSDF importance sampling", mUseImportanceSampling);
    widget.tooltip("Use importance sampling for materials", true);

    widget.checkbox("NEE", mUseNEE);
    widget.checkbox("MIS", mUseMIS);
    widget.checkbox("power heuristic", mMISUsePowerHeuristic, true);
    widget.tooltip("Active: power heuristic; Inactive: balance heuristic", true);
    widget.checkbox("RR", mUseRR);
    if (kUseImGui)
    {
        ImGui::PushItemWidth(80);
        ImGui::InputFloat("RR start value", &mRRProbStartValue);
        ImGui::PopItemWidth();
    }
    else
    {
        widget.var("RR start value", mRRProbStartValue, 1.0f, 4.0f);
    }
    widget.tooltip("Starting value of the survival probability", true);
    if (kUseImGui)
    {
        ImGui::PushItemWidth(80);
        ImGui::InputFloat("RR reduction factor", &mRRProbReductionFactor);
        ImGui::PopItemWidth();
    }
    else
    {
        widget.var("RR reduction factor", mRRProbReductionFactor, 0.1f, 0.99f);
    }
    widget.tooltip("Gets multiplied to the initial survival probability at each interaction", true);
    if (Gui::Group emissive_sampler_group = widget.group("EmissiveSampler"))
    {
        if (mpEmissiveSampler) mpEmissiveSampler->renderUI(emissive_sampler_group);
    }
    if (Gui::Group debug_group = widget.group("Debug"))
    {
        debug_group.checkbox("Path length", mDebugPathLength);
        mpPixelDebug->renderUI(debug_group);
    }

    // reload shader and set options change flag (explicitly apply changes)
    // In execute() we will pass the flag to other passes for reset of temporal data etc.
    if (widget.button("Reload shader"))
    {
        mOptionsChanged = true;
        reset();
    }
}

void ComputePathTracer::setScene(RenderContext* pRenderContext, const ref<Scene>& pScene)
{
    mpScene = pScene;
    reset();
}


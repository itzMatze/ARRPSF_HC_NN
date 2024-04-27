#include "ComputePathTracer.h"
#include "RenderGraph/RenderPassHelpers.h"
#include "RenderGraph/RenderPassStandardFlags.h"
#include "imgui.h"

#include <random>
#include <string>

namespace
{
constexpr bool kUseImGui = true;
const std::string kPTShaderFile("RenderPasses/ComputePathTracer/ComputePathTracer.slang");
const std::string khashCacheResolveShaderFile("RenderPasses/ComputePathTracer/HashCacheResolve.slang");
const std::string kGradientClearShaderFile("RenderPasses/ComputePathTracer/tinynn/GradientClear.slang");
const std::string kGradientDescentShaderFile("RenderPasses/ComputePathTracer/tinynn/GradientDescentPrimal.slang");

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
const std::string kEnableHashCache = "enableHashCache";
const std::string kEnableNN = "enableNN";
const std::string kHashCacheHashMapSizeExponent = "hashCacheHashMapSizeExponent";
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
        else if (key == kEnableHashCache) mEnableHashCache = value;
        else if (key == kEnableNN) mNNParams.enable = value;
        else if (key == kHashCacheHashMapSizeExponent)
        {
            mHashCacheHashMapSizeExp = uint32_t(value);
            mHashCacheHashMapSize = std::pow(2u, mHashCacheHashMapSizeExp);
        }
        else logWarning("Unknown property '{}' in ComputePathTracer properties.", key);
    }
}

ComputePathTracer::ComputePathTracer(ref<Device> pDevice, const Properties& props) : RenderPass(pDevice)
{
    mpSampleGenerator = SampleGenerator::create(mpDevice, SAMPLE_GENERATOR_UNIFORM);
    mpPixelDebug = std::make_unique<PixelDebug>(mpDevice);
    //mpPixelDebug->enable();
    parseProperties(props);
}

void ComputePathTracer::reset()
{
    mNNParams.optimizerParams.step_count = 0;
    mNNParams.update();
    // Retain the options for the emissive sampler.
    if (auto lightBVHSampler = dynamic_cast<LightBVHSampler*>(mpEmissiveSampler.get()))
    {
        mLightBVHOptions = lightBVHSampler->getOptions();
    }
    mpEmissiveSampler = nullptr;
    mpEnvMapSampler = nullptr;
    mpSamplerBlock = nullptr;
    for (auto& b : mBuffers) b = nullptr;
    mFrameCount = 0;
    for (auto& p : mPasses) p = nullptr;
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
    props[kEnableHashCache] = mEnableHashCache;
    props[kEnableNN] = mNNParams.enable;
    props[kHashCacheHashMapSizeExponent] = mHashCacheHashMapSizeExp;
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
    defineList["HASH_CACHE_DEBUG_VOXELS"] = mHashCacheDebugVoxels ? "1" : "0";
    defineList["HASH_CACHE_DEBUG_COLOR"] = mHashCacheDebugColor ? "1" : "0";
    defineList["HASH_CACHE_DEBUG_LEVELS"] = mHashCacheDebugLevels ? "1" : "0";
    defineList["HASH_CACHE_HASHMAP_SIZE"] = std::to_string(mHashCacheHashMapSize);
    defineList["USE_IMPORTANCE_SAMPLING"] = mUseImportanceSampling ? "1" : "0";
    defineList["USE_ANALYTIC_LIGHTS"] = mpScene->useAnalyticLights() ? "1" : "0";
    defineList["USE_EMISSIVE_LIGHTS"] = mpScene->useEmissiveLights() ? "1" : "0";
    defineList["USE_ENV_LIGHT"] = mpScene->useEnvLight() ? "1" : "0";
    defineList["USE_ENV_BACKGROUND"] = mpScene->useEnvBackground() ? "1" : "0";
    defineList["NN_DEBUG"] = mNNParams.debugOutput ? "1" : "0";
    defineList["NN_PARAM_COUNT"] = std::to_string(mNNParams.nnParamCount);
    defineList["NN_GRAD_OFFSET"] = std::to_string(mNNParams.gradOffset);
    defineList["NN_OPTIMIZER_TYPE"] = std::to_string(mNNParams.optimizerParams.type);
    defineList["NN_LEARNING_RATE"] = fmt::format("{:.12f}", mNNParams.optimizerParams.learn_r);
    defineList["NN_PARAM_0"] = fmt::format("{:.12f}", mNNParams.optimizerParams.param_0);
    defineList["NN_PARAM_1"] = fmt::format("{:.12f}", mNNParams.optimizerParams.param_1);
    defineList["NN_PARAM_2"] = fmt::format("{:.12f}", mNNParams.optimizerParams.param_2);
    defineList["NN_LAYER_WIDTH"] = std::to_string(mNNParams.nnLayerWidth);
    defineList["NN_LAYER_COUNT"] = std::to_string(mNNParams.nnLayerCount);

    if (!mPasses[TRAIN_NN_FILL_CACHE_PASS] && (mHashCacheActive || mNNParams.active))
    {
        defineList["HASH_CACHE_UPDATE"] = mHashCacheActive ? "1" : "0";
        defineList["HASH_CACHE_QUERY"] = "0";
        defineList["NN_TRAIN"] = mNNParams.active ? "1" : "0";
        defineList["NN_QUERY"] = "0";
        ProgramDesc desc;
        desc.addShaderModules(mpScene->getShaderModules());
        desc.addShaderLibrary(kPTShaderFile).csEntry("main");
        desc.addTypeConformances(mpScene->getTypeConformances());
        mPasses[TRAIN_NN_FILL_CACHE_PASS] = ComputePass::create(mpDevice, desc, defineList, true);
    }
    if (!mPasses[PATH_TRACING_PASS])
    {
        defineList["HASH_CACHE_UPDATE"] = "0";
        defineList["HASH_CACHE_QUERY"] = mHashCacheActive ? "1" : "0";
        defineList["NN_TRAIN"] = "0";
        defineList["NN_QUERY"] = mNNParams.active ? "1" : "0";
        ProgramDesc desc;
        desc.addShaderModules(mpScene->getShaderModules());
        desc.addShaderLibrary(kPTShaderFile).csEntry("main");
        desc.addTypeConformances(mpScene->getTypeConformances());
        mPasses[PATH_TRACING_PASS] = ComputePass::create(mpDevice, desc, defineList, true);
    }
    if (!mPasses[RESOLVE_PASS] && mHashCacheActive)
    {
        ProgramDesc desc;
        desc.addShaderLibrary(khashCacheResolveShaderFile).csEntry("hashCacheResolve");
        mPasses[RESOLVE_PASS] = ComputePass::create(mpDevice, desc, defineList, true);
    }
    if (!mPasses[NN_GRADIENT_CLEAR_PASS] && mNNParams.active)
    {
        ProgramDesc desc;
        desc.addShaderLibrary(kGradientClearShaderFile).csEntry("main");
        mPasses[NN_GRADIENT_CLEAR_PASS] = ComputePass::create(mpDevice, desc, defineList, true);
    }
    if (!mPasses[NN_GRADIENT_DESCENT_PASS] && mNNParams.active)
    {
        ProgramDesc desc;
        desc.addShaderLibrary(kGradientDescentShaderFile).csEntry("main");
        mPasses[NN_GRADIENT_DESCENT_PASS] = ComputePass::create(mpDevice, desc, defineList, true);
    }
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
    if (mHashCacheActive)
    {
        if (!mBuffers[HASH_ENTRIES_BUFFER]) mBuffers[HASH_ENTRIES_BUFFER] = mpDevice->createStructuredBuffer(sizeof(uint32_t), mHashCacheHashMapSize);
        // 128 bits per entry
        if (!mBuffers[HASH_CACHE_VOXEL_DATA_BUFFER_0]) mBuffers[HASH_CACHE_VOXEL_DATA_BUFFER_0] = mpDevice->createBuffer((128 / 8) * mHashCacheHashMapSize);
        if (!mBuffers[HASH_CACHE_VOXEL_DATA_BUFFER_1]) mBuffers[HASH_CACHE_VOXEL_DATA_BUFFER_1] = mpDevice->createBuffer((128 / 8) * mHashCacheHashMapSize);
    }
    std::mt19937 rnd(0);  // Generates random integers
    std::uniform_real_distribution<float> dis(mNNParams.weightInitBound.x, mNNParams.weightInitBound.y);
    auto gen = [&dis, &rnd](){ return dis(rnd); };
    std::vector<float> data(mNNParams.nnParamCount);
    std::generate(data.begin(), data.end(), gen);
    if (!mBuffers[NN_PRIMAL_BUFFER]) mBuffers[NN_PRIMAL_BUFFER] = mpDevice->createBuffer(mNNParams.nnParamCount * sizeof(float), ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess, MemoryType::DeviceLocal, data.data());
    if (!mBuffers[NN_GRADIENT_BUFFER]) mBuffers[NN_GRADIENT_BUFFER] = mpDevice->createBuffer(mNNParams.nnParamCount * sizeof(float), ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess, MemoryType::DeviceLocal, data.data());
    if (!mBuffers[NN_GRADIENT_AUX_BUFFER]) mBuffers[NN_GRADIENT_AUX_BUFFER] = mpDevice->createBuffer(mNNParams.nnParamCount * sizeof(float) * 4);
}

void ComputePathTracer::setupBuffers()
{
    if (!mpSamplerBlock)
    {
        mpSamplerBlock = ParameterBlock::create(mpDevice, mPasses[PATH_TRACING_PASS]->getProgram()->getReflector()->getParameterBlock("gSampler"));
    }
}

void ComputePathTracer::bindData(const RenderData& renderData, uint2 frameDim)
{
    mCamPos = mpScene->getCamera()->getPosition();
    if (mHashCacheActive || mNNParams.active)
    {
        auto var = mPasses[TRAIN_NN_FILL_CACHE_PASS]->getRootVar();
        var["CB"]["gFrameDim"] = frameDim;
        var["CB"]["gFrameCount"] = mFrameCount;
        var["CB"]["gCamPos"] = mCamPos;
        mpScene->bindShaderData(var["gScene"]);
        mpSampleGenerator->bindShaderData(var);
        if (mpEnvMapSampler) mpEnvMapSampler->bindShaderData(mpSamplerBlock->getRootVar()["envMapSampler"]);
        if (mpEmissiveSampler) mpEmissiveSampler->bindShaderData(mpSamplerBlock->getRootVar()["emissiveSampler"]);
        var["gSampler"] = mpSamplerBlock;
        if (mHashCacheActive)
        {
            var["gHashEntriesBuffer"] = mBuffers[HASH_ENTRIES_BUFFER];
            var["gHashCacheVoxelDataBuffer"] = mFrameCount % 2 == 0 ? mBuffers[HASH_CACHE_VOXEL_DATA_BUFFER_0] : mBuffers[HASH_CACHE_VOXEL_DATA_BUFFER_1];
            var["gHashCacheVoxelDataBufferPrev"] = mFrameCount % 2 == 1 ? mBuffers[HASH_CACHE_VOXEL_DATA_BUFFER_0] : mBuffers[HASH_CACHE_VOXEL_DATA_BUFFER_1];
        }
        if (mNNParams.active)
        {
            var["PrimalBuffer"] = mBuffers[NN_PRIMAL_BUFFER];
            var["GradientBuffer"] = mBuffers[NN_GRADIENT_BUFFER];
        }
        for (auto channel : kInputChannels) var[channel.texname] = renderData.getTexture(channel.name);
        for (auto channel : kOutputChannels) var[channel.texname] = renderData.getTexture(channel.name);
        mpPixelDebug->prepareProgram(mPasses[TRAIN_NN_FILL_CACHE_PASS]->getProgram(), var);
    }
    if (mHashCacheActive)
    {
        auto var = mPasses[RESOLVE_PASS]->getRootVar();
        var["CB"]["gCamPos"] = mCamPos;
        var["gHashEntriesBuffer"] = mBuffers[HASH_ENTRIES_BUFFER];
        var["gHashCacheVoxelDataBuffer"] = mFrameCount % 2 == 0 ? mBuffers[HASH_CACHE_VOXEL_DATA_BUFFER_0] : mBuffers[HASH_CACHE_VOXEL_DATA_BUFFER_1];
        var["gHashCacheVoxelDataBufferPrev"] = mFrameCount % 2 == 1 ? mBuffers[HASH_CACHE_VOXEL_DATA_BUFFER_0] : mBuffers[HASH_CACHE_VOXEL_DATA_BUFFER_1];
        mpPixelDebug->prepareProgram(mPasses[RESOLVE_PASS]->getProgram(), var);
    }
    {
        auto var = mPasses[PATH_TRACING_PASS]->getRootVar();
        var["CB"]["gFrameDim"] = frameDim;
        var["CB"]["gFrameCount"] = mFrameCount;
        var["CB"]["gCamPos"] = mCamPos;
        mpScene->bindShaderData(var["gScene"]);
        mpSampleGenerator->bindShaderData(var);
        if (mpEnvMapSampler) mpEnvMapSampler->bindShaderData(mpSamplerBlock->getRootVar()["envMapSampler"]);
        if (mpEmissiveSampler) mpEmissiveSampler->bindShaderData(mpSamplerBlock->getRootVar()["emissiveSampler"]);
        var["gSampler"] = mpSamplerBlock;
        if (mHashCacheActive)
        {
            var["gHashEntriesBuffer"] = mBuffers[HASH_ENTRIES_BUFFER];
            var["gHashCacheVoxelDataBuffer"] = mFrameCount % 2 == 0 ? mBuffers[HASH_CACHE_VOXEL_DATA_BUFFER_0] : mBuffers[HASH_CACHE_VOXEL_DATA_BUFFER_1];
            var["gHashCacheVoxelDataBufferPrev"] = mFrameCount % 2 == 1 ? mBuffers[HASH_CACHE_VOXEL_DATA_BUFFER_0] : mBuffers[HASH_CACHE_VOXEL_DATA_BUFFER_1];
        }
        if (mNNParams.active)
        {
            var["PrimalBuffer"] = mBuffers[NN_PRIMAL_BUFFER];
            var["GradientBuffer"] = mBuffers[NN_GRADIENT_BUFFER];
        }
        for (auto channel : kInputChannels) var[channel.texname] = renderData.getTexture(channel.name);
        for (auto channel : kOutputChannels) var[channel.texname] = renderData.getTexture(channel.name);
        mpPixelDebug->prepareProgram(mPasses[PATH_TRACING_PASS]->getProgram(), var);
    }
    if (mNNParams.active)
    {
        auto var = mPasses[NN_GRADIENT_CLEAR_PASS]->getRootVar();
        var["GradientBuffer"] = mBuffers[NN_GRADIENT_BUFFER];
        mpPixelDebug->prepareProgram(mPasses[NN_GRADIENT_CLEAR_PASS]->getProgram(), var);
    }
    if (mNNParams.active)
    {
        auto var = mPasses[NN_GRADIENT_DESCENT_PASS]->getRootVar();
        var["CB"]["t"] = mNNParams.optimizerParams.step_count;
        var["PrimalBuffer"] = mBuffers[NN_PRIMAL_BUFFER];
        var["GradientBuffer"] = mBuffers[NN_GRADIENT_BUFFER];
        var["GradientAuxBuffer"] = mBuffers[NN_GRADIENT_AUX_BUFFER];
        mpPixelDebug->prepareProgram(mPasses[NN_GRADIENT_BUFFER]->getProgram(), var);
    }
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
        mHashCacheActive = mEnableHashCache;
        mNNParams.active = mNNParams.enable;
        setupData(pRenderContext);
        createPasses(renderData);
        setupBuffers();
        mOptionsChanged = false;
    }
    bindData(renderData, frameDim);

    const uint2 targetDim = renderData.getDefaultTextureDims();
    FALCOR_ASSERT(targetDim.x > 0 && targetDim.y > 0);
    for (uint32_t i = 0; i < 4; i++)
    {
        if (mNNParams.active) mPasses[NN_GRADIENT_CLEAR_PASS]->execute(pRenderContext, mNNParams.nnParamCount, 1);
        if (mHashCacheActive || mNNParams.active)
        {
            mPasses[TRAIN_NN_FILL_CACHE_PASS]->getRootVar()["CB"]["gTrainIteration"] = i;
            mPasses[TRAIN_NN_FILL_CACHE_PASS]->execute(pRenderContext, frameDim.x / 10, frameDim.y / 10);
        }
        if (mNNParams.active) mPasses[NN_GRADIENT_DESCENT_PASS]->execute(pRenderContext, mNNParams.nnParamCount, 1);
    }
    if (mHashCacheActive) mPasses[RESOLVE_PASS]->execute(pRenderContext, mHashCacheHashMapSize, 1);
    mPasses[PATH_TRACING_PASS]->execute(pRenderContext, frameDim.x, frameDim.y);
    mpPixelDebug->endFrame(pRenderContext);
    mFrameCount++;
    mNNParams.optimizerParams.step_count++;
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
    if (Gui::Group hash_cache_group = widget.group("Hash Cache"))
    {
        hash_cache_group.checkbox("Enable Hash Cache", mEnableHashCache);
        if (kUseImGui)
        {
            ImGui::PushItemWidth(40);
            ImGui::InputScalar("hashMapSizeExponent", ImGuiDataType_U32, &mHashCacheHashMapSizeExp);
            ImGui::PopItemWidth();
        }
        hash_cache_group.checkbox("Debug voxels", mHashCacheDebugVoxels);
        hash_cache_group.checkbox("Debug color", mHashCacheDebugColor);
        hash_cache_group.checkbox("Debug levels", mHashCacheDebugLevels);
    }
    if (Gui::Group nn_group = widget.group("NN"))
    {
        nn_group.checkbox("Enable NN", mNNParams.enable);
        if (kUseImGui)
        {
            if (Gui::Group nn_optimizer_group = nn_group.group("Optimizer"))
            {
                ImGui::PushItemWidth(160);
                ImGui::InputFloat("learning rate", &mNNParams.optimizerParams.learn_r, 0.0f, 0.0f, "%.6f");
                if (mNNParams.optimizerParams.type == mNNParams.SGD)
                {
                    ImGui::InputFloat("momentum", &mNNParams.optimizerParams.param_0, 0.0f, 0.0f, "%.8f");
                    ImGui::InputFloat("dampening", &mNNParams.optimizerParams.param_1, 0.0f, 0.0f, "%.8f");
                }
                else if (mNNParams.optimizerParams.type == mNNParams.ADAM)
                {
                    ImGui::InputFloat("beta_1", &mNNParams.optimizerParams.param_0, 0.0f, 0.0f, "%.8f");
                    ImGui::InputFloat("beta_2", &mNNParams.optimizerParams.param_1, 0.0f, 0.0f, "%.8f");
                    ImGui::InputFloat("epsilon", &mNNParams.optimizerParams.param_2, 0.0f, 0.0f, "%.8f");
                }
                ImGui::PopItemWidth();
            }
            ImGui::PushItemWidth(120);
            nn_group.dropdown("NN layer width", mNNParams.nnLayerWidthList, mNNParams.nnLayerWidth);
            ImGui::InputInt("NN layer count", &mNNParams.nnLayerCount);
            nn_group.checkbox("Debug NN output", mNNParams.debugOutput);
            ImGui::Text("Weight init bounds");
            ImGui::InputFloat("min", &mNNParams.weightInitBound.x, 0.0f, 0.0f, "%.6f");
            ImGui::InputFloat("max", &mNNParams.weightInitBound.y, 0.0f, 0.0f, "%.6f");
            ImGui::PopItemWidth();
            ImGui::Separator();
        }
        else
        { }
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
        mHashCacheHashMapSize = std::pow(2, mHashCacheHashMapSizeExp);
        mOptionsChanged = true;
        reset();
    }
}

void ComputePathTracer::setScene(RenderContext* pRenderContext, const ref<Scene>& pScene)
{
    mpScene = pScene;
    mCamPos = mpScene->getCamera()->getPosition();
    reset();
}


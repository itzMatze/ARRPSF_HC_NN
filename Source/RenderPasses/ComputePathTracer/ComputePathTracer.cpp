#include "ComputePathTracer.h"
#include "Core/API/Formats.h"
#include "RenderGraph/RenderPassHelpers.h"
#include "RenderGraph/RenderPassStandardFlags.h"
#include "imgui.h"

#include <string>

namespace
{
const std::string kPTShaderFile("RenderPasses/ComputePathTracer/ComputePathTracer.slang");
const std::string kPTTrainShaderFile("RenderPasses/ComputePathTracer/ComputePathTracerTrain.slang");
const std::string kHCResolveShaderFile("RenderPasses/ComputePathTracer/RadianceHashCacheResolve.slang");
const std::string kHCResetShaderFile("RenderPasses/ComputePathTracer/RadianceHashCacheReset.slang");
const std::string kGradientClearShaderFile("RenderPasses/ComputePathTracer/tinynn/GradientClear.slang");
const std::string kGradientDescentShaderFile("RenderPasses/ComputePathTracer/tinynn/GradientDescentPrimal.slang");
const std::string kNNResetShaderFile("RenderPasses/ComputePathTracer/tinynn/NNReset.slang");
const std::string kIRDebugVisShaderFile("RenderPasses/ComputePathTracer/IRDebugVis.slang");

// inputs
const ChannelDesc kInputVBuffer{"vbuffer", "gVBuffer", "Visibility buffer in packed format"};
const ChannelDesc kInputViewDir{"viewW", "gViewW", "World-space view direction (xyz float format)"};
const ChannelDesc kInputRefImage{"refImage", "gRefImage", "Reference image for the current scene. Used for debugging."};
const ChannelList kInputChannels = {kInputVBuffer, kInputViewDir, kInputRefImage};

// outputs
const ChannelDesc kOutputColor = { "color", "gOutputColor", "Output color (sum of direct and indirect)", false, ResourceFormat::RGBA32Float };
constexpr uint2 kIRDebugOutputDim(1000, 1000);
const ChannelDesc kIRDebugOutputColor = { "ir_debug", "gIRDebugOutputColor", "Output color of IR debug visualization", false, ResourceFormat::RGBA32Float };
const ChannelDesc kIRDebugOutputColorRef = { "ir_debug_ref", "gIRDebugOutputColorRef", "Output color of the path traced IR debug visualization for reference", false, ResourceFormat::RGBA32Float };
const ChannelList kOutputChannels = {kOutputColor, kIRDebugOutputColor};

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
const std::string kHCHashMapSizeExponent = "HCHashMapSizeExponent";
const std::string kHCInjectRadianceSpread = "HCInjectRadianceSpread";
const std::string kHCDebugColor = "HCDebugColor";
const std::string kRRSurvivalProbOption = "RRSurvivalProbOption";
const std::string kNNDebugOutput = "NNDebugOutput";
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
        else if (key == kUseRR) mRRParams.active = value;
        else if (key == kLightBVHOptions) mLightBVHOptions = value;
        else if (key == kHCHashMapSizeExponent)
        {
            mHCParams.hashMapSizeExp = uint32_t(value);
            mHCParams.hashMapSize = std::pow(2u, mHCParams.hashMapSizeExp);
        }
        else if (key == kHCInjectRadianceSpread) mHCParams.injectRadianceSpread = value;
        else if (key == kHCDebugColor) mHCParams.debugColor = value;
        else if (key == kRRSurvivalProbOption) mRRParams.survivalProbOption = value;
        else if (key == kNNDebugOutput) mNNParams.debugOutput = value;
        else logWarning("Unknown property '{}' in ComputePathTracer properties.", key);
    }
}

ComputePathTracer::ComputePathTracer(ref<Device> pDevice, const Properties& props) : RenderPass(pDevice)
{
    mpSampleGenerator = SampleGenerator::create(mpDevice, SAMPLE_GENERATOR_UNIFORM);
    mpPixelDebug = std::make_unique<PixelDebug>(mpDevice);
    mpPixelDebug->enable();
    parseProperties(props);
}

void ComputePathTracer::reset()
{
    mNNParams.optimizerParams.step_count = 0;
    mRRParams.update();
    mNNParams.update();
    mHCParams.update();
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
    props[kUseRR] = mRRParams.active;
    props[kRRProbStartValue] = mRRParams.probStartValue;
    props[kRRProbReductionFactor] = mRRParams.probReductionFactor;
    props[kHCHashMapSizeExponent] = mHCParams.hashMapSizeExp;
    props[kHCInjectRadianceSpread] = mHCParams.injectRadianceSpread;
    props[kHCDebugColor] = mHCParams.debugColor;
    props[kRRSurvivalProbOption] = mRRParams.survivalProbOption;
    props[kNNDebugOutput] = mNNParams.debugOutput;
    return props;
}

RenderPassReflection ComputePathTracer::reflect(const CompileData& compileData)
{
    RenderPassReflection reflector;
    // Define our input/output channels.
    addRenderPassInputs(reflector, kInputChannels);
    addRenderPassOutputs(reflector, {kOutputColor});
    addRenderPassOutputs(reflector, {kIRDebugOutputColor}, ResourceBindFlags::UnorderedAccess, kIRDebugOutputDim);
    addRenderPassOutputs(reflector, {kIRDebugOutputColorRef}, ResourceBindFlags::UnorderedAccess, kIRDebugOutputDim);
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
    defineList["USE_RR"] = mRRParams.active ? "1" : "0";
    defineList["RR_PROB_START_VALUE"] = fmt::format("{:.4f}", mRRParams.probStartValue);
    defineList["RR_PROB_REDUCTION_FACTOR"] = fmt::format("{:.4f}", mRRParams.probReductionFactor);
    defineList["DEBUG_PATH_LENGTH"] = mDebugPathLength ? "1" : "0";
    defineList["USE_RHC"] = mHCParams.hcMethod == HCParams::USE_RHC ? "1" : "0";
    defineList["USE_IRHC"] = mHCParams.hcMethod == HCParams::USE_IRHC ? "1" : "0";
    defineList["HC_DEBUG_VOXELS"] = mHCParams.debugVoxels ? "1" : "0";
    defineList["HC_DEBUG_COLOR"] = mHCParams.debugColor ? "1" : "0";
    defineList["HC_DEBUG_LEVELS"] = mHCParams.debugLevels ? "1" : "0";
    defineList["HC_HASHMAP_SIZE"] = std::to_string(mHCParams.hashMapSize);
    defineList["USE_IMPORTANCE_SAMPLING"] = mUseImportanceSampling ? "1" : "0";
    defineList["USE_ANALYTIC_LIGHTS"] = mpScene->useAnalyticLights() ? "1" : "0";
    defineList["USE_EMISSIVE_LIGHTS"] = mpScene->useEmissiveLights() ? "1" : "0";
    defineList["USE_ENV_LIGHT"] = mpScene->useEnvLight() ? "1" : "0";
    defineList["USE_ENV_BACKGROUND"] = mpScene->useEnvBackground() ? "1" : "0";
    defineList["USE_NRC"] = mNNParams.nnMethod == NNParams::USE_NRC ? "1" : "0";
    defineList["USE_NIRC"] = mNNParams.nnMethod == NNParams::USE_NIRC ? "1" : "0";
    defineList["NN_USE_HASH_ENC"] = mNNParams.encMethod == NNParams::USE_HASH_ENC ? "1" : "0";
    defineList["NN_USE_HASH_ENC_INTERPOLATION"] = mNNParams.encMethod == NNParams::USE_HASH_ENC_INTERPOLATION ? "1" : "0";
    defineList["NN_USE_FREQ_ENC"] = mNNParams.encMethod == NNParams::USE_FREQ_ENC ? "1" : "0";
    defineList["USE_MULTI_LEVEL_DIR"] = mNNParams.featureHashEncUseMultiLevelDir ? "1" : "0";
    defineList["NN_DEBUG"] = mNNParams.debugOutput ? "1" : "0";
    defineList["IR_DEBUG_OUTPUT_WIDTH"] = std::to_string(kIRDebugOutputDim.x);
    defineList["IR_DEBUG_OUTPUT_HEIGHT"] = std::to_string(kIRDebugOutputDim.y);
    defineList["NN_PARAM_COUNT"] = std::to_string(mNNParams.nnParamCount);
    defineList["NN_WEIGHT_INIT_LOWER_BOUND"] = fmt::format("{:.12f}", mNNParams.weightInitBound.x);
    defineList["NN_WEIGHT_INIT_UPPER_BOUND"] = fmt::format("{:.12f}", mNNParams.weightInitBound.y);
    defineList["NN_GRAD_OFFSET"] = std::to_string(mNNParams.gradOffset);
    defineList["NN_GRADIENT_AUX_ELEMENTS"] = std::to_string(mNNParams.gradientAuxElements);
    defineList["NN_OPTIMIZER_TYPE"] = std::to_string(mNNParams.optimizerParams.type);

    defineList["NN_PARAM_0"] = fmt::format("{:.12f}", mNNParams.optimizerParams.param_0);
    defineList["NN_PARAM_1"] = fmt::format("{:.12f}", mNNParams.optimizerParams.param_1);
    defineList["NN_LAYER_WIDTH"] = std::to_string(mNNParams.nnLayerWidth);
    defineList["MLP_COUNT"] = std::to_string(mNNParams.nnLayerCount.size());
    for (uint i = 0; i < mNNParams.nnLayerCount.size(); i++) defineList[std::string("NN_LAYER_COUNT") + std::to_string(i)] = std::to_string(mNNParams.nnLayerCount[i]);
    defineList["NN_TRAINING_BOUNCES"] = std::to_string(mNNParams.trainingBounces);
    defineList["FEATURE_HASH_GRID_SIZE"] = std::to_string(mNNParams.featureHashMapSize);
    defineList["FEATURE_HASH_GRID_PLACES_PER_ELEMENT"] = std::to_string(mNNParams.featureHashMapPlacesPerElement);
    defineList["FEATURE_HASH_ENC_SEPARATE_LEVEL_GRIDS"] = mNNParams.featureHashEncSeparateLevelGrids ? "1" : "0";
    defineList["FEATURE_HASH_GRID_PROBING_SIZE"] = std::to_string(mNNParams.featureHashMapProbingSize);

    if (!mPasses[TRAIN_NN_FILL_CACHE_PASS] && (mHCParams.active || mNNParams.active))
    {
        defineList["HC_UPDATE"] = mHCParams.active ? "1" : "0";
        defineList["HC_QUERY"] = "0";
        defineList["NN_TRAIN"] = mNNParams.active ? "1" : "0";
        defineList["NN_QUERY"] = "0";
        // use default rr for training
        defineList["RR_OPTION_BITS"] = "0";
        defineList["HC_INJECT_RADIANCE_SPREAD"] = "0";
        ProgramDesc desc;
        desc.addShaderModules(mpScene->getShaderModules());
        desc.addShaderLibrary(kPTTrainShaderFile).csEntry("main");
        desc.addTypeConformances(mpScene->getTypeConformances());
        mPasses[TRAIN_NN_FILL_CACHE_PASS] = ComputePass::create(mpDevice, desc, defineList, true);
    }
    if (!mPasses[PATH_TRACING_PASS])
    {
        defineList["HC_UPDATE"] = "0";
        defineList["HC_QUERY"] = mHCParams.active ? "1" : "0";
        defineList["NN_TRAIN"] = "0";
        defineList["NN_QUERY"] = mNNParams.active ? "1" : "0";
        defineList["RR_OPTION_BITS"] = std::to_string(mRRParams.getOptionBits());
        // when using the nn during pt the threads need to be kept running for the cooperative matrices
        defineList["KEEP_THREADS"] = mNNParams.keepThreads ? "1" : "0";
        defineList["INJECT_RADIANCE_RR"] = mRRParams.injectRadiance ? "1" : "0";
        defineList["HC_INJECT_RADIANCE_SPREAD"] = mHCParams.injectRadianceSpread ? "1" : "0";
        defineList["NN_INJECT_RADIANCE_SPREAD"] = mNNParams.injectRadianceSpread ? "1" : "0";
        ProgramDesc desc;
        desc.addShaderModules(mpScene->getShaderModules());
        desc.addShaderLibrary(kPTShaderFile).csEntry("main");
        desc.addTypeConformances(mpScene->getTypeConformances());
        mPasses[PATH_TRACING_PASS] = ComputePass::create(mpDevice, desc, defineList, true);
    }
    if (!mPasses[HC_RESOLVE_PASS] && mHCParams.active)
    {
        defineList["HC_UPDATE"] = "1";
        defineList["HC_QUERY"] = "1";
        ProgramDesc desc;
        desc.addShaderLibrary(kHCResolveShaderFile).csEntry("hashCacheResolve");
        mPasses[HC_RESOLVE_PASS] = ComputePass::create(mpDevice, desc, defineList, true);
    }
    if (!mPasses[HC_RESET_PASS] && mHCParams.active)
    {
        defineList["HC_UPDATE"] = "1";
        defineList["HC_QUERY"] = "1";
        ProgramDesc desc;
        desc.addShaderLibrary(kHCResetShaderFile).csEntry("main");
        mPasses[HC_RESET_PASS] = ComputePass::create(mpDevice, desc, defineList, true);
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
    if (!mPasses[NN_RESET_PASS] && mNNParams.active)
    {
        ProgramDesc desc;
        desc.addShaderLibrary(kNNResetShaderFile).csEntry("main");
        mPasses[NN_RESET_PASS] = ComputePass::create(mpDevice, desc, defineList, true);
    }
    if (!mPasses[IR_DEBUG_PASS] && mIRDebugPassParams.active)
    {
        defineList["SHOW_NIRC"] = mIRDebugPassParams.irMethod == IRDebugPassParam::SHOW_NIRC ? "1" : "0";
        defineList["SHOW_IRHC"] = mIRDebugPassParams.irMethod == IRDebugPassParam::SHOW_IRHC ? "1" : "0";
        ProgramDesc desc;
        desc.addShaderModules(mpScene->getShaderModules());
        desc.addShaderLibrary(kIRDebugVisShaderFile).csEntry("main");
        desc.addTypeConformances(mpScene->getTypeConformances());
        mPasses[IR_DEBUG_PASS] = ComputePass::create(mpDevice, desc, defineList, true);
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
    if (mHCParams.active)
    {
        if (!mBuffers[HC_HASH_GRID_ENTRIES_BUFFER]) mBuffers[HC_HASH_GRID_ENTRIES_BUFFER] = mpDevice->createStructuredBuffer(sizeof(uint64_t), mHCParams.hashMapSize);
        // 128 bits per entry
        if (!mBuffers[HC_VOXEL_DATA_BUFFER_0]) mBuffers[HC_VOXEL_DATA_BUFFER_0] = mpDevice->createBuffer(16 * mHCParams.hashMapSize);
        if (!mBuffers[HC_VOXEL_DATA_BUFFER_1]) mBuffers[HC_VOXEL_DATA_BUFFER_1] = mpDevice->createBuffer(16 * mHCParams.hashMapSize);
    }
    if (mNNParams.active)
    {
        if (!mBuffers[NN_PRIMAL_BUFFER]) mBuffers[NN_PRIMAL_BUFFER] = mpDevice->createBuffer(mNNParams.nnParamCount * sizeof(float16_t));
        if (!mBuffers[NN_FILTERED_PRIMAL_BUFFER])
            mBuffers[NN_FILTERED_PRIMAL_BUFFER] = mpDevice->createBuffer(
                mNNParams.nnParamCount * sizeof(float16_t), ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess | ResourceBindFlags::Shared);
        if (!mBuffers[NN_GRADIENT_BUFFER]) mBuffers[NN_GRADIENT_BUFFER] = mpDevice->createBuffer(mNNParams.nnParamCount * sizeof(float));
        if (!mBuffers[NN_GRADIENT_COUNT_BUFFER]) mBuffers[NN_GRADIENT_COUNT_BUFFER] = mpDevice->createBuffer(mNNParams.nnParamCount * sizeof(float));
        mNNParams.gradientAuxElements = mNNParams.nnParamCount * 4;
        if (!mBuffers[NN_GRADIENT_AUX_BUFFER]) mBuffers[NN_GRADIENT_AUX_BUFFER] = mpDevice->createBuffer(mNNParams.gradientAuxElements * sizeof(float));
        if (mNNParams.featureHashMapProbingSize > 0 && !mBuffers[FEATURE_HASH_GRID_ENTRIES_BUFFER]) mBuffers[FEATURE_HASH_GRID_ENTRIES_BUFFER] = mpDevice->createStructuredBuffer(sizeof(uint64_t), mNNParams.featureHashMapSize / mNNParams.featureHashMapPlacesPerElement);
    }
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
    if (mHCParams.active || mNNParams.active)
    {
        auto var = mPasses[TRAIN_NN_FILL_CACHE_PASS]->getRootVar();
        var["CB"]["gFrameDim"] = frameDim;
        var["CB"]["gFrameCount"] = mFrameCount;
        var["CB"]["gCamPos"] = mCamPos;
        var["CB"]["gWeightsAddress"] = mBuffers[NN_PRIMAL_BUFFER]->getGpuAddress();
        mpScene->bindShaderData(var["gScene"]);
        mpSampleGenerator->bindShaderData(var);
        if (mpEnvMapSampler) mpEnvMapSampler->bindShaderData(mpSamplerBlock->getRootVar()["envMapSampler"]);
        if (mpEmissiveSampler) mpEmissiveSampler->bindShaderData(mpSamplerBlock->getRootVar()["emissiveSampler"]);
        var["gSampler"] = mpSamplerBlock;
        if (mHCParams.active)
        {
            var["gHCHashGridEntriesBuffer"] = mBuffers[HC_HASH_GRID_ENTRIES_BUFFER];
            var["gHCVoxelDataBuffer"] = mFrameCount % 2 == 0 ? mBuffers[HC_VOXEL_DATA_BUFFER_0] : mBuffers[HC_VOXEL_DATA_BUFFER_1];
            var["gHCVoxelDataBufferPrev"] = mFrameCount % 2 == 1 ? mBuffers[HC_VOXEL_DATA_BUFFER_0] : mBuffers[HC_VOXEL_DATA_BUFFER_1];
        }
        if (mNNParams.active)
        {
            var["PrimalBuffer"] = mBuffers[NN_PRIMAL_BUFFER];
            var["GradientBuffer"] = mBuffers[NN_GRADIENT_BUFFER];
            var["GradientCountBuffer"] = mBuffers[NN_GRADIENT_COUNT_BUFFER];
            if (mNNParams.featureHashMapProbingSize > 0 && mBuffers[FEATURE_HASH_GRID_ENTRIES_BUFFER]) var["gFeatureHashGridEntriesBuffer"] = mBuffers[FEATURE_HASH_GRID_ENTRIES_BUFFER];
        }
        var[kInputVBuffer.texname] = renderData.getTexture(kInputVBuffer.name);
        var[kInputViewDir.texname] = renderData.getTexture(kInputViewDir.name);
        mpPixelDebug->prepareProgram(mPasses[TRAIN_NN_FILL_CACHE_PASS]->getProgram(), var);
    }
    if (mHCParams.active)
    {
        auto var = mPasses[HC_RESOLVE_PASS]->getRootVar();
        var["gHCHashGridEntriesBuffer"] = mBuffers[HC_HASH_GRID_ENTRIES_BUFFER];
        var["gHCVoxelDataBuffer"] = mFrameCount % 2 == 0 ? mBuffers[HC_VOXEL_DATA_BUFFER_0] : mBuffers[HC_VOXEL_DATA_BUFFER_1];
        var["gHCVoxelDataBufferPrev"] = mFrameCount % 2 == 1 ? mBuffers[HC_VOXEL_DATA_BUFFER_0] : mBuffers[HC_VOXEL_DATA_BUFFER_1];
        mpPixelDebug->prepareProgram(mPasses[HC_RESOLVE_PASS]->getProgram(), var);
    }
    if (mHCParams.active && mHCParams.reset)
    {
        auto var = mPasses[HC_RESET_PASS]->getRootVar();
        var["gHCHashGridEntriesBuffer"] = mBuffers[HC_HASH_GRID_ENTRIES_BUFFER];
        var["gHCVoxelDataBuffer"] = mBuffers[HC_VOXEL_DATA_BUFFER_0];
        var["gHCVoxelDataBufferPrev"] = mBuffers[HC_VOXEL_DATA_BUFFER_1];
        mpPixelDebug->prepareProgram(mPasses[HC_RESET_PASS]->getProgram(), var);
    }
    {
        auto var = mPasses[PATH_TRACING_PASS]->getRootVar();
        var["CB"]["gFrameDim"] = frameDim;
        var["CB"]["gFrameCount"] = mFrameCount;
        var["CB"]["gCamPos"] = mCamPos;
        var["CB"]["gHashEncDebugLevel"] = mNNParams.featureHashMapDebugShowLevel;
        uint64_t address = mBuffers[NN_FILTERED_PRIMAL_BUFFER]->getGpuAddress();
        var["CB"]["gWeightsAddress"] = address;

        mpScene->bindShaderData(var["gScene"]);
        mpSampleGenerator->bindShaderData(var);
        if (mpEnvMapSampler) mpEnvMapSampler->bindShaderData(mpSamplerBlock->getRootVar()["envMapSampler"]);
        if (mpEmissiveSampler) mpEmissiveSampler->bindShaderData(mpSamplerBlock->getRootVar()["emissiveSampler"]);
        var["gSampler"] = mpSamplerBlock;
        if (mHCParams.active)
        {
            var["gHCHashGridEntriesBuffer"] = mBuffers[HC_HASH_GRID_ENTRIES_BUFFER];
            var["gHCVoxelDataBuffer"] = mFrameCount % 2 == 0 ? mBuffers[HC_VOXEL_DATA_BUFFER_0] : mBuffers[HC_VOXEL_DATA_BUFFER_1];
            var["gHCVoxelDataBufferPrev"] = mFrameCount % 2 == 1 ? mBuffers[HC_VOXEL_DATA_BUFFER_0] : mBuffers[HC_VOXEL_DATA_BUFFER_1];
        }
        if (mNNParams.active)
        {

            var["PrimalBuffer"] = mBuffers[NN_FILTERED_PRIMAL_BUFFER];
            var["GradientBuffer"] = mBuffers[NN_GRADIENT_BUFFER];
            var["GradientCountBuffer"] = mBuffers[NN_GRADIENT_COUNT_BUFFER];
            if (mNNParams.featureHashMapProbingSize > 0 && mBuffers[FEATURE_HASH_GRID_ENTRIES_BUFFER]) var["gFeatureHashGridEntriesBuffer"] = mBuffers[FEATURE_HASH_GRID_ENTRIES_BUFFER];
        }
        for (auto channel : kInputChannels) var[channel.texname] = renderData.getTexture(channel.name);
        var[kOutputColor.texname] = renderData.getTexture(kOutputColor.name);
        mpPixelDebug->prepareProgram(mPasses[PATH_TRACING_PASS]->getProgram(), var);
    }
    if (mNNParams.active)
    {
        auto var = mPasses[NN_GRADIENT_CLEAR_PASS]->getRootVar();
        var["GradientBuffer"] = mBuffers[NN_GRADIENT_BUFFER];
        var["GradientCountBuffer"] = mBuffers[NN_GRADIENT_COUNT_BUFFER];
        mpPixelDebug->prepareProgram(mPasses[NN_GRADIENT_CLEAR_PASS]->getProgram(), var);
    }
    if (mNNParams.active)
    {
        auto var = mPasses[NN_GRADIENT_DESCENT_PASS]->getRootVar();
        var["CB"]["t"] = mNNParams.optimizerParams.step_count;
        var["CB"]["lr"] = mNNParams.optimizerParams.learn_r;
        var["CB"]["filter_alpha"] = mNNParams.filterAlpha;
        var["PrimalBuffer"] = mBuffers[NN_PRIMAL_BUFFER];
        var["FilteredPrimalBuffer"] = mBuffers[NN_FILTERED_PRIMAL_BUFFER];
        var["GradientBuffer"] = mBuffers[NN_GRADIENT_BUFFER];
        var["GradientCountBuffer"] = mBuffers[NN_GRADIENT_COUNT_BUFFER];
        var["GradientAuxBuffer"] = mBuffers[NN_GRADIENT_AUX_BUFFER];
        mpPixelDebug->prepareProgram(mPasses[NN_GRADIENT_DESCENT_PASS]->getProgram(), var);
    }
    if (mNNParams.active && mNNParams.reset)
    {
        auto var = mPasses[NN_RESET_PASS]->getRootVar();
        mpSampleGenerator->bindShaderData(var);
        var["PrimalBuffer"] = mBuffers[NN_PRIMAL_BUFFER];
        var["FilteredPrimalBuffer"] = mBuffers[NN_FILTERED_PRIMAL_BUFFER];
        var["GradientAuxBuffer"] = mBuffers[NN_GRADIENT_AUX_BUFFER];
        if (mNNParams.featureHashMapProbingSize > 0 && mBuffers[FEATURE_HASH_GRID_ENTRIES_BUFFER]) var["gFeatureHashGridEntriesBuffer"] = mBuffers[FEATURE_HASH_GRID_ENTRIES_BUFFER];
        mpPixelDebug->prepareProgram(mPasses[NN_RESET_PASS]->getProgram(), var);
    }
    if (mPasses[IR_DEBUG_PASS])
    {
        auto var = mPasses[IR_DEBUG_PASS]->getRootVar();
        var["CB"]["gCamPos"] = mCamPos;
        var["CB"]["gDebugPixel"] = mpPixelDebug->getSelectedPixel();
        var["CB"]["gFrameCount"] = mFrameCount;
        var["CB"]["gMLPIndex"] = mIRDebugPassParams.nircMLPIndex;
        var["CB"]["gShowTransmission"] = mIRDebugPassParams.showTransmission;
        var["CB"]["gApplyBSDF"] = mIRDebugPassParams.applyBSDF;
        var["CB"]["gAccumulate"] = mIRDebugPassParams.accumulate;
        uint64_t address = mBuffers[NN_FILTERED_PRIMAL_BUFFER]->getGpuAddress();
        var["CB"]["gWeightsAddress"] = address;
        if (mHCParams.active)
        {
            var["gHCHashGridEntriesBuffer"] = mBuffers[HC_HASH_GRID_ENTRIES_BUFFER];
            var["gHCVoxelDataBuffer"] = mFrameCount % 2 == 0 ? mBuffers[HC_VOXEL_DATA_BUFFER_0] : mBuffers[HC_VOXEL_DATA_BUFFER_1];
            var["gHCVoxelDataBufferPrev"] = mFrameCount % 2 == 1 ? mBuffers[HC_VOXEL_DATA_BUFFER_0] : mBuffers[HC_VOXEL_DATA_BUFFER_1];
        }
        if (mpEnvMapSampler) mpEnvMapSampler->bindShaderData(mpSamplerBlock->getRootVar()["envMapSampler"]);
        if (mpEmissiveSampler) mpEmissiveSampler->bindShaderData(mpSamplerBlock->getRootVar()["emissiveSampler"]);
        var["gSampler"] = mpSamplerBlock;
        mpScene->bindShaderData(var["gScene"]);
        mpSampleGenerator->bindShaderData(var);
        var["PrimalBuffer"] = mBuffers[NN_FILTERED_PRIMAL_BUFFER];
        if (mNNParams.featureHashMapProbingSize > 0 && mBuffers[FEATURE_HASH_GRID_ENTRIES_BUFFER]) var["gFeatureHashGridEntriesBuffer"] = mBuffers[FEATURE_HASH_GRID_ENTRIES_BUFFER];
        var[kInputVBuffer.texname] = renderData.getTexture(kInputVBuffer.name);
        var[kInputViewDir.texname] = renderData.getTexture(kInputViewDir.name);
        var[kIRDebugOutputColor.texname] = renderData.getTexture(kIRDebugOutputColor.name);
        var[kIRDebugOutputColorRef.texname] = renderData.getTexture(kIRDebugOutputColorRef.name);
        mpPixelDebug->prepareProgram(mPasses[IR_DEBUG_PASS]->getProgram(), var);
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
        //reset();
        renderData.getDictionary()[Falcor::kRenderPassRefreshFlags] = Falcor::RenderPassRefreshFlags::RenderOptionsChanged;
        // activate hc if it is used somewhere
        mHCParams.active = mRRParams.requiresHC() | mHCParams.injectRadianceSpread | mHCParams.debugColor | mHCParams.debugLevels | mHCParams.debugVoxels | (mIRDebugPassParams.irMethod == IRDebugPassParam::SHOW_IRHC && mIRDebugPassParams.active);
        // activate nn if it is used somewhere
        mNNParams.active = mRRParams.requiresNN() | mNNParams.debugOutput | (mIRDebugPassParams.irMethod == IRDebugPassParam::SHOW_NIRC && mIRDebugPassParams.active) | mNNParams.injectRadianceSpread;
        mNNParams.keepThreads = mNNParams.active;
        // only allow activation of ir debug pass if either nn or hc is using incident radiance
        mIRDebugPassParams.active &= ((mNNParams.nnMethod == NNParams::USE_NIRC && mIRDebugPassParams.irMethod == IRDebugPassParam::SHOW_NIRC)
            | (mHCParams.hcMethod == HCParams::USE_IRHC && mIRDebugPassParams.irMethod == IRDebugPassParam::SHOW_IRHC));
        setupData(pRenderContext);
        createPasses(renderData);
        setupBuffers();
        mOptionsChanged = false;
    }
    bindData(renderData, frameDim);

    const uint2 targetDim = renderData.getDefaultTextureDims();
    FALCOR_ASSERT(targetDim.x > 0 && targetDim.y > 0);
    if (mHCParams.active && mHCParams.reset)
    {
        mHCParams.reset = false;
        mPasses[HC_RESET_PASS]->execute(pRenderContext, mHCParams.hashMapSize, 1);
    }
    if (mNNParams.active && mNNParams.reset)
    {
        mNNParams.reset = false;
        mPasses[NN_RESET_PASS]->execute(pRenderContext, std::max(mNNParams.gradientAuxElements, mNNParams.nnParamCount), 1);
    }
    {
        FALCOR_PROFILE(pRenderContext, "ComputePathTracer::training");
        for (uint32_t i = 0; i < 4; i++)
        {
            if (mNNParams.active) mPasses[NN_GRADIENT_CLEAR_PASS]->execute(pRenderContext, mNNParams.nnParamCount, 1);
            if (mHCParams.active || mNNParams.active)
            {
                mPasses[TRAIN_NN_FILL_CACHE_PASS]->getRootVar()["CB"]["gTrainIteration"] = i;
                mPasses[TRAIN_NN_FILL_CACHE_PASS]->execute(pRenderContext, frameDim.x / 10, frameDim.y / 10);
            }
            if (mNNParams.active && mNNParams.train) mPasses[NN_GRADIENT_DESCENT_PASS]->execute(pRenderContext, mNNParams.nnParamCount, 1);
        }
        if (mHCParams.active) mPasses[HC_RESOLVE_PASS]->execute(pRenderContext, mHCParams.hashMapSize, 1);
    }
    {
        FALCOR_PROFILE(pRenderContext, "ComputePathTracer::pt");
        mPasses[PATH_TRACING_PASS]->execute(pRenderContext, frameDim.x, frameDim.y);
    }
    if (mPasses[IR_DEBUG_PASS])
    {
        FALCOR_PROFILE(pRenderContext, "ComputePathTracer::ir_debug");
        mPasses[IR_DEBUG_PASS]->execute(pRenderContext, kIRDebugOutputDim.x, kIRDebugOutputDim.y);
    }
    mpPixelDebug->endFrame(pRenderContext);
    mFrameCount++;
    mNNParams.optimizerParams.step_count++;
}

void ComputePathTracer::renderUI(Gui::Widgets& widget)
{
    ImGui::PushItemWidth(40);
    ImGui::Text("Bounce limits");
    ImGui::SameLine();
    ImGui::InputScalar("min", ImGuiDataType_U32, &mLowerBounceCount);
    ImGui::SameLine();
    ImGui::InputScalar("max", ImGuiDataType_U32, &mUpperBounceCount);
    ImGui::PopItemWidth();
    widget.tooltip("Inclusive range of bounces that contribute to final image color", true);

    widget.checkbox("BSDF importance sampling", mUseImportanceSampling);
    widget.tooltip("Use importance sampling for materials", true);

    widget.checkbox("NEE", mUseNEE);
    widget.checkbox("MIS", mUseMIS);
    widget.checkbox("power heuristic", mMISUsePowerHeuristic, true);
    widget.tooltip("Active: power heuristic; Inactive: balance heuristic", true);
    if (Gui::Group rr_group = widget.group("RR"))
    {
        rr_group.checkbox("enable", mRRParams.active);
        ImGui::PushItemWidth(120);
        rr_group.dropdown("survival prob", mRRParams.survivalProbOptionList, mRRParams.survivalProbOption);
        rr_group.tooltip("Determine the survival probability using one of the option.\ndefault: use a constantly shrinking probability based on the parameters\nexpected thp: based on the expected contribution to come\nadrrs: based on the weight window method from adrrs", true);
        mRRParams.update();
        ImGui::PopItemWidth();
        if (mRRParams.requiresReductionParams())
        {
            ImGui::PushItemWidth(80);
            ImGui::InputFloat("RR start value", &mRRParams.probStartValue);
            ImGui::PopItemWidth();
            rr_group.tooltip("Starting value of the survival probability", true);
            ImGui::PushItemWidth(80);
            ImGui::InputFloat("RR reduction factor", &mRRParams.probReductionFactor);
            ImGui::PopItemWidth();
            rr_group.tooltip("Gets multiplied to the initial survival probability at each interaction", true);
        }
        else if (mRRParams.requiresPCE())
        {
            ImGui::PushItemWidth(120);
            rr_group.dropdown("contrib estimation", mRRParams.pathContribEstimateOptionList, mRRParams.pathContribEstimateOption);
            rr_group.tooltip("Estimate the expected radiance to come at a vertex on a path.\nhc: use estimate from hc\nnn: use estimate from nn", true);
            rr_group.checkbox("inject local radiance estimate to rr", mRRParams.injectRadiance);
            ImGui::PopItemWidth();
        }
        if (mRRParams.requiresPME())
        {
            ImGui::PushItemWidth(120);
            rr_group.dropdown("pixel measurement estimation", mRRParams.pixelMeasurementEstimateOptionList, mRRParams.pixelMeasurementEstimateOption);
            rr_group.tooltip("Estimate the total measurement of a pixel for adrrs.\nhc: use estimate from hc\nnn: use estimate from nn", true);
            ImGui::PopItemWidth();
        }
    }
    if (Gui::Group emissive_sampler_group = widget.group("EmissiveSampler"))
    {
        if (mpEmissiveSampler) mpEmissiveSampler->renderUI(emissive_sampler_group);
    }
    // radiance hash cache
    if (Gui::Group hc_group = widget.group("Radiance Hash Cache"))
    {
        hc_group.text(std::string("active: ") + (mHCParams.active ? "true" : "false"));
        hc_group.dropdown("HC method", mHCParams.hcMethodList, mHCParams.hcMethod);
        ImGui::PushItemWidth(40);
        ImGui::InputScalar("hashMapSizeExponent", ImGuiDataType_U32, &mHCParams.hashMapSizeExp);
        ImGui::PopItemWidth();
        hc_group.tooltip("Use the radiance estimate from the hc instead of the rr weights.", true);
        hc_group.checkbox("inject radiance to spread", mHCParams.injectRadianceSpread);
        hc_group.tooltip("Terminate the path as soon as the accumulated roughness blurred the inaccuracies of the hc away. Then, query the hc for a radiance estimate.", true);
        hc_group.checkbox("debug voxels", mHCParams.debugVoxels);
        hc_group.checkbox("debug color", mHCParams.debugColor);
        hc_group.checkbox("debug levels", mHCParams.debugLevels);
        mHCParams.reset |= widget.button("Reset hc");
    }
    // neural network
    if (Gui::Group nn_group = widget.group("NN"))
    {
        nn_group.text(std::string("active: ") + (mNNParams.active ? "true" : "false"));
        nn_group.checkbox("train", mNNParams.train);
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
            }
            ImGui::PopItemWidth();
        }
        ImGui::PushItemWidth(120);
        nn_group.dropdown("NN layer width", mNNParams.nnLayerWidthList, mNNParams.nnLayerWidth);
        nn_group.dropdown("NN method", mNNParams.nnMethodList, mNNParams.nnMethod);
        if (mNNParams.nnMethod == NNParams::USE_NIRC) mNNParams.mlpCount = 1;
        else if (mNNParams.nnMethod == NNParams::USE_NRC) mNNParams.mlpCount = 1;
        if (mNNParams.nnLayerCount.size() != mNNParams.mlpCount) mNNParams.nnLayerCount.resize(mNNParams.mlpCount, 1);
        for (uint i = 0; i < mNNParams.nnLayerCount.size(); i++) ImGui::InputInt(std::string(std::string("MLP ") + std::to_string(i) + std::string(" layer count")).c_str(), &mNNParams.nnLayerCount[i]);
        nn_group.dropdown("enc method", mNNParams.encMethodList, mNNParams.encMethod);
        ImGui::InputFloat("Filter alpha", &mNNParams.filterAlpha, 0.0f, 0.0f, "%.4f");
        nn_group.checkbox("inject radiance to spread", mNNParams.injectRadianceSpread);
        nn_group.checkbox("debug NN output", mNNParams.debugOutput);
        ImGui::Text("Weight init bounds");
        ImGui::InputFloat("min", &mNNParams.weightInitBound.x, 0.0f, 0.0f, "%.6f");
        ImGui::InputFloat("max", &mNNParams.weightInitBound.y, 0.0f, 0.0f, "%.6f");
        ImGui::InputInt("training bounces", &mNNParams.trainingBounces);
        ImGui::Separator();
        ImGui::Text("input encoding");
        nn_group.checkbox("hash enc separate level grids", mNNParams.featureHashEncSeparateLevelGrids);
        nn_group.checkbox("hash enc use multi level dir", mNNParams.featureHashEncUseMultiLevelDir);
        ImGui::InputInt("hash enc debug show level", &mNNParams.featureHashMapDebugShowLevel);
        ImGui::InputInt("hash enc probing size", &mNNParams.featureHashMapProbingSize);
        nn_group.tooltip("The number of slots that are tested when the current slot is occupied.", true);
        ImGui::PopItemWidth();
        ImGui::Separator();
        mNNParams.reset |= widget.button("Reset nn");
        ImGui::Separator();
    }
    if (Gui::Group debug_group = widget.group("Debug"))
    {
        ImGui::Separator();
        ImGui::Text("IR debug pass");
        debug_group.checkbox("enable", mIRDebugPassParams.active);
        debug_group.dropdown("IR method", mIRDebugPassParams.irMethodList, mIRDebugPassParams.irMethod);
        ImGui::InputInt("mlp index", &mIRDebugPassParams.nircMLPIndex);
        debug_group.checkbox("show transmission", mIRDebugPassParams.showTransmission);
        debug_group.checkbox("apply bsdf", mIRDebugPassParams.applyBSDF);
        debug_group.checkbox("accumulate", mIRDebugPassParams.accumulate);
        ImGui::Separator();
        debug_group.checkbox("path length", mDebugPathLength);
        mpPixelDebug->renderUI(debug_group);
    }

    // reload shader and set options change flag (explicitly apply changes)
    // In execute() we will pass the flag to other passes for reset of temporal data etc.
    if (widget.button("Reload shader"))
    {
        mOptionsChanged = true;
        //mPasses[PATH_TRACING_PASS] = nullptr;
        reset();
    }
}

void ComputePathTracer::setScene(RenderContext* pRenderContext, const ref<Scene>& pScene)
{
    mpScene = pScene;
    mCamPos = mpScene->getCamera()->getPosition();
    mpScene->toggleAnimations(false);
    reset();
}


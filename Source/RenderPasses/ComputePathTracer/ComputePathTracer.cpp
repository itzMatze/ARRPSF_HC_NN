#include "ComputePathTracer.h"
#include "Core/API/Formats.h"
#include "RenderGraph/RenderPassHelpers.h"
#include "RenderGraph/RenderPassStandardFlags.h"
#include "imgui.h"

#include <random>
#include <string>

namespace
{
const std::string kPTShaderFile("RenderPasses/ComputePathTracer/ComputePathTracer.slang");
const std::string kPTTrainShaderFile("RenderPasses/ComputePathTracer/ComputePathTracerTrain.slang");
const std::string kRHCResolveShaderFile("RenderPasses/ComputePathTracer/RadianceHashCacheResolve.slang");
const std::string kRHCResetShaderFile("RenderPasses/ComputePathTracer/RadianceHashCacheReset.slang");
const std::string kGradientClearShaderFile("RenderPasses/ComputePathTracer/tinynn/GradientClear.slang");
const std::string kGradientDescentShaderFile("RenderPasses/ComputePathTracer/tinynn/GradientDescentPrimal.slang");
const std::string kNNResetShaderFile("RenderPasses/ComputePathTracer/tinynn/NNReset.slang");
const std::string kNIRCDebugVisShaderFile("RenderPasses/ComputePathTracer/NIRCDebugVis.slang");

// inputs
const ChannelDesc kInputVBuffer{"vbuffer", "gVBuffer", "Visibility buffer in packed format"};
const ChannelDesc kInputViewDir{"viewW", "gViewW", "World-space view direction (xyz float format)"};
const ChannelDesc kInputRefImage{"refImage", "gRefImage", "Reference image for the current scene. Used for debugging."};
const ChannelList kInputChannels = {kInputVBuffer, kInputViewDir, kInputRefImage};

// outputs
const ChannelDesc kOutputColor = { "color", "gOutputColor", "Output color (sum of direct and indirect)", false, ResourceFormat::RGBA32Float };
constexpr uint2 kNIRCDebugOutputDim(1000, 1000);
const ChannelDesc kNIRCDebugOutputColor = { "nirc_debug", "gNIRCDebugOutputColor", "Output color of NIRC debug visualization", false, ResourceFormat::RGBA32Float };
const ChannelDesc kNIRCDebugOutputColorRef = { "nirc_debug_ref", "gNIRCDebugOutputColorRef", "Output color of the path traced NIRC debug visualization for reference", false, ResourceFormat::RGBA32Float };
const ChannelList kOutputChannels = {kOutputColor, kNIRCDebugOutputColor};

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
const std::string kRHCHashMapSizeExponent = "RHCHashMapSizeExponent";
const std::string kRHCInjectRadianceRR = "RHCInjectRadianceRR";
const std::string kRHCInjectRadianceSpread = "RHCInjectRadianceSpread";
const std::string kRHCDebugColor = "RHCDebugColor";
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
        else if (key == kRHCHashMapSizeExponent)
        {
            mRHCParams.hashMapSizeExp = uint32_t(value);
            mRHCParams.hashMapSize = std::pow(2u, mRHCParams.hashMapSizeExp);
        }
        else if (key == kRHCInjectRadianceRR) mRHCParams.injectRadianceRR = value;
        else if (key == kRHCInjectRadianceSpread) mRHCParams.injectRadianceSpread = value;
        else if (key == kRHCDebugColor) mRHCParams.debugColor = value;
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
    props[kUseRR] = mRRParams.active;
    props[kRRProbStartValue] = mRRParams.probStartValue;
    props[kRRProbReductionFactor] = mRRParams.probReductionFactor;
    props[kRHCHashMapSizeExponent] = mRHCParams.hashMapSizeExp;
    props[kRHCInjectRadianceRR] = mRHCParams.injectRadianceRR;
    props[kRHCInjectRadianceSpread] = mRHCParams.injectRadianceSpread;
    props[kRHCDebugColor] = mRHCParams.debugColor;
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
    addRenderPassOutputs(reflector, {kNIRCDebugOutputColor}, ResourceBindFlags::UnorderedAccess, kNIRCDebugOutputDim);
    addRenderPassOutputs(reflector, {kNIRCDebugOutputColorRef}, ResourceBindFlags::UnorderedAccess, kNIRCDebugOutputDim);
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
    defineList["R_HC_DEBUG_VOXELS"] = mRHCParams.debugVoxels ? "1" : "0";
    defineList["R_HC_DEBUG_COLOR"] = mRHCParams.debugColor ? "1" : "0";
    defineList["R_HC_DEBUG_LEVELS"] = mRHCParams.debugLevels ? "1" : "0";
    defineList["R_HC_HASHMAP_SIZE"] = std::to_string(mRHCParams.hashMapSize);
    defineList["USE_IMPORTANCE_SAMPLING"] = mUseImportanceSampling ? "1" : "0";
    defineList["USE_ANALYTIC_LIGHTS"] = mpScene->useAnalyticLights() ? "1" : "0";
    defineList["USE_EMISSIVE_LIGHTS"] = mpScene->useEmissiveLights() ? "1" : "0";
    defineList["USE_ENV_LIGHT"] = mpScene->useEnvLight() ? "1" : "0";
    defineList["USE_ENV_BACKGROUND"] = mpScene->useEnvBackground() ? "1" : "0";
    defineList["NN_DEBUG"] = mNNParams.debugOutput ? "1" : "0";
    defineList["NIRC_DEBUG_OUTPUT_WIDTH"] = std::to_string(kNIRCDebugOutputDim.x);
    defineList["NIRC_DEBUG_OUTPUT_HEIGHT"] = std::to_string(kNIRCDebugOutputDim.y);
    defineList["NN_PARAM_COUNT"] = std::to_string(mNNParams.nnParamCount);
    defineList["NN_WEIGHT_INIT_LOWER_BOUND"] = fmt::format("{:.12f}", mNNParams.weightInitBound.x);
    defineList["NN_WEIGHT_INIT_UPPER_BOUND"] = fmt::format("{:.12f}", mNNParams.weightInitBound.y);
    defineList["NN_GRAD_OFFSET"] = std::to_string(mNNParams.gradOffset);
    defineList["NN_GRADIENT_AUX_ELEMENTS"] = std::to_string(mNNParams.gradientAuxElements);
    defineList["NN_OPTIMIZER_TYPE"] = std::to_string(mNNParams.optimizerParams.type);
    defineList["NN_LEARNING_RATE"] = fmt::format("{:.12f}", mNNParams.optimizerParams.learn_r);
    defineList["NN_PARAM_0"] = fmt::format("{:.12f}", mNNParams.optimizerParams.param_0);
    defineList["NN_PARAM_1"] = fmt::format("{:.12f}", mNNParams.optimizerParams.param_1);
    defineList["NN_PARAM_2"] = fmt::format("{:.12f}", mNNParams.optimizerParams.param_2);
    defineList["NN_LAYER_WIDTH"] = std::to_string(mNNParams.nnLayerWidth);
    defineList["MLP_COUNT"] = std::to_string(mNNParams.nnLayerCount.size());
    for (uint i = 0; i < mNNParams.nnLayerCount.size(); i++) defineList[std::string("NN_LAYER_COUNT") + std::to_string(i)] = std::to_string(mNNParams.nnLayerCount[i]);
    defineList["NN_TRAINING_BOUNCES"] = std::to_string(mNNParams.trainingBounces);
    defineList["FEATURE_HASH_GRID_SIZE"] = std::to_string(mNNParams.featureHashMapSize);
    defineList["FEATURE_HASH_GRID_PLACES_PER_ELEMENT"] = std::to_string(mNNParams.featureHashMapPlacesPerElement);
    defineList["FEATURE_HASH_GRID_PROBING_SIZE"] = std::to_string(mNNParams.featureHashMapProbingSize);

    if (!mPasses[TRAIN_NN_FILL_CACHE_PASS] && (mRHCParams.active || mNNParams.active))
    {
        defineList["R_HC_UPDATE"] = mRHCParams.active ? "1" : "0";
        defineList["R_HC_QUERY"] = "0";
        defineList["NN_TRAIN"] = mNNParams.active ? "1" : "0";
        defineList["NN_QUERY"] = "0";
        // use default rr for training
        defineList["RR_OPTION_BITS"] = "0";
        defineList["R_HC_INJECT_RADIANCE_RR"] = "0";
        defineList["R_HC_INJECT_RADIANCE_SPREAD"] = "0";
        ProgramDesc desc;
        desc.addShaderModules(mpScene->getShaderModules());
        desc.addShaderLibrary(kPTTrainShaderFile).csEntry("main");
        desc.addTypeConformances(mpScene->getTypeConformances());
        mPasses[TRAIN_NN_FILL_CACHE_PASS] = ComputePass::create(mpDevice, desc, defineList, true);
    }
    if (!mPasses[PATH_TRACING_PASS])
    {
        defineList["R_HC_UPDATE"] = "0";
        defineList["R_HC_QUERY"] = mRHCParams.active ? "1" : "0";
        defineList["NN_TRAIN"] = "0";
        defineList["NN_QUERY"] = mNNParams.active ? "1" : "0";
        defineList["RR_OPTION_BITS"] = std::to_string(mRRParams.getOptionBits());
        // when using the nn during pt the threads need to be kept running for the cooperative matrices
        defineList["KEEP_THREADS"] = mNNParams.keepThreads ? "1" : "0";
        defineList["R_HC_INJECT_RADIANCE_RR"] = mRHCParams.injectRadianceRR ? "1" : "0";
        defineList["R_HC_INJECT_RADIANCE_SPREAD"] = mRHCParams.injectRadianceSpread ? "1" : "0";
        ProgramDesc desc;
        desc.addShaderModules(mpScene->getShaderModules());
        desc.addShaderLibrary(kPTShaderFile).csEntry("main");
        desc.addTypeConformances(mpScene->getTypeConformances());
        mPasses[PATH_TRACING_PASS] = ComputePass::create(mpDevice, desc, defineList, true);
    }
    if (!mPasses[RHC_RESOLVE_PASS] && mRHCParams.active)
    {
        defineList["R_HC_UPDATE"] = "1";
        defineList["R_HC_QUERY"] = "1";
        ProgramDesc desc;
        desc.addShaderLibrary(kRHCResolveShaderFile).csEntry("hashCacheResolve");
        mPasses[RHC_RESOLVE_PASS] = ComputePass::create(mpDevice, desc, defineList, true);
    }
    if (!mPasses[RHC_RESET_PASS] && mRHCParams.active)
    {
        defineList["R_HC_UPDATE"] = "1";
        defineList["R_HC_QUERY"] = "1";
        ProgramDesc desc;
        desc.addShaderLibrary(kRHCResetShaderFile).csEntry("main");
        mPasses[RHC_RESET_PASS] = ComputePass::create(mpDevice, desc, defineList, true);
    }
    if (!mPasses[NN_GRADIENT_CLEAR_PASS] && mNNParams.active)
    {
        ProgramDesc desc;
        desc.addShaderLibrary(kGradientClearShaderFile).csEntry("main");
        mPasses[NN_GRADIENT_CLEAR_PASS] = ComputePass::create(mpDevice, desc, defineList, true);
    }
    if (!mPasses[NN_GRADIENT_DESCENT_PASS] && mNNParams.active)
    {
        defineList["NN_FILTER_ALPHA"] = fmt::format("{:.12f}", mNNParams.filterAlpha);
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
    if (!mPasses[NIRC_DEBUG_PASS] && mNNParams.active)
    {
        ProgramDesc desc;
        desc.addShaderModules(mpScene->getShaderModules());
        desc.addShaderLibrary(kNIRCDebugVisShaderFile).csEntry("main");
        desc.addTypeConformances(mpScene->getTypeConformances());
        mPasses[NIRC_DEBUG_PASS] = ComputePass::create(mpDevice, desc, defineList, true);
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
    if (mRHCParams.active)
    {
        if (!mBuffers[R_HC_HASH_GRID_ENTRIES_BUFFER]) mBuffers[R_HC_HASH_GRID_ENTRIES_BUFFER] = mpDevice->createStructuredBuffer(sizeof(uint64_t), mRHCParams.hashMapSize);
        // 128 bits per entry
        if (!mBuffers[R_HC_VOXEL_DATA_BUFFER_0]) mBuffers[R_HC_VOXEL_DATA_BUFFER_0] = mpDevice->createBuffer(16 * mRHCParams.hashMapSize);
        if (!mBuffers[R_HC_VOXEL_DATA_BUFFER_1]) mBuffers[R_HC_VOXEL_DATA_BUFFER_1] = mpDevice->createBuffer(16 * mRHCParams.hashMapSize);
    }
    if (mNNParams.active)
    {
        if (!mBuffers[NN_PRIMAL_BUFFER]) mBuffers[NN_PRIMAL_BUFFER] = mpDevice->createBuffer(mNNParams.nnParamCount * sizeof(float));
        if (!mBuffers[NN_FILTERED_PRIMAL_BUFFER]) mBuffers[NN_FILTERED_PRIMAL_BUFFER] = mpDevice->createBuffer(mNNParams.nnParamCount * sizeof(float));
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
    if (mRHCParams.active || mNNParams.active)
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
        if (mRHCParams.active)
        {
            var["gRHCHashGridEntriesBuffer"] = mBuffers[R_HC_HASH_GRID_ENTRIES_BUFFER];
            var["gRHCVoxelDataBuffer"] = mFrameCount % 2 == 0 ? mBuffers[R_HC_VOXEL_DATA_BUFFER_0] : mBuffers[R_HC_VOXEL_DATA_BUFFER_1];
            var["gRHCVoxelDataBufferPrev"] = mFrameCount % 2 == 1 ? mBuffers[R_HC_VOXEL_DATA_BUFFER_0] : mBuffers[R_HC_VOXEL_DATA_BUFFER_1];
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
    if (mRHCParams.active)
    {
        auto var = mPasses[RHC_RESOLVE_PASS]->getRootVar();
        var["gRHCHashGridEntriesBuffer"] = mBuffers[R_HC_HASH_GRID_ENTRIES_BUFFER];
        var["gRHCVoxelDataBuffer"] = mFrameCount % 2 == 0 ? mBuffers[R_HC_VOXEL_DATA_BUFFER_0] : mBuffers[R_HC_VOXEL_DATA_BUFFER_1];
        var["gRHCVoxelDataBufferPrev"] = mFrameCount % 2 == 1 ? mBuffers[R_HC_VOXEL_DATA_BUFFER_0] : mBuffers[R_HC_VOXEL_DATA_BUFFER_1];
        mpPixelDebug->prepareProgram(mPasses[RHC_RESOLVE_PASS]->getProgram(), var);
    }
    if (mRHCParams.active && mRHCParams.reset)
    {
        auto var = mPasses[RHC_RESET_PASS]->getRootVar();
        var["gRHCHashGridEntriesBuffer"] = mBuffers[R_HC_HASH_GRID_ENTRIES_BUFFER];
        var["gRHCVoxelDataBuffer"] = mBuffers[R_HC_VOXEL_DATA_BUFFER_0];
        var["gRHCVoxelDataBufferPrev"] = mBuffers[R_HC_VOXEL_DATA_BUFFER_1];
        mpPixelDebug->prepareProgram(mPasses[RHC_RESET_PASS]->getProgram(), var);
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
        if (mRHCParams.active)
        {
            var["gRHCHashGridEntriesBuffer"] = mBuffers[R_HC_HASH_GRID_ENTRIES_BUFFER];
            var["gRHCVoxelDataBuffer"] = mFrameCount % 2 == 0 ? mBuffers[R_HC_VOXEL_DATA_BUFFER_0] : mBuffers[R_HC_VOXEL_DATA_BUFFER_1];
            var["gRHCVoxelDataBufferPrev"] = mFrameCount % 2 == 1 ? mBuffers[R_HC_VOXEL_DATA_BUFFER_0] : mBuffers[R_HC_VOXEL_DATA_BUFFER_1];
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
    if (mNNParams.active && mNNParams.nircDebug)
    {
        auto var = mPasses[NIRC_DEBUG_PASS]->getRootVar();
        var["CB"]["gDebugPixel"] = mpPixelDebug->getSelectedPixel();
        var["CB"]["gFrameCount"] = mFrameCount;
        var["CB"]["gMLPIndex"] = mNNParams.nircMLPIndex;
        var["CB"]["gShowTransmission"] = mNNParams.nircDebugShowTransmission;
        var["CB"]["gApplyBSDF"] = mNNParams.nircDebugApplyBSDF;
        if (mpEnvMapSampler) mpEnvMapSampler->bindShaderData(mpSamplerBlock->getRootVar()["envMapSampler"]);
        if (mpEmissiveSampler) mpEmissiveSampler->bindShaderData(mpSamplerBlock->getRootVar()["emissiveSampler"]);
        var["gSampler"] = mpSamplerBlock;
        mpScene->bindShaderData(var["gScene"]);
        mpSampleGenerator->bindShaderData(var);
        var["PrimalBuffer"] = mBuffers[NN_FILTERED_PRIMAL_BUFFER];
        if (mNNParams.featureHashMapProbingSize > 0 && mBuffers[FEATURE_HASH_GRID_ENTRIES_BUFFER]) var["gFeatureHashGridEntriesBuffer"] = mBuffers[FEATURE_HASH_GRID_ENTRIES_BUFFER];
        var[kInputVBuffer.texname] = renderData.getTexture(kInputVBuffer.name);
        var[kInputViewDir.texname] = renderData.getTexture(kInputViewDir.name);
        var[kNIRCDebugOutputColor.texname] = renderData.getTexture(kNIRCDebugOutputColor.name);
        var[kNIRCDebugOutputColorRef.texname] = renderData.getTexture(kNIRCDebugOutputColorRef.name);
        mpPixelDebug->prepareProgram(mPasses[NIRC_DEBUG_PASS]->getProgram(), var);
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
        // activate rhc if it is used somewhere
        mRHCParams.active = mRRParams.requiresRHC() | mRHCParams.injectRadianceRR | mRHCParams.injectRadianceSpread | mRHCParams.debugColor | mRHCParams.debugLevels | mRHCParams.debugVoxels;
        // activate nn if it is used somewhere
        mNNParams.active = mRRParams.requiresNN() | mNNParams.debugOutput | mNNParams.nircDebug;
        mNNParams.keepThreads = mNNParams.active;
        setupData(pRenderContext);
        createPasses(renderData);
        setupBuffers();
        mOptionsChanged = false;
    }
    bindData(renderData, frameDim);

    const uint2 targetDim = renderData.getDefaultTextureDims();
    FALCOR_ASSERT(targetDim.x > 0 && targetDim.y > 0);
    if (mRHCParams.active && mRHCParams.reset)
    {
        mRHCParams.reset = false;
        mPasses[RHC_RESET_PASS]->execute(pRenderContext, mRHCParams.hashMapSize, 1);
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
            if (mRHCParams.active || mNNParams.active)
            {
                mPasses[TRAIN_NN_FILL_CACHE_PASS]->getRootVar()["CB"]["gTrainIteration"] = i;
                mPasses[TRAIN_NN_FILL_CACHE_PASS]->execute(pRenderContext, frameDim.x / 10, frameDim.y / 10);
            }
            if (mNNParams.active) mPasses[NN_GRADIENT_DESCENT_PASS]->execute(pRenderContext, mNNParams.nnParamCount, 1);
        }
        if (mRHCParams.active) mPasses[RHC_RESOLVE_PASS]->execute(pRenderContext, mRHCParams.hashMapSize, 1);
    }
    {
        FALCOR_PROFILE(pRenderContext, "ComputePathTracer::pt");
        mPasses[PATH_TRACING_PASS]->execute(pRenderContext, frameDim.x, frameDim.y);
    }
    if (mNNParams.active && mNNParams.nircDebug)
    {
        FALCOR_PROFILE(pRenderContext, "ComputePathTracer::nirc_debug");
        mPasses[NIRC_DEBUG_PASS]->execute(pRenderContext, kNIRCDebugOutputDim.x, kNIRCDebugOutputDim.y);
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
            rr_group.tooltip("Estimate the expected radiance to come at a vertex on a path.\nrhc: use estimate from rhc\nnn: use estimate from nn", true);
            ImGui::PopItemWidth();
        }
        if (mRRParams.requiresPME())
        {
            ImGui::PushItemWidth(120);
            rr_group.dropdown("pixel measurement estimation", mRRParams.pixelMeasurementEstimateOptionList, mRRParams.pixelMeasurementEstimateOption);
            rr_group.tooltip("Estimate the total measurement of a pixel for adrrs.\nrhc: use estimate from rhc\nnn: use estimate from nn", true);
            ImGui::PopItemWidth();
        }
        mRRParams.update();
    }
    if (Gui::Group emissive_sampler_group = widget.group("EmissiveSampler"))
    {
        if (mpEmissiveSampler) mpEmissiveSampler->renderUI(emissive_sampler_group);
    }
    // radiance hash cache
    if (Gui::Group rhc_group = widget.group("Radiance Hash Cache"))
    {
        rhc_group.text(std::string("active: ") + (mRHCParams.active ? "true" : "false"));
        ImGui::PushItemWidth(40);
        ImGui::InputScalar("hashMapSizeExponent", ImGuiDataType_U32, &mRHCParams.hashMapSizeExp);
        ImGui::PopItemWidth();
        rhc_group.checkbox("inject radiance to rr", mRHCParams.injectRadianceRR);
        rhc_group.tooltip("Use the radiance estimate from the rhc instead of the rr weights.", true);
        rhc_group.checkbox("inject radiance to spread", mRHCParams.injectRadianceSpread);
        rhc_group.tooltip("Terminate the path as soon as the accumulated roughness blurred the inaccuracies of the rhc away. Then, query the rhc for a radiance estimate.", true);
        rhc_group.checkbox("debug voxels", mRHCParams.debugVoxels);
        rhc_group.checkbox("debug color", mRHCParams.debugColor);
        rhc_group.checkbox("debug levels", mRHCParams.debugLevels);
        mRHCParams.reset |= widget.button("Reset rhc");
    }
    // neural network
    if (Gui::Group nn_group = widget.group("NN"))
    {
        nn_group.text(std::string("active: ") + (mNNParams.active ? "true" : "false"));
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
        ImGui::InputInt("MLP count", &mNNParams.mlpCount);
        mNNParams.mlpCount = std::min(mNNParams.mlpCount, 3);
        if (mNNParams.nnLayerCount.size() != mNNParams.mlpCount) mNNParams.nnLayerCount.resize(mNNParams.mlpCount);
        for (uint i = 0; i < mNNParams.nnLayerCount.size(); i++) ImGui::InputInt(std::string(std::string("MLP ") + std::to_string(i) + std::string(" layer count")).c_str(), &mNNParams.nnLayerCount[i]);
        ImGui::InputFloat("Filter alpha", &mNNParams.filterAlpha, 0.0f, 0.0f, "%.4f");
        nn_group.checkbox("debug NN output", mNNParams.debugOutput);
        nn_group.checkbox("debug NIRC", mNNParams.nircDebug);
        ImGui::InputInt("NIRC debug mlp index", &mNNParams.nircMLPIndex);
        nn_group.checkbox("NIRC debug show transmission", mNNParams.nircDebugShowTransmission);
        nn_group.checkbox("NIRC debug apply bsdf", mNNParams.nircDebugApplyBSDF);
        ImGui::Text("Weight init bounds");
        ImGui::InputFloat("min", &mNNParams.weightInitBound.x, 0.0f, 0.0f, "%.6f");
        ImGui::InputFloat("max", &mNNParams.weightInitBound.y, 0.0f, 0.0f, "%.6f");
        ImGui::InputInt("training bounces", &mNNParams.trainingBounces);
        ImGui::Separator();
        ImGui::Text("input encoding");
        ImGui::InputInt("hash enc probing size", &mNNParams.featureHashMapProbingSize);
        nn_group.tooltip("The number of slots that are tested when the current slot is occupied.", true);
        ImGui::PopItemWidth();
        mNNParams.reset |= widget.button("Reset nn");
        ImGui::Separator();
    }
    if (Gui::Group debug_group = widget.group("Debug"))
    {
        debug_group.checkbox("path length", mDebugPathLength);
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
    mCamPos = mpScene->getCamera()->getPosition();
    reset();
}


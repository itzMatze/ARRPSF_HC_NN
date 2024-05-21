#if NN_TRAIN
#include "tinynn/TinynnHalfMLP.hlsli"
#include "tinynn/TinynnFeaturegrid.hlsli"
#endif
#include "Scene/SceneDefines.slangh"
#include "Utils/Math/MathConstants.slangh"

import Scene.Scene;
import Scene.RaytracingInline;
import Scene.Shading;
import Scene.HitInfo;
import Scene.Material.ShadingUtils;
import Rendering.Lights.EmissiveLightSampler;
import Utils.Sampling.SampleGenerator;
import Utils.Math.MathHelpers;
import Utils.Geometry.GeometryHelpers;
import Utils.Color.ColorHelpers;
import Utils.Color.ColorMap;
import Utils.Debug.PixelDebug;

import HashCacheHashGridCommon;
import HashCacheCommon;
import LightSampling;

cbuffer CB
{
    uint gFrameCount;
    uint gTrainIteration;
    uint2 gFrameDim;
    float3 gCamPos;
}

// Inputs
Texture2D<PackedHitInfo> gVBuffer;
Texture2D<float4> gViewW;

// Static configuration based on defines set from the host.
#define is_valid(name) (is_valid_##name != 0)
static const uint kLowerBounceCount = LOWER_BOUNCE_COUNT;
static const uint kUpperBounceCount = UPPER_BOUNCE_COUNT;
static const float kRRProbStartValue = RR_PROB_START_VALUE;
static const float kRRProbReductionFactor = RR_PROB_REDUCTION_FACTOR;
static const bool kUseNEE = USE_NEE;
static const bool kUseMIS = USE_MIS;
static const bool kMISUsePowerHeuristic = MIS_USE_POWER_HEURISTIC;
static const bool kUseRR = USE_RR;
static const bool kDebugPathLength = DEBUG_PATH_LENGTH;
static const bool kUseImportanceSampling = USE_IMPORTANCE_SAMPLING;
static const bool kUseEnvBackground = USE_ENV_BACKGROUND;
static const float3 kDefaultBackgroundColor = float3(0.0, 0.0, 0.05);
static const uint kHCHashMapSize = HC_HASHMAP_SIZE;
static const uint kHCDebugVoxels = HC_DEBUG_VOXELS;
static const uint kHCDebugColor = HC_DEBUG_COLOR;
static const uint kHCDebugLevels = HC_DEBUG_LEVELS;
static const uint kNNMaxTrainingBounces = NN_TRAINING_BOUNCES;
static const uint kFeatureHashGridCapacity = FEATURE_HASH_GRID_CAPACITY;

#if NN_TRAIN
struct NNHitInfo
{
    float3 pos;
    float3 dir;
    float3 normal;
    float3 thp;
    float3 radiance;

    __init()
    {
        pos = float3(0.0);
        dir = float3(0.0);
        thp = float3(0.0);
        radiance = float3(0.0);
    }
};

static NNHitInfo nnHitInfoList[kNNMaxTrainingBounces];
static NNHitInfo nnNeeHitInfoList[kNNMaxTrainingBounces];

typedef MLPHalf32X32<NN_LAYER_COUNT0, ReLU> MLPModule0;
static MLPModule0 gMlp0;
#if MLP_COUNT > 1
typedef MLPHalf32X32<NN_LAYER_COUNT1, ReLU> MLPModule1;
static MLPModule1 gMlp1;
#endif
#if MLP_COUNT > 2
typedef MLPHalf32X32<NN_LAYER_COUNT2, ReLU> MLPModule2;
static MLPModule2 gMlp2;
#endif
#endif

#if NN_TRAIN
void updateNNHitInfoRadiance(float3 radiance, int bounce)
{
    // in nirc emitted radiance is applied as incident radiance to the previous position
    bounce -= 1;
    // found light, propagate it through the path
    for (int i = min(kNNMaxTrainingBounces - 1, bounce); i >= 0; i--)
    {
        // light is emitted at current vertex; so, its not reflected by the bsdf
        if (i != bounce) radiance *= nnHitInfoList[i].thp;
        nnHitInfoList[i].radiance += radiance;
    }
}
#endif

struct ScatterRayData
{
#if HC_UPDATE
    HashCacheState hashCacheState;
    float materialRoughness;
    float spread;
#endif
#if NN_TRAIN
    FeatureHashGridData featureHashGrid;
#endif
    float3 radiance;  ///< Accumulated outgoing radiance from path.
    float3 thp;       ///< Current path throughput. This is updated at each path vertex.
    float3 cur_radiance;
    float3 cur_thp;
    uint numBounces;  ///< Path length in number of path segments (0 at origin, 1 at first secondary hit, etc.). Max 2^31.
    float3 origin;    ///< Next path segment origin.
    float3 direction; ///< Next path segment direction.
    float3 normal;
    float pdf;
    float t;
    bool lightSampledUpper;
    bool lightSampledLower;
    bool deltaLobe;

    SampleGenerator sg; ///< Per-ray state for the sample generator (up to 16B).

    /**
     * Initializes ray payload with default parameters.
     */
    __init(SampleGenerator sg)
    {
        this.numBounces = 0;
        this.radiance = float3(0, 0, 0);
        this.thp = float3(1, 1, 1);
        this.cur_radiance = float3(0, 0, 0);
        this.cur_thp = float3(1, 1, 1);
        this.origin = float3(0, 0, 0);
        this.direction = float3(0, 0, 0);
        this.pdf = 1.0;
        this.t = 0.0;
        lightSampledUpper = false;
        lightSampledLower = false;
        // first ray starts at camera which has a "deltaLobe"
        deltaLobe = true;
        this.sg = sg;
    }

    bool isLightSampledUpper() { return lightSampledUpper; }
    bool isLightSampledLower() { return lightSampledLower; }
    bool isLightSampled() { return lightSampledUpper || lightSampledLower; }
    bool isDeltaLobe() { return deltaLobe; }
    [mutating] void setLightSampled(bool upper, bool lower)
    {
        lightSampledUpper = upper;
        lightSampledLower = lower;
    }
    [mutating] void setDeltaLobe(bool delta) { deltaLobe = delta; }
};

/** Evaluates the currently configured heuristic for multiple importance sampling (MIS).
    \param[in] n0 Number of samples taken from the first sampling strategy.
    \param[in] p0 Pdf for the first sampling strategy.
    \param[in] n1 Number of samples taken from the second sampling strategy.
    \param[in] p1 Pdf for the second sampling strategy.
    \return Weight for the contribution from the first strategy (p0).
*/
float evalMIS(float n0, float p0, float n1, float p1)
{
    if (kMISUsePowerHeuristic)
    {
        // Power two heuristic
        float q0 = (n0 * p0) * (n0 * p0);
        float q1 = (n1 * p1) * (n1 * p1);
        return q0 / (q0 + q1);
    }
    else
    {
        // Balance heuristic
        float q0 = n0 * p0;
        float q1 = n1 * p1;
        return q0 / (q0 + q1);
    }
}

/**
 * Traces a shadow ray towards a light source.
 * @param[in] ray Shadow ray to trace
 * @return True if light is visible, false otherwise.
 */
bool traceShadowRay(const Ray ray)
{
    SceneRayQuery<1> rayQuery;
    return (rayQuery.traceVisibilityRay(ray, RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH, 0xff));
}

/**
 * Traces a shadow ray towards a light source.
 * @param[in] origin Ray origin for the shadow ray.
 * @param[in] dir Direction from shading point towards the light source (normalized).
 * @param[in] distance Distance to the light source.
 * @return True if light is visible, false otherwise.
 */
bool traceShadowRay(float3 origin, float3 dir, float distance)
{
    const Ray ray = Ray(origin, dir, 0.0f, distance);
    return traceShadowRay(ray);
}

/**
 * Setup ShadingData based on loaded vertex/material attributes for a hit point.
 * @param[in] hit Hit information.
 * @param[in] rayDir Normalized ray direction.
 * @param[in] lod Method for computing texture level-of-detail.
 * @return ShadingData struct.
 */
ShadingData loadShadingData(const TriangleHit hit, const float3 rayDir, const ITextureSampler lod)
{
    VertexData v = {};
    uint materialID = {};
    v = gScene.getVertexData(hit);
    materialID = gScene.getMaterialID(hit.instanceID);
    ShadingData sd = gScene.materials.prepareShadingData(v, materialID, -rayDir, lod);
    return sd;
}

bool handleHit(const HitInfo hit, inout ScatterRayData rayData)
{
    if (hit.getType() != HitType::Triangle) return false;

    ShadingData sd;
    IMaterialInstance mi;
    TriangleHit triangleHit;
    BSDFProperties bsdfProperties;
#if HC_UPDATE
    HashCacheHitData hashCacheHitData;
#endif
    triangleHit = hit.getTriangleHit();
    let lod = ExplicitLodTextureSampler(0.f);
    sd = loadShadingData(triangleHit, rayData.direction, lod);
    mi = gScene.materials.getMaterialInstance(sd, lod);
    bsdfProperties = mi.getProperties(sd);
#if HC_UPDATE
    // Construct HashCacheHitData structure needed for creating a query point at this hit location
    hashCacheHitData.positionWorld = sd.posW;
    hashCacheHitData.normalWorld = sd.getOrientedFaceNormal();
#endif

    BSDFSample bsdfSample;
    // # light hit by chance
    // always show directly visible lights
    // incorporate emission when: NEE is not active or could not sample this light at the last vertex, MIS is used
    bool computeEmissive = rayData.numBounces >= kLowerBounceCount && kUseEmissiveLights && (!kUseNEE || kUseMIS || !rayData.isLightSampled() || rayData.isDeltaLobe());
    if (computeEmissive && any(bsdfProperties.emission > 0.f))
    {
        float misWeight = 1.0;
        // only apply MIS when NEE is used and was able to sample the light at the last vertex
        if (kUseEmissiveLights && kUseNEE && kUseMIS && rayData.numBounces > 0 && rayData.isLightSampled() && !rayData.isDeltaLobe())
        {
            TriangleLightHit lightHit;
            lightHit.triangleIndex = gScene.lightCollection.getTriangleIndex(triangleHit.instanceID, triangleHit.primitiveIndex);
            lightHit.posW = sd.posW;
            lightHit.normalW = sd.getOrientedFaceNormal();
            // Evaluate PDF at the hit, had it been generated with light sampling.
            // Emissive light samplers have an option to exclusively sample the upper hemisphere.
            bool upperHemisphere = rayData.isLightSampledUpper() && !rayData.isLightSampledLower();
            float lightPdf = getEmissiveSelectionProbability() * gSampler.emissiveSampler.evalPdf(rayData.origin, rayData.normal, upperHemisphere, lightHit);
            // Compute MIS weight by combining this with BSDF sampling.
            // Note we can assume path.pdf > 0.f since we shouldn't have got here otherwise.
            misWeight = evalMIS(1, rayData.pdf, 1, lightPdf);
        }
        rayData.cur_radiance += misWeight * bsdfProperties.emission;
    }

    // # NEE
    LightSample ls;
    PathVertex vertex = PathVertex(sd.posW, sd.faceN, sd.frontFacing);
    const uint lobeTypes = mi.getLobeTypes(sd);
    // NEE not applicable to DeltaLobes, only apply if bounce falls in requested bounce range
    const bool applyNEE = kUseNEE && ((lobeTypes & uint(LobeType::NonDelta)) != 0) && ((rayData.numBounces + 1) >= kLowerBounceCount && (rayData.numBounces + 1) <= kUpperBounceCount);
    rayData.setLightSampled(false, false);
    if (applyNEE)
    {
        // sample a light and store in which hemispheres a light was searched for
        bool sampleUpperHemisphere = ((lobeTypes & uint(LobeType::NonDeltaReflection)) != 0);
        bool sampleLowerHemisphere = ((lobeTypes & uint(LobeType::NonDeltaTransmission)) != 0);
        bool validSample = generateLightSample(vertex, sampleUpperHemisphere, sampleLowerHemisphere, rayData.sg, ls);
        rayData.setLightSampled(sampleUpperHemisphere, sampleLowerHemisphere);
        if (validSample)
        {
            // apply BSDF
            float3 weight = mi.eval(sd, ls.dir, rayData.sg);
            float3 Lr = weight * ls.Li;
            // apply MIS only to non-analytic lights
            if (ls.lightType != uint(LightSampleType::Analytic))
            {
                float scatterPdf = mi.evalPdf(sd, ls.dir, kUseImportanceSampling);
                Lr *= evalMIS(1, ls.pdf, 1, scatterPdf);
            }
            if (any(Lr > 0.f))
            {
                Ray shadowRay = ls.getVisibilityRay();
                bool visible = traceShadowRay(shadowRay);
                if (!visible)
                {
                    ls.Li = float3(0.0);
                }
                rayData.cur_radiance += Lr;
            }
        }
#if NN_TRAIN
        if (rayData.numBounces < kNNMaxTrainingBounces)
        {
            nnNeeHitInfoList[rayData.numBounces].pos = sd.posW;
            nnNeeHitInfoList[rayData.numBounces].dir = ls.dir;
            nnNeeHitInfoList[rayData.numBounces].normal = sd.getOrientedFaceNormal();
            nnNeeHitInfoList[rayData.numBounces].radiance = ls.Li;
        }
#endif
    }

#if HC_UPDATE
    hashCacheUpdateHit(rayData.hashCacheState, hashCacheHitData, rayData.cur_radiance);
#endif
#if NN_TRAIN
    updateNNHitInfoRadiance(rayData.cur_radiance, rayData.numBounces);
#endif // NN_TRAIN

    // generate scatter ray for the next path segment.
    // Sample material.
    if (!mi.sample(sd, rayData.sg, bsdfSample, kUseImportanceSampling)) return false;

    // delta lobes require some caution, store that one is used
    rayData.setDeltaLobe(bsdfSample.isLobe(LobeType::Delta));
    // prevent self-intersection
    rayData.origin = sd.computeRayOrigin(!bsdfSample.isLobe(LobeType::Transmission));
    // update rayData
    rayData.pdf = bsdfSample.pdf;
    rayData.normal = sd.getOrientedFaceNormal();
    rayData.cur_thp *= bsdfSample.weight;
    rayData.direction = bsdfSample.wo;
#if NN_TRAIN
    if (rayData.numBounces < kNNMaxTrainingBounces)
    {
        nnHitInfoList[rayData.numBounces].pos = sd.posW;
        nnHitInfoList[rayData.numBounces].dir = rayData.direction;
        nnHitInfoList[rayData.numBounces].normal = sd.getOrientedFaceNormal();
        nnHitInfoList[rayData.numBounces].thp = rayData.cur_thp;
    }
    else nnHitInfoList[kNNMaxTrainingBounces - 1].thp *= rayData.cur_thp;
#endif
    if (kUseRR)
    {
        float survival_prob = (kRRProbStartValue * pow(kRRProbReductionFactor, rayData.numBounces));
        print("survival_prob", survival_prob);
        if (sampleNext1D(rayData.sg) > survival_prob)
        {
            // russian roulette not survived
            return false;
        }
        rayData.cur_thp /= min(survival_prob, 1.0);
    }
#if HC_UPDATE
    hashCacheSetThroughput(rayData.hashCacheState, rayData.cur_thp);
#endif // HC_UPDATE
    return any(rayData.cur_thp > 0.f);
}

/**
 * Traces a scatter ray based on ray parameters stored in the ray payload.
 * @param[in] rayData Describes the ray parameters. The struct is modified based on the result.
 * @return returns whether the path can be continued
 */
bool traceScatterRay(inout ScatterRayData rayData)
{
    const Ray ray = Ray(rayData.origin, rayData.direction, 0.0f, kRayTMax);
    SceneRayQuery<1> rayQuery;
    HitInfo hit;
    if (!rayQuery.traceRay(ray, hit, rayData.t, RAY_FLAG_NONE, 0xff))
    {
        // add contribution from environment map
        bool computeEnv = kUseEnvLight && (!kUseNEE || kUseMIS || !rayData.isLightSampled() || rayData.isDeltaLobe());
        if (computeEnv && rayData.numBounces >= kLowerBounceCount && rayData.numBounces <= kUpperBounceCount)
        {
            float misWeight = 1.f;
            if (kUseNEE && kUseMIS && rayData.isLightSampled() && !rayData.isDeltaLobe())
            {
                // If NEE and MIS are enabled, and we've already sampled the env map,
                // then we need to evaluate the MIS weight here to account for the remaining contribution.

                // Evaluate PDF, had it been generated with light sampling.
                float lightPdf = getEnvMapSelectionProbability() * gSampler.envMapSampler.evalPdf(rayData.direction);

                // Compute MIS weight by combining this with BSDF sampling.
                // Note we can assume path.pdf > 0.f since we shouldn't have got here otherwise.
                misWeight = evalMIS(1, rayData.pdf, 1, lightPdf);
            }

            float3 Le = gSampler.envMapSampler.eval(rayData.direction);
            rayData.cur_radiance += misWeight * Le;
        }
#if HC_UPDATE
        hashCacheUpdateMiss(rayData.hashCacheState, rayData.cur_radiance);
#endif // HC_UPDATE
#if NN_TRAIN
        updateNNHitInfoRadiance(rayData.cur_radiance, rayData.numBounces);
#endif // NN_TRAIN
        return false;
    }
    return handleHit(hit, rayData);
}

/**
 * This is the main entry point for the path tracer.
 *
 * One path per pixel is generated, which is traced into the scene.
 * The path tracer is written as a for-loop over path segments.
 *
 * Built-in light sources (point, directional) are sampled explicitly at each
 * path vertex. The contributions from mesh lights are added when those are
 * hit. The environment map contribution is added when a scattered ray does
 * not hit any geometry.
 *
 * @param[in] pixel Pixel to trace a path for.
 * @param[in] frameDim Dimension of the frame in pixels.
 * @return Returns the estimated color (radiance).
 */
float3 tracePath(const uint2 pixel, const uint2 frameDim, ScatterRayData rayData)
{
    float3 outColor = float3(0.f);
    const HitInfo hit = HitInfo(gVBuffer[pixel]);

    if (!hit.isValid() || hit.getType() != HitType::Triangle)
    {
        // Background pixel.
        outColor = kUseEnvBackground ? gScene.envMap.eval(-gViewW[pixel].xyz) : kDefaultBackgroundColor;
#if NN_TRAIN
        updateNNHitInfoRadiance(outColor, 0);
#endif
        return outColor;
    }

#if HC_UPDATE
    rayData.hashCacheState.gridParameters.cameraPosition = gCamPos.xyz;
    rayData.hashCacheState.hashMapData.capacity = kHCHashMapSize;
    rayData.materialRoughness = 0.0f;
    rayData.spread = 0.0f;
    rayData.hashCacheState.pathLength = 0;
#endif // HC_UPDATE
    rayData.direction = -gViewW[pixel].xyz;
    if (!handleHit(hit, rayData)) return rayData.cur_radiance;
    rayData.numBounces++;
    // Follow path into the scene and compute its total contribution.
    for (uint i = 1; i <= kUpperBounceCount; i++)
    {
        rayData.radiance += rayData.thp * rayData.cur_radiance;
        rayData.thp *= rayData.cur_thp;
        rayData.cur_thp = float3(1.0f);
        rayData.cur_radiance = float3(0.0f);
        if (!traceScatterRay(rayData)) break;
        rayData.numBounces++;
    }
    // Store contribution from scatter ray.
    rayData.radiance += rayData.thp * rayData.cur_radiance;
    rayData.thp *= rayData.cur_thp;
    outColor += rayData.radiance;
    if (kDebugPathLength)
    {
        print("Path Length:", rayData.numBounces);
        return colormapViridis(float(rayData.numBounces) / float(kUpperBounceCount));
    }
    return outColor;
}

static const int2 gFeatureGridDim = int2(120, 68);

[Differentiable]
float L2Loss(float3 value, no_diff float3 target, no_diff float3 normValue) {
    return dot((value - target), (value - target)) / (dot(normValue, normValue) + 0.01);
}

[numthreads(32, 4, 1)]
void main(uint3 dispatchThreadId: SV_DispatchThreadID,
    int3 groupThreadId: SV_GroupThreadID,
    int3 groupId: SV_GroupID)
{
    SampleGenerator sg = SampleGenerator(dispatchThreadId.xy + gFrameDim * gTrainIteration, gFrameCount);
    uint2 pixel = dispatchThreadId.xy * 10 + uint2(sampleNext2D(sg) * 10.99);

    bool mainThread = (any(pixel >= gFrameDim)) ? false : true;
    printSetPixel(pixel);
    ScatterRayData rayData = ScatterRayData(sg);
#if NN_TRAIN
    [ForceUnroll]
    for (uint i = 0; i < kNNMaxTrainingBounces; i++) nnHitInfoList[i] = NNHitInfo();
    [ForceUnroll]
    for (uint i = 0; i < kNNMaxTrainingBounces; i++) nnNeeHitInfoList[i] = NNHitInfo();
    const ThreadInfo thread_info = ThreadInfo(groupThreadId.xy, int2(32, 4));
    uint param_offset = 0; uint grad_offset = 0;
    gMlp0 = MLPModule0(param_offset, grad_offset, thread_info);
    rayData.featureHashGrid = FeatureHashGridData(kFeatureHashGridCapacity);
    TensorView featureHashGridView = TensorView(param_offset, grad_offset, 32, 32);
#endif
    float3 outputColor = float3(0.0, 1.0, 0.0);
    if (mainThread) outputColor = tracePath(pixel, gFrameDim, rayData);

// NN
#if NN_TRAIN
    for (uint i = 0; i < kNNMaxTrainingBounces; i++)
    {
        HalfFeature<32> feature = computeHashEncFeature(nnHitInfoList[i].pos, nnHitInfoList[i].dir, nnHitInfoList[i].normal, rayData.featureHashGrid, featureHashGridView);
        //HalfFeature<32> feature = computeFreqEncFeature(nnHitInfoList[i].pos, nnHitInfoList[i].dir);
        HalfFeature<32> output = MLPModule0.forward(gMlp0, feature);
        HalfFeature<32>.Differential output_grad;
        float3 target_color = nnHitInfoList[i].radiance;
        float3 color = float3(output.vals[0], output.vals[1], output.vals[2]);
        var color_pair = diffPair(color);
        float loss = L2Loss(color, target_color, color);
        bwd_diff(L2Loss)(color_pair, target_color, color, 1);
        // set gradient to zero if current hitInfoList entry is invalid as it was never updated or if current thread is just helper thread
        const float gradient_scalar = (!mainThread || length(nnHitInfoList[i].dir) < 0.1) ? 0.0 : 1.0;
        output_grad.vals[0] = float16_t(color_pair.d.x * gradient_scalar);
        output_grad.vals[1] = float16_t(color_pair.d.y * gradient_scalar);
        output_grad.vals[2] = float16_t(color_pair.d.z * gradient_scalar);
        var input_feature_pair = diffPair(feature);
        bwd_diff(MLPModule0.forward)(gMlp0, input_feature_pair, output_grad);
        bwd_diff(computeHashEncFeature)(nnHitInfoList[i].pos, nnHitInfoList[i].dir, nnHitInfoList[i].normal, rayData.featureHashGrid, featureHashGridView, input_feature_pair.d);
    }
    // nee training
    for (uint i = 0; i < kNNMaxTrainingBounces; i++)
    {
        HalfFeature<32> feature = computeHashEncFeature(nnNeeHitInfoList[i].pos, nnNeeHitInfoList[i].dir, nnNeeHitInfoList[i].normal, rayData.featureHashGrid, featureHashGridView);
        //HalfFeature<32> feature = computeFreqEncFeature(nnNeeHitInfoList[i].pos, nnNeeHitInfoList[i].dir);
        HalfFeature<32> output = MLPModule0.forward(gMlp0, feature);
        HalfFeature<32>.Differential output_grad;
        float3 target_color = nnNeeHitInfoList[i].radiance;
        float3 color = float3(output.vals[0], output.vals[1], output.vals[2]);
        var color_pair = diffPair(color);
        float loss = L2Loss(color, target_color, color);
        bwd_diff(L2Loss)(color_pair, target_color, color, 1);
        // set gradient to zero if current hitInfoList entry is invalid as it was never updated or if current thread is just helper thread
        const float gradient_scalar = (!mainThread || length(nnNeeHitInfoList[i].dir) < 0.1) ? 0.0 : 1.0;
        output_grad.vals[0] = float16_t(color_pair.d.x * gradient_scalar);
        output_grad.vals[1] = float16_t(color_pair.d.y * gradient_scalar);
        output_grad.vals[2] = float16_t(color_pair.d.z * gradient_scalar);
        var input_feature_pair = diffPair(feature);
        bwd_diff(MLPModule0.forward)(gMlp0, input_feature_pair, output_grad);
        bwd_diff(computeHashEncFeature)(nnNeeHitInfoList[i].pos, nnNeeHitInfoList[i].dir, nnNeeHitInfoList[i].normal, rayData.featureHashGrid, featureHashGridView, input_feature_pair.d);
    }
    // use dummy normal for camera
    HalfFeature<32> feature = computeHashEncFeature(gCamPos.xyz, -gViewW[pixel].xyz, float3(1.0), rayData.featureHashGrid, featureHashGridView);
    //HalfFeature<32> feature = computeFreqEncFeature(gCamPos.xyz, -gViewW[pixel].xyz);
    HalfFeature<32> output = MLPModule0.forward(gMlp0, feature);
    HalfFeature<32>.Differential output_grad;
    float3 target_color = outputColor;
    float3 color = float3(output.vals[0], output.vals[1], output.vals[2]);
    var color_pair = diffPair(color);
    float loss = L2Loss(color, target_color, color);
    bwd_diff(L2Loss)(color_pair, target_color, color, 1);
    // set gradient to zero if current thread is just helper thread
    const float gradient_scalar = (!mainThread) ? 0.0 : 1.0;
    output_grad.vals[0] = float16_t(color_pair.d.x * gradient_scalar);
    output_grad.vals[1] = float16_t(color_pair.d.y * gradient_scalar);
    output_grad.vals[2] = float16_t(color_pair.d.z * gradient_scalar);
    var input_feature_pair = diffPair(feature);
    bwd_diff(MLPModule0.forward)(gMlp0, input_feature_pair, output_grad);
    bwd_diff(computeHashEncFeature)(gCamPos.xyz, -gViewW[pixel].xyz, float3(1.0), rayData.featureHashGrid, featureHashGridView, input_feature_pair.d);
#endif
}


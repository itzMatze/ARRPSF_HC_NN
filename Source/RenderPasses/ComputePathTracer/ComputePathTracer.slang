#if NN_QUERY
#include "tinynn/TinynnHalfMLP.slang"
#include "tinynn/TinynnFeatureEncodings.hlsli"
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

import RadianceHashCacheHashGridCommon;
import RadianceHashCacheCommon;
import LightSampling;

cbuffer CB
{
    uint gFrameCount;
    uint2 gFrameDim;
    float3 gCamPos;
    int gHashEncDebugLevel;
    uint64_t gWeightsAddress;
}

#define GLSL_SHARED_MEMORY_SIZE 5120

// inputs
Texture2D<PackedHitInfo> gVBuffer;
Texture2D<float4> gViewW;
Texture2D<float4> gRefImage;

// outputs
RWTexture2D<float4> gOutputColor;

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
static const uint kHCInjectRadianceSpread = HC_INJECT_RADIANCE_SPREAD;
static const float kNRCInjectRadianceSpreadThreshold = 0.01;
static const float kNIRCInjectRadianceSpreadThreshold = 0.01;
static const uint kNNMaxTrainingBounces = NN_TRAINING_BOUNCES;

#if NN_QUERY
typedef MLPHalf32X32<NN_LAYER_COUNT0, ReLU> MLPModule0;
static MLPModule0 gMlp0;
#endif

#if KEEP_THREADS
static bool gDone = false;
#define MASK_BLOCK if(!gDone)
#define TERMINATE_PATH_0 gDone = true
#define TERMINATE_PATH_1 gDone = true
#else
#define MASK_BLOCK
#define TERMINATE_PATH_0 return false
#define TERMINATE_PATH_1 return rayData.cur_radiance
#endif

static uint2 gPixel;

struct ScatterRayData
{
    // spread for sharc method and for nrc method
    float spread[2];
    float initialSpread;
    float materialRoughness;
#if HC_QUERY
    hc::HashCacheState hashCacheState;
#endif
#if NN_QUERY
    FeatureHashGrid featureHashGrid0;
#if MLP_COUNT > 1
    FeatureHashGrid featureHashGrid1;
#endif
#endif
    float3 radiance;  ///< Accumulated outgoing radiance from path.
    float3 thp;       ///< Current path throughput. This is updated at each path vertex.
    float3 cur_radiance;
    float3 cur_thp;
    float luminance_estimate;
    uint numBounces;  ///< Path length in number of path segments (0 at origin, 1 at first secondary hit, etc.). Max 2^31.
    float3 origin;    ///< Next path segment origin.
    float3 direction; ///< Next path segment direction.
    float3 normal;
    float pdf;
    float t;
    float distance;
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
}

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
    MASK_BLOCK if (hit.getType() != HitType::Triangle) TERMINATE_PATH_0;

    ShadingData sd;
    IMaterialInstance mi;
    TriangleHit triangleHit;
    BSDFProperties bsdfProperties;
#if HC_QUERY
    hc::HashCacheHitData hashCacheHitData;
#endif
    MASK_BLOCK
    {
        triangleHit = hit.getTriangleHit();
        let lod = ExplicitLodTextureSampler(0.f);
        sd = loadShadingData(triangleHit, rayData.direction, lod);
        mi = gScene.materials.getMaterialInstance(sd, lod);
        bsdfProperties = mi.getProperties(sd);
        if (rayData.t == 0.0)
            rayData.distance = distance(sd.posW, gCamPos);
        else
            rayData.distance += rayData.t;
        float alpha = min(rayData.materialRoughness * rayData.materialRoughness, 0.99);
        rayData.materialRoughness = bsdfProperties.roughness;
        alpha *= alpha;
        rayData.spread[0] += rayData.t * sqrt(0.5f * alpha / (1.0f - alpha));
        if (rayData.numBounces == 0)
        {
            rayData.initialSpread = distance(gCamPos, sd.posW) / (M_4PI * dot(-rayData.direction, sd.getOrientedFaceNormal()));
        }
        else if (rayData.pdf != 0.0f)
        {
            rayData.spread[1] +=
                sqrt(distance(sd.posW, rayData.origin) / (rayData.pdf * abs(dot(-rayData.direction, sd.getOrientedFaceNormal()))));
        }

#if NN_QUERY
        if (gHashEncDebugLevel >= 0)
        {
            rayData.cur_radiance = rayData.featureHashGrid0.hashmap.DebugColoredHash(
                sd.posW, -gViewW[gPixel].xyz, sd.getOrientedFaceNormal(), gHashEncDebugLevel
            );
            TERMINATE_PATH_0;
        }
#endif
    }
#if NN_DEBUG && USE_NRC
    if (gDone || (rayData.numBounces >= kLowerBounceCount && rayData.numBounces <= kUpperBounceCount))
    {
        HalfFeature<32> feature;
        if (!gDone)
        {
            feature = computeFeature(sd.posW, rayData.direction, sd.getOrientedFaceNormal(), rayData.featureHashGrid0);
        }
        else
        {
            [ForceUnroll] for (uint i = 0; i < 32; i++) feature.vals[i] = 0.0h;
        }

        rayData.cur_radiance = float3(0.0, 0.0, 0.0);
        
            HalfFeature<32> output = MLPModule0.forward_fast(gMlp0, feature);
            MASK_BLOCK
            {
                if (isnan(output.vals[0]))
                    rayData.cur_radiance = float3(1.0, 0.0, 0.0);
                else if (isinf(output.vals[0]))
                    rayData.cur_radiance = float3(1.0, 0.0, 1.0);
                else
                    rayData.cur_radiance += float3(output.vals[0], output.vals[1], output.vals[2]);
            }
        
        TERMINATE_PATH_0;
    }
#endif
#if USE_NRC && NN_INJECT_RADIANCE_SPREAD
    bool validHit = rayData.spread[1] * rayData.spread[1] > kNRCInjectRadianceSpreadThreshold * rayData.initialSpread;
    if (WaveActiveAnyTrue(validHit))
    {
        HalfFeature<32> feature;
        if (!gDone && validHit)
        {
            feature = computeFeature(sd.posW, rayData.direction, sd.getOrientedFaceNormal(), rayData.featureHashGrid0);
        }
        else
        {
            [ForceUnroll] for (uint i = 0; i < 32; i++) feature.vals[i] = 0.0h;
        }
        
            HalfFeature<32> output = MLPModule0.forward_fast(gMlp0, feature);
            if (!gDone && validHit)
            {
                rayData.cur_radiance += float3(output.vals[0], output.vals[1], output.vals[2]);
                TERMINATE_PATH_0;
            }
        
    }
#endif
    MASK_BLOCK
    {
#if HC_QUERY
        // Construct HashCacheHitData structure needed for creating a query point at this hit location
        hashCacheHitData.positionWorld = sd.posW;
        hashCacheHitData.normalWorld = sd.getOrientedFaceNormal();
        hashCacheHitData.distance = rayData.distance;
        hashCacheHitData.direction = rayData.direction;
        uint gridLevel = rayData.hashCacheState.hashMapData.GetGridLevel(hashCacheHitData.distance);
        if (kHCDebugLevels)
        {
            // the actual max level is 1024 (see clamp in function)
            // however, the upper levels are never really used, thus use 8 to get meaningful visualization
            rayData.cur_radiance = colormapViridis(float(gridLevel) / 8.0f);
            return false;
        }
#if USE_RHC
        if ((kHCDebugVoxels || kHCDebugColor) && (rayData.numBounces >= kLowerBounceCount && rayData.numBounces <= kUpperBounceCount))
        {
            // debug active; voxel debug will fetch voxel visualization values, otherwise just get color
            float3 debugColor;
            hc::hashCacheGetCachedRadiance(rayData.hashCacheState, hashCacheHitData, debugColor, kHCDebugVoxels);
            rayData.cur_radiance = debugColor;
            return false;
        }

        if (kHCInjectRadianceSpread)
        {
            float voxelSize = rayData.hashCacheState.hashMapData.GetVoxelSize(gridLevel);
            bool isValidHit = rayData.spread[0] * lerp(2.0f, 4.0f, sampleNext1D(rayData.sg)) > voxelSize;
            isValidHit &= rayData.numBounces > 0;

            float3 hashCacheRadiance;
            if (isValidHit && hc::hashCacheGetCachedRadiance(rayData.hashCacheState, hashCacheHitData, hashCacheRadiance))
            {
                rayData.cur_radiance = hashCacheRadiance;
                TERMINATE_PATH_0; // Terminate the path once we've looked up into the cache
            }
        }
#endif
#endif
    }

    MASK_BLOCK
    {
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
    }

    // # NEE
    LightSample ls;
    float3 Lr;
    bool applyNEE = false;
    bool validSample = false;
    float3 weight = float3(0.0);
    MASK_BLOCK
    {
        PathVertex vertex = PathVertex(sd.posW, sd.faceN, sd.frontFacing);
        const uint lobeTypes = mi.getLobeTypes(sd);
        // NEE not applicable to DeltaLobes, only apply if bounce falls in requested bounce range
        applyNEE = kUseNEE && ((lobeTypes & uint(LobeType::NonDelta)) != 0) && ((rayData.numBounces + 1) >= kLowerBounceCount && (rayData.numBounces + 1) <= kUpperBounceCount);
        rayData.setLightSampled(false, false);
        if (applyNEE)
        {
            // sample a light and store in which hemispheres a light was searched for
            bool sampleUpperHemisphere = ((lobeTypes & uint(LobeType::NonDeltaReflection)) != 0);
            bool sampleLowerHemisphere = ((lobeTypes & uint(LobeType::NonDeltaTransmission)) != 0);
            validSample = generateLightSample(vertex, sampleUpperHemisphere, sampleLowerHemisphere, rayData.sg, ls);
            rayData.setLightSampled(sampleUpperHemisphere, sampleLowerHemisphere);
            if (validSample)
            {
                // apply BSDF
                weight = mi.eval(sd, ls.dir, rayData.sg);
                Lr = weight * ls.Li;
                // apply MIS only to non-analytic lights
                if (ls.lightType != uint(LightSampleType::Analytic))
                {
                    float scatterPdf = mi.evalPdf(sd, ls.dir, kUseImportanceSampling);
                    Lr *= evalMIS(1, ls.pdf, 1, scatterPdf);
                }
            }
        }
    }

    MASK_BLOCK
    {
        if (applyNEE && validSample)
        {
            if (any(Lr > 0.f))
            {
                Ray shadowRay = ls.getVisibilityRay();
                bool visible = traceShadowRay(shadowRay);
                if (visible)
                {
                    rayData.cur_radiance += Lr;
                }
            }
        }
    }

    BSDFSample bsdfSample;
    MASK_BLOCK
    {
        // generate scatter ray for the next path segment.
        // Sample material.
        if (!mi.sample(sd, rayData.sg, bsdfSample, kUseImportanceSampling)) TERMINATE_PATH_0;
    }

    MASK_BLOCK
    {
        // delta lobes require some caution, store that one is used
        rayData.setDeltaLobe(bsdfSample.isLobe(LobeType::Delta));
        // prevent self-intersection
        rayData.origin = sd.computeRayOrigin(!bsdfSample.isLobe(LobeType::Transmission));
        // update rayData
        rayData.pdf = bsdfSample.pdf;
        rayData.normal = sd.getOrientedFaceNormal();
        rayData.cur_thp *= bsdfSample.weight;
    }
#if USE_NIRC
#if NN_DEBUG
    if (rayData.numBounces + 1 >= kLowerBounceCount && rayData.numBounces + 1 <= kUpperBounceCount)
    {
        HalfFeature<32> feature;
        // evaluate nn for the bsdf sampled direction
        if (!gDone)
        {
            feature = computeFeature(sd.posW, bsdfSample.wo, sd.getOrientedFaceNormal(), rayData.featureHashGrid0);
        }
        else
        {
            [ForceUnroll]
            for (uint i = 0; i < 32; i++) feature.vals[i] = 0.0h;
        }
        HalfFeature<32> output = MLPModule0.forward_fast(gMlp0, feature);
        float3 color = float3(output.vals[0], output.vals[1], output.vals[2]) * bsdfSample.weight;
        MASK_BLOCK rayData.cur_radiance += color;
        TERMINATE_PATH_0;
    }
#elif NN_INJECT_RADIANCE_SPREAD
    bool validHit = rayData.spread[1] * rayData.spread[1] > kNIRCInjectRadianceSpreadThreshold * rayData.initialSpread;
    if (WaveActiveAnyTrue(validHit))
    {
        HalfFeature<32> feature;
        if (!gDone && validHit)
        {
            feature = computeFeature(sd.posW, bsdfSample.wo, sd.getOrientedFaceNormal(), rayData.featureHashGrid0);
        }
        else
        {
            [ForceUnroll]
            for (uint i = 0; i < 32; i++) feature.vals[i] = 0.0h;
        }
        HalfFeature<32> output = MLPModule0.forward_fast(gMlp0, feature);
        if (!gDone && validHit)
        {
            rayData.cur_radiance += float3(output.vals[0], output.vals[1], output.vals[2]) * bsdfSample.weight;
            TERMINATE_PATH_0;
        }
    }
#endif
#endif
#if HC_QUERY && USE_IRHC
    MASK_BLOCK
    {
        hashCacheHitData.direction = bsdfSample.wo;
#if HC_DEBUG_COLOR
        if (rayData.numBounces >= kLowerBounceCount && rayData.numBounces <= kUpperBounceCount)
        {
            float3 debugColor;
            hc::hashCacheGetCachedRadiance(rayData.hashCacheState, hashCacheHitData, debugColor);
            rayData.cur_radiance += debugColor * bsdfSample.weight;
            TERMINATE_PATH_0;
        }
#endif
    }
    MASK_BLOCK
    {
        if (kHCInjectRadianceSpread)
        {
            uint gridLevel = rayData.hashCacheState.hashMapData.GetGridLevel(hashCacheHitData.distance);
            float voxelSize = rayData.hashCacheState.hashMapData.GetVoxelSize(gridLevel);
            bool isValidHit = rayData.spread[0] * lerp(2.0f, 4.0f, sampleNext1D(rayData.sg)) > voxelSize;
            isValidHit &= rayData.numBounces > 0;

            float3 hashCacheRadiance;
            if (isValidHit && hc::hashCacheGetCachedRadiance(rayData.hashCacheState, hashCacheHitData, hashCacheRadiance))
            {
                rayData.cur_radiance += hashCacheRadiance * bsdfSample.weight;
                TERMINATE_PATH_0; // Terminate the path once we've looked up into the cache
            }
        }
    }
#endif


    if (kUseRR)
    {
        if (rayData.numBounces == 0)
        {
            // set pixel measurement estimate
#if RR_OPTION_BITS & (1u << 5)
            rayData.luminance_estimate = luminance(gRefImage[gPixel].xyz);
#elif RR_OPTION_BITS & (1u << 6)
            float3 hcRadiance;
            // only use hc for survival prob if the hc is ready at the current position, otherwise signal that the query was not successful
            if (!hc::hashCacheGetCachedRadiance(rayData.hashCacheState, hashCacheHitData, hcRadiance))
            {
                hcRadiance = float3(-1.0);
            }
            // if hc query was not successful use 0.5 as luminance
#if USE_IRHC
            hcRadiance *= bsdfSample.weight;
            hcRadiance += rayData.cur_radiance;
#endif
            rayData.luminance_estimate = hcRadiance.x >= 0.0 ? luminance(hcRadiance) : 0.5;
            print("hc global estimate", hcRadiance);
#elif RR_OPTION_BITS & (1u << 7)
#if USE_NIRC
            HalfFeature<32> feature;
            if (!gDone)
            {
                feature = computeFeature(sd.posW, bsdfSample.wo, sd.getOrientedFaceNormal(), rayData.featureHashGrid0);
            }
            else
            {
                [ForceUnroll]
                for (uint i = 0; i < 32; i++) feature.vals[i] = 0.0h;
            }
            HalfFeature<32> output = MLPModule0.forward_fast(gMlp0, feature);
            float3 nnRadiance = float3(output.vals[0], output.vals[1], output.vals[2]) * bsdfSample.weight + rayData.cur_radiance;
            rayData.luminance_estimate = luminance(nnRadiance);
            print("nirc global estimate", nnRadiance);
#elif USE_NRC
            HalfFeature<32> feature;
            if (!gDone)
            {
                feature = computeFeature(sd.posW, -gViewW[gPixel].xyz, sd.getOrientedFaceNormal(), rayData.featureHashGrid0);
            }
            else
            {
                [ForceUnroll]
                for (uint i = 0; i < 32; i++) feature.vals[i] = 0.0h;
            }
            HalfFeature<32> output = MLPModule0.forward_fast(gMlp0, feature);
            float3 nnRadiance = float3(output.vals[0], output.vals[1], output.vals[2]);
            rayData.luminance_estimate = luminance(nnRadiance);
            print("nrc global estimate", nnRadiance);
#endif
#endif
        }

        // use the parameters as fallback; will be overwritten by the actually used method
        float survival_prob = (kRRProbStartValue * pow(kRRProbReductionFactor, rayData.numBounces));;
        float3 radiance = float3(0.0);
#if RR_OPTION_BITS & ((1u << 1) | (1u << 2))
#if RR_OPTION_BITS & (1u << 3)
        // only use hc for survival prob if the hc is ready at the current position, otherwise signal that the query was not successful
        if (!hc::hashCacheGetCachedRadiance(rayData.hashCacheState, hashCacheHitData, radiance))
        {
            radiance = float3(-1.0);
        }
#if USE_IRHC
        radiance *= bsdfSample.weight;
#endif
#elif RR_OPTION_BITS & (1u << 4)
#if USE_NIRC
        HalfFeature<32> feature;
        if (!gDone)
        {
            feature = computeFeature(sd.posW, bsdfSample.wo, sd.getOrientedFaceNormal(), rayData.featureHashGrid0);
        }
        else
        {
            [ForceUnroll]
            for (uint i = 0; i < 32; i++) feature.vals[i] = 0.0h;
        }
        HalfFeature<32> output = MLPModule0.forward_fast(gMlp0, feature);
        radiance = float3(output.vals[0], output.vals[1], output.vals[2]) * bsdfSample.weight;
#elif USE_NRC
        HalfFeature<32> feature;
        if (!gDone)
        {
            feature = computeFeature(sd.posW, rayData.direction, sd.getOrientedFaceNormal(), rayData.featureHashGrid0);
        }
        else
        {
            [ForceUnroll]
            for (uint i = 0; i < 32; i++) feature.vals[i] = 0.0h;
        }
        HalfFeature<32> output = MLPModule0.forward_fast(gMlp0, feature);
        radiance = float3(output.vals[0], output.vals[1], output.vals[2]);
#endif
#endif
#if RR_OPTION_BITS & (1u << 1)
        if (radiance.x >= 0.0)
        {
            radiance *= rayData.thp;
            survival_prob = luminance(radiance);
            survival_prob = min(survival_prob + 0.5, (kRRProbStartValue * pow(kRRProbReductionFactor, rayData.numBounces)));
        }
#elif RR_OPTION_BITS & (1u << 2)
        // adrr
        float local_luminance_estimate = luminance(radiance);
        if (local_luminance_estimate < 0.001) local_luminance_estimate = 0.001;
        float center = rayData.luminance_estimate / local_luminance_estimate;
        static const float s = 5.0;
        float low = (2.0 * center) / (1.0 + s);
        float high = s * low;
        float weight = luminance(rayData.thp);
        if (weight < low) survival_prob = weight / low;
        else survival_prob = 1.0;
#endif
#endif
        MASK_BLOCK
        {
            rayData.direction = bsdfSample.wo;
            if (sampleNext1D(rayData.sg) > survival_prob)
            {
                // russian roulette not survived
#if INJECT_RADIANCE_RR
                if (radiance.x >= 0.0)
                {
                    rayData.cur_radiance += radiance * rayData.cur_thp;
                }
                TERMINATE_PATH_0;
            }
#else
                TERMINATE_PATH_0;
            }
            else rayData.cur_thp /= min(survival_prob, 1.0);
#endif
        }
    }
    MASK_BLOCK
    {
        rayData.direction = bsdfSample.wo;
        return any(rayData.cur_thp > 0.f);
    }
#if KEEP_THREADS
    return false;
#endif
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
    MASK_BLOCK
    {
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
            TERMINATE_PATH_0;
        }
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
 * @param[in] frameDim Dimension of the frame in pixels.
 * @return Returns the estimated color (radiance).
 */
float3 tracePath(const uint2 frameDim, ScatterRayData rayData)
{
    float3 outColor = float3(0.f);
    const HitInfo hit = HitInfo(gVBuffer[gPixel]);

    if (!hit.isValid() || hit.getType() != HitType::Triangle)
    {
        // Background pixel.
        rayData.cur_radiance = kUseEnvBackground ? gScene.envMap.eval(-gViewW[gPixel].xyz) : kDefaultBackgroundColor;
        TERMINATE_PATH_1;
    }

    rayData.spread[0] = 0.0f;
    rayData.spread[1] = 0.0f;
    rayData.materialRoughness = 0.0f;
    rayData.direction = -gViewW[gPixel].xyz;
    if (!handleHit(hit, rayData)) TERMINATE_PATH_1;
    MASK_BLOCK rayData.numBounces++;
    // Follow path into the scene and compute its total contribution.
    for (uint i = 1; i <= kUpperBounceCount; i++)
    {
        rayData.radiance += rayData.thp * rayData.cur_radiance;
        rayData.thp *= rayData.cur_thp;
        rayData.cur_thp = float3(1.0f);
        rayData.cur_radiance = float3(0.0f);
#if KEEP_THREADS
        if (WaveActiveAllTrue(gDone)) break;
        traceScatterRay(rayData);
#else
        if (!traceScatterRay(rayData)) break;
#endif
        MASK_BLOCK rayData.numBounces++;
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

[numthreads(32, 4, 1)]
void main(uint3 dispatchThreadId: SV_DispatchThreadID,
    int3 groupThreadId: SV_GroupThreadID,
    int3 groupId: SV_GroupID)
{
    gPixel = dispatchThreadId.xy;
#if NN_QUERY
    if (any(gPixel >= gFrameDim)) gDone = true;
#else
    if (any(gPixel >= gFrameDim)) return;
#endif
    printSetPixel(gPixel);
    SampleGenerator sg = SampleGenerator(gPixel, gFrameCount);
    ScatterRayData rayData = ScatterRayData(sg);
#if NN_QUERY
    const ThreadInfo thread_info = ThreadInfo(groupThreadId.xy, int2(32, 4));
    uint param_offset = 0; uint grad_offset = 0;
    gMlp0 = MLPModule0(param_offset, grad_offset, thread_info, gWeightsAddress);
    rayData.featureHashGrid0 = FeatureHashGrid(param_offset, grad_offset);
#endif
    float3 outputColor = float3(0.0, 1.0, 0.0);
    outputColor = tracePath(gFrameDim, rayData);
    gOutputColor[gPixel] = float4(outputColor, 1.0f);
}


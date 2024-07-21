#include "tinynn/TinynnHalfMLP.hlsli"
#include "tinynn/TinynnFeatureEncodings.hlsli"
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
    float3 gCamPos;
    uint2 gDebugPixel;
    uint gFrameCount;
    uint gMLPIndex;
    bool gShowTransmission;
    bool gApplyBSDF;
    bool gAccumulate;
}

// Inputs
Texture2D<PackedHitInfo> gVBuffer;
Texture2D<float4> gViewW;

// Outputs
RWTexture2D<float4> gIRDebugOutputColor;
RWTexture2D<float4> gIRDebugOutputColorRef;

#if SHOW_NIRC
// Static configuration based on defines set from the host.
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

struct ScatterRayData
{
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
    if (hit.getType() != HitType::Triangle) return false;

    ShadingData sd;
    IMaterialInstance mi;
    TriangleHit triangleHit;
    BSDFProperties bsdfProperties;
    triangleHit = hit.getTriangleHit();
    let lod = ExplicitLodTextureSampler(0.f);
    sd = loadShadingData(triangleHit, rayData.direction, lod);
    mi = gScene.materials.getMaterialInstance(sd, lod);
    bsdfProperties = mi.getProperties(sd);

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
                    Lr = float3(0.0);
                }
                rayData.cur_radiance += Lr;
            }
        }
    }

    // generate scatter ray for the next path segment.
    // Sample material.
    if (!mi.sample(sd, rayData.sg, bsdfSample, kUseImportanceSampling)) return false;

    // delta lobes require some caution, store that one is used
    rayData.setDeltaLobe(bsdfSample.isLobe(LobeType::Delta));
    // prevent self-intersection
    rayData.origin = sd.computeRayOrigin(!bsdfSample.isLobe(LobeType::Transmission));
    // update rayData
    rayData.pdf = bsdfSample.pdf;
    rayData.direction = bsdfSample.wo;
    rayData.normal = sd.getOrientedFaceNormal();
    rayData.cur_thp *= bsdfSample.weight;
    if (kUseRR)
    {
        float survival_prob = (kRRProbStartValue * pow(kRRProbReductionFactor, rayData.numBounces));
        if (sampleNext1D(rayData.sg) > survival_prob)
        {
            // russian roulette not survived
            return false;
        }
        rayData.cur_thp /= min(survival_prob, 1.0);
    }
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
float3 tracePath(const uint2 frameDim, ScatterRayData rayData)
{
    float3 outColor = float3(0.f);
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

// https://gist.github.com/kevinmoran/b45980723e53edeb8a5a43c49f134724
matrix<float, 3, 3> rotateAlign(float3 v1, float3 v2)
{
    const float cos_a = dot(v1, v2);
    if (cos_a > 0.999)
    {
        // vectors are already aligned, return identity
        return matrix<float, 3, 3>(float3(1.0, 0.0, 0.0),
                                   float3(0.0, 1.0, 0.0),
                                   float3(0.0, 0.0, 1.0));
    }
    else if (cos_a < -0.999)
    {
        // vectors are exactly opposing, negate vector
        return matrix<float, 3, 3>(float3(-1.0, 0.0, 0.0),
                                   float3(0.0, -1.0, 0.0),
                                   float3(0.0, 0.0, -1.0));
    }
    else
    {
        const float3 axis = cross(v1, v2);
        const float k = 1.0 / (1.0 + cos_a);
        return matrix<float, 3, 3>(float3((axis.x * axis.x * k) + cos_a, (axis.y * axis.x * k) - axis.z, (axis.z * axis.x * k) + axis.y),
                                   float3((axis.x * axis.y * k) + axis.z, (axis.y * axis.y * k) + cos_a, (axis.z * axis.y * k) - axis.x),
                                   float3((axis.x * axis.z * k) - axis.y, (axis.y * axis.z * k) + axis.x, (axis.z * axis.z * k) + cos_a));
    }
}

[numthreads(32, 4, 1)]
void main(uint3 dispatchThreadId: SV_DispatchThreadID,
    int3 groupThreadId: SV_GroupThreadID,
    int3 groupId: SV_GroupID)
{
    uint2 pixel = dispatchThreadId.xy;
    bool mainThread = true;
    const uint width = IR_DEBUG_OUTPUT_WIDTH;
    const uint height = IR_DEBUG_OUTPUT_HEIGHT;
    if (any(pixel >= uint2(width, height))) mainThread = false;
    //if (any(pixel >= uint2(width, height)) && traceShadowRay(float3(0.0), float3(1.0, 0.0, 0.0), 0.0)) mainThread = false;
    printSetPixel(pixel);
    SampleGenerator sg = SampleGenerator(pixel, gFrameCount);
#if SHOW_NIRC
    const ThreadInfo thread_info = ThreadInfo(groupThreadId.xy, int2(32, 4));
    uint param_offset = 0; uint grad_offset = 0;
    gMlp0 = MLPModule0(param_offset, grad_offset, thread_info);
    FeatureHashGrid featureHashGrid0 = FeatureHashGrid(param_offset, grad_offset);
#if MLP_COUNT > 1
    gMlp1 = MLPModule1(param_offset, grad_offset, thread_info);
    FeatureHashGrid featureHashGrid1 = FeatureHashGrid(param_offset, grad_offset);
#endif
#endif
    const HitInfo hit = HitInfo(gVBuffer[gDebugPixel]);
    if (hit.getType() != HitType::Triangle) mainThread = false;
    float3 normal = float3(1.0);
    float3 pos = float3(1.0);
    float3 dir = float3(1.0);
    ShadingData sd;
    IMaterialInstance mi;
    float3 outputColor;
    if (mainThread)
    {
        TriangleHit triangleHit = hit.getTriangleHit();
        let lod = ExplicitLodTextureSampler(0.f);
        sd = loadShadingData(triangleHit, -gViewW[gDebugPixel].xyz, lod);
        mi = gScene.materials.getMaterialInstance(sd, lod);
        BSDFProperties bsdfProperties = mi.getProperties(sd);
        pos = sd.posW;

        ScatterRayData rayData = ScatterRayData(sg);
        rayData.direction = normalize(pos - gCamPos);
        handleHit(hit, rayData);

        normal = gShowTransmission ? -sd.getOrientedFaceNormal() : sd.getOrientedFaceNormal();
        float pitch = -M_PI_2 + ((float(pixel.x) + 0.5) / float(height)) * M_PI;
        float yaw = -M_PI_2 + ((float(pixel.y) + 0.5) / float(width)) * M_PI;
        dir = normalize(float3(cos(yaw) * cos(pitch), sin(yaw) * cos(pitch), sin(pitch)));
        const matrix<float, 3, 3> rot = rotateAlign(float3(1.0, 0.0, 0.0), normal);
        dir = normalize(mul(rot, dir));
        rayData = ScatterRayData(sg);
        rayData.numBounces++;
        rayData.origin = sd.computeRayOrigin(!gShowTransmission);
        rayData.direction = dir;
        outputColor = tracePath(uint2(width, height), rayData);
    }
    float3 outputColorIR = float3(0.0, 1.0, 0.0);
#if SHOW_NIRC
    HalfFeature<32> output;
    if (gMLPIndex == 0)
    {
        HalfFeature<32> feature = computeFeature(pos, dir, normal, featureHashGrid0);
        output = MLPModule0.forward(gMlp0, feature);
    }
#if MLP_COUNT > 1
    else if (gMLPIndex == 1)
    {
        HalfFeature<32> feature = computeFeature(pos, dir, normal, featureHashGrid1);
        output = MLPModule1.forward(gMlp1, feature);
    }
#endif
    if (isnan(output.vals[0])) outputColorIR = float3(1.0, 0.0, 0.0);
    else if (isinf(output.vals[0])) outputColorIR = float3(1.0, 0.0, 1.0);
    else outputColorIR = float3(output.vals[0], output.vals[1], output.vals[2]);
#elif SHOW_IRHC
    hc::HashCacheState hashCacheState;
    hc::HashCacheHitData hashCacheHitData;
    hashCacheHitData.positionWorld = sd.posW;
    hashCacheHitData.normalWorld = sd.getOrientedFaceNormal();
    hashCacheHitData.distance = distance(gCamPos, sd.posW);
    hashCacheHitData.direction = dir;
    float3 debugColor;
    hc::hashCacheGetCachedRadiance(hashCacheState, hashCacheHitData, debugColor);
    outputColorIR = debugColor;
#endif
    if (mainThread && gApplyBSDF)
    {
        BSDFSample bsdfSample;
        float3 weight;
        // make delta lobes visible by manually setting the weight if current material uses delta lobe
        if (mi.sample(sd, sg, bsdfSample, kUseImportanceSampling) && bsdfSample.isLobe(LobeType::Delta))
        {
            if (dot(dir, bsdfSample.wo) > 0.999) weight = float3(1.0);
            else weight = float3(0.0);
        }
        else weight = mi.eval(sd, dir, sg);
        outputColorIR *= weight;
        outputColor *= weight;
    }
    if (mainThread)
    {
        gIRDebugOutputColor[pixel] = float4(outputColorIR, 1.0f);
        float4 value = gIRDebugOutputColorRef[pixel];
        value.xyz = value.xyz * value.w + outputColor;
        value.w++;
        value.xyz /= value.w;
        if (gAccumulate) gIRDebugOutputColorRef[pixel] = value;
        else gIRDebugOutputColorRef[pixel] = float4(outputColor, 1.0);
    }
}


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

import LightSampling;

cbuffer CB
{
    uint gFrameCount;
    uint2 gFrameDim;
}

// Inputs
Texture2D<PackedHitInfo> gVBuffer;
Texture2D<float4> gViewW;

// Outputs
RWTexture2D<float4> gOutputColor;

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

struct ScatterRayData
{
    float3 radiance;  ///< Accumulated outgoing radiance from path.
    float3 thp;       ///< Current path throughput. This is updated at each path vertex.
    uint numBounces;  ///< Path length in number of path segments (0 at origin, 1 at first secondary hit, etc.). Max 2^31.
    float3 origin;    ///< Next path segment origin.
    float3 direction; ///< Next path segment direction.
    float3 normal;
    float pdf;
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
        this.origin = float3(0, 0, 0);
        this.direction = float3(0, 0, 0);
        this.pdf = 1.0;
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

/**
 * Generate a new scatter ray or terminate.
 * @param[in] sd Shading data.
 * @param[in] mi Material instance.
 * @param[in] rayOrigin Ray origin for the new ray.
 * @param[in,out] rayData Ray payload.
 * @return True if the path continues.
 */
bool generateScatterRay(const ShadingData sd, const IMaterialInstance mi, inout ScatterRayData rayData)
{
    // Sample material.
    BSDFSample bsdfSample;
    if (mi.sample(sd, rayData.sg, bsdfSample, kUseImportanceSampling))
    {
        // delta lobes require some caution, store that one is used
        if (bsdfSample.isLobe(LobeType::Delta)) rayData.setDeltaLobe(true);
        else rayData.setDeltaLobe(false);

        // prevent self-intersection
        if (bsdfSample.isLobe(LobeType::Transmission)) rayData.origin = sd.computeRayOrigin(false);
        else rayData.origin = sd.computeRayOrigin(true);
        // update rayData
        rayData.direction = bsdfSample.wo;
        rayData.pdf = bsdfSample.pdf;
        rayData.normal = sd.getOrientedFaceNormal();
        rayData.thp *= bsdfSample.weight;
        if (kUseRR)
        {
            float survival_prob = (kRRProbStartValue * pow(kRRProbReductionFactor, rayData.numBounces));
            if (sampleNext1D(rayData.sg) > survival_prob)
            {
                // rr not survived
                return false;
            }
            rayData.thp /= min(survival_prob, 1.0);
        }
        return any(rayData.thp > 0.f);
    }
    return false;
}

bool handleHit(const HitInfo hit, const Ray ray, inout ScatterRayData rayData)
{
    if (hit.getType() != HitType::Triangle) return false;
    let triangleHit = hit.getTriangleHit();
    let lod = ExplicitLodTextureSampler(0.f);
    ShadingData sd = loadShadingData(triangleHit, ray.dir, lod);
    IMaterialInstance mi = gScene.materials.getMaterialInstance(sd, lod);
    BSDFProperties bsdfProperties = mi.getProperties(sd);
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
        rayData.radiance += rayData.thp * misWeight * bsdfProperties.emission;
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
            // apply MIS only to non-analytic lights
            if (ls.lightType != uint(LightSampleType::Analytic))
            {
                float scatterPdf = mi.evalPdf(sd, ls.dir, kUseImportanceSampling);
                ls.Li *= evalMIS(1, ls.pdf, 1, scatterPdf);
            }

            // apply BSDF
            float3 weight = mi.eval(sd, ls.dir, rayData.sg);
            float3 Lr = weight * ls.Li;
            if (any(Lr > 0.f))
            {
                Ray shadowRay = ls.getVisibilityRay();
                bool visible = traceShadowRay(shadowRay);
                if (visible) rayData.radiance += rayData.thp * Lr;
            }
        }
    }
    // generate scatter ray for the next path segment.
    return (generateScatterRay(sd, mi, rayData));
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
    float t = 0.0f;
    if (rayQuery.traceRay(ray, hit, t, RAY_FLAG_NONE, 0xff))
    {
        return handleHit(hit, ray, rayData);
    }
    else
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
            rayData.radiance += rayData.thp * misWeight * Le;
        }
        return false;
    }
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
float3 tracePath(const uint2 pixel, const uint2 frameDim)
{
    float3 outColor = float3(0.f);
    const HitInfo hit = HitInfo(gVBuffer[pixel]);
    if (hit.isValid() && hit.getType() == HitType::Triangle)
    {
        SampleGenerator sg = SampleGenerator(pixel, gFrameCount);
        ScatterRayData rayData = ScatterRayData(sg);
        if (!handleHit(hit, Ray(gScene.camera.getPosition(), -gViewW[pixel].xyz), rayData)) return rayData.radiance;
        rayData.numBounces++;
        // Follow path into the scene and compute its total contribution.
        while (rayData.numBounces <= kUpperBounceCount)
        {
            if (!traceScatterRay(rayData)) break;
            rayData.numBounces++;
        }
        // Store contribution from scatter ray.
        outColor += rayData.radiance;
        if (kDebugPathLength) return colormapViridis(float(rayData.numBounces) / float(kUpperBounceCount));
    }
    else
    {
        // Background pixel.
        outColor = kUseEnvBackground ? gScene.envMap.eval(-gViewW[pixel].xyz) : kDefaultBackgroundColor;
    }
    return outColor;
}

[numthreads(16, 16, 1)]
void main(uint3 dispatchThreadId: SV_DispatchThreadID)
{
    uint2 pixel = dispatchThreadId.xy;
    if (any(pixel >= gFrameDim)) return;
    printSetPixel(pixel);

    float3 outputColor = tracePath(pixel, gFrameDim);
    gOutputColor[pixel] = float4(outputColor, 1.0f);
}


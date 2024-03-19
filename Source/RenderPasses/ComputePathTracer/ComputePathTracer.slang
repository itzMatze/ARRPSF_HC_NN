#include "Scene/SceneDefines.slangh"
#include "Utils/Math/MathConstants.slangh"

import Scene.Scene;
import Scene.RaytracingInline;
import Scene.Shading;
import Scene.HitInfo;
import Scene.Material.ShadingUtils;
import Utils.Sampling.SampleGenerator;
import Utils.Math.MathHelpers;
import Utils.Geometry.GeometryHelpers;
import Utils.Color.ColorHelpers;
import Utils.Color.ColorMap;
import Utils.Debug.PixelDebug;
import Rendering.Lights.EnvMapSampler;
import Rendering.Lights.EmissiveLightSampler;
import Rendering.Lights.EmissiveLightSamplerHelpers;
import Rendering.Lights.LightHelpers;

cbuffer CB
{
    uint gFrameCount;
    uint2 gFrameDim;
}

// Inputs
Texture2D<PackedHitInfo> gVBuffer;
Texture2D<float4> gViewW; // Optional

// Outputs
RWTexture2D<float4> gOutputColor;

struct PathTracer
{
    // Samplers
    EnvMapSampler envMapSampler;                    ///< Environment map sampler. Only valid when kUseEnvLight == true.
    EmissiveLightSampler emissiveSampler;           ///< Emissive light sampler. Only valid when kUseEmissiveLights == true.
}
ParameterBlock<PathTracer> gPathTracer;

// Static configuration based on defines set from the host.
#define is_valid(name) (is_valid_##name != 0)
static const uint kMaxBounces = MAX_BOUNCES;
static const float kRRProbStartValue = RR_PROB_START_VALUE;
static const float kRRProbReductionFactor = RR_PROB_REDUCTION_FACTOR;
static const bool kDebugPathLength = DEBUG_PATH_LENGTH;
static const uint kPathLengthUpperLimit = PATH_LENGTH_UPPER_LIMIT;
static const bool kComputeDirect = COMPUTE_DIRECT;
static const bool kUseImportanceSampling = USE_IMPORTANCE_SAMPLING;
static const bool kUseAnalyticLights = USE_ANALYTIC_LIGHTS;
static const bool kUseEmissiveLights = USE_EMISSIVE_LIGHTS;
static const bool kUseEnvLight = USE_ENV_LIGHT;
static const bool kUseEnvBackground = USE_ENV_BACKGROUND;
static const float3 kDefaultBackgroundColor = float3(0.0, 0.0, 0.05);
static const float kRayTMax = FLT_MAX;

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

/** Types of samplable lights.
*/
enum class LightSampleType : uint32_t
{
    EnvMap = 0,
    Emissive = 1,
    Analytic = 2
};

/** Describes a light sample.
*/
struct LightSample
{
    float3  Li;         ///< Incident radiance at the shading point (unshadowed). This is already divided by the pdf.
    float   pdf;        ///< Pdf with respect to solid angle at the shading point.
    float3  origin;     ///< Ray origin for visibility evaluation (offseted to avoid self-intersection).
    float   distance;   ///< Ray distance for visibility evaluation (shortened to avoid self-intersection).
    float3  dir;        ///< Ray direction for visibility evaluation (normalized).
    uint    lightType;  ///< Light type this sample comes from (LightType casted to uint).

    Ray getVisibilityRay() { return Ray(origin, dir, 0.f, distance); }
};

/** Describes a path vertex.
*/
struct PathVertex
{
    float3 pos;         ///< Vertex position.
    float3 faceNormal;  ///< Geometry normal at the vertex (zero if not on a surface).
    bool frontFacing;   ///< True if path vertex is on the front-facing side (if on a surface).

    /** Initializes a path vertex.
        \param[in] index Vertex index.
        \param[in] pos Vertex position.
        \param[in] faceNormal Geometry normal.
        \param[in] frontFacing Front-facing flag.
    */
    __init(float3 pos, float3 faceNormal = float3(0.f), bool frontFacing = true)
    {
        this.pos = pos;
        this.faceNormal = faceNormal;
        this.frontFacing = frontFacing;
    }

    /** Get position with offset applied in direction of the geometry normal to avoid self-intersection
        for visibility rays.
        \param[in] rayDir Direction of the visibility ray (does not need to be normalized).
        \return Returns the offseted position.
    */
    float3 getRayOrigin(float3 rayDir)
    {
        return computeRayOrigin(pos, dot(faceNormal, rayDir) >= 0 ? faceNormal : -faceNormal);
    }

    /** Returns the oriented face normal.
        \return Face normal flipped to the same side as the view vector.
    */
    float3 getOrientedFaceNormal()
    {
        return frontFacing ? faceNormal : -faceNormal;
    }
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
    // Balance heuristic
    float q0 = n0 * p0;
    float q1 = n1 * p1;
    return q0 / (q0 + q1);
/*
    switch (MISHeuristic(kMISHeuristic))
    {
    case MISHeuristic::Balance:
    {
        // Balance heuristic
        float q0 = n0 * p0;
        float q1 = n1 * p1;
        return q0 / (q0 + q1);
    }
    case MISHeuristic::PowerTwo:
    {
        // Power two heuristic
        float q0 = (n0 * p0) * (n0 * p0);
        float q1 = (n1 * p1) * (n1 * p1);
        return q0 / (q0 + q1);
    }
    case MISHeuristic::PowerExp:
    {
        // Power exp heuristic
        float q0 = pow(n0 * p0, kMISPowerExponent);
        float q1 = pow(n1 * p1, kMISPowerExponent);
        return q0 / (q0 + q1);
    }
    default:
        return 0.f;
    }
*/
}

/** Generates a light sample on the environment map.
    \param[in] vertex Path vertex.
    \param[in,out] sg Sample generator.
    \param[out] ls Struct describing valid samples.
    \return True if the sample is valid and has nonzero contribution, false otherwise.
*/
bool generateEnvMapSample(const PathVertex vertex, inout SampleGenerator sg, out LightSample ls)
{
    ls = {}; // Default initialization to avoid divergence at returns.

    if (!kUseEnvLight) return false;

    // Sample environment map.
    EnvMapSample lightSample;
    if (!gPathTracer.envMapSampler.sample(sampleNext2D(sg), lightSample)) return false;

    // Setup returned sample.
    ls.Li = lightSample.pdf > 0.f ? lightSample.Le / lightSample.pdf : float3(0);
    ls.pdf = lightSample.pdf;
    ls.origin = vertex.getRayOrigin(lightSample.dir);
    ls.distance = kRayTMax;
    ls.dir = lightSample.dir;

    return any(ls.Li > 0.f);
}

/** Generates a light sample on the emissive geometry.
    \param[in] vertex Path vertex.
    \param[in] upperHemisphere True if only upper hemisphere should be considered.
    \param[in,out] sg Sample generator.
    \param[out] ls Struct describing valid samples.
    \return True if the sample is valid and has nonzero contribution, false otherwise.
*/
bool generateEmissiveSample(const PathVertex vertex, const bool upperHemisphere, inout SampleGenerator sg, out LightSample ls)
{
    ls = {}; // Default initialization to avoid divergence at returns.
    if (!kUseEmissiveLights) return false;

    TriangleLightSample tls;
    if (!gPathTracer.emissiveSampler.sampleLight(vertex.pos, vertex.getOrientedFaceNormal(), upperHemisphere, sg, tls))
    {
        return false;
    }

    // Setup returned sample.
    ls.Li = tls.pdf > 0.f ? tls.Le / tls.pdf : float3(0);
    ls.pdf = tls.pdf;
    // Offset shading and light position to avoid self-intersection.
    float3 lightPos = computeRayOrigin(tls.posW, tls.normalW);
    ls.origin = vertex.getRayOrigin(lightPos - vertex.pos);
    float3 toLight = lightPos - ls.origin;
    ls.distance = length(toLight);
    ls.dir = normalize(toLight);

    return any(ls.Li > 0.f);
}

/** Generates a light sample on the analytic lights.
    \param[in] vertex Path vertex.
    \param[in,out] sg Sample generator.
    \param[out] ls Struct describing valid samples.
    \return True if the sample is valid and has nonzero contribution, false otherwise.
*/
bool generateAnalyticLightSample(const PathVertex vertex, inout SampleGenerator sg, out LightSample ls)
{
    ls = {}; // Default initialization to avoid divergence at returns.

    uint lightCount = gScene.getLightCount();
    if (!kUseAnalyticLights || lightCount == 0) return false;

    // Sample analytic light source selected uniformly from the light list.
    uint lightIndex = min(uint(sampleNext1D(sg) * lightCount), lightCount - 1);

    // Sample local light source.
    AnalyticLightSample lightSample;
    if (!sampleLight(vertex.pos, gScene.getLight(lightIndex), sg, lightSample)) return false;

    // Setup returned sample.
    ls.pdf = lightSample.pdf / lightCount;
    ls.Li = lightSample.Li * lightCount;
    // Offset shading position to avoid self-intersection.
    ls.origin = vertex.getRayOrigin(lightSample.dir);
    // Analytic lights do not currently have a geometric representation in the scene.
    // Do not worry about adjusting the ray length to avoid self-intersections at the light.
    ls.distance = lightSample.distance;
    ls.dir = lightSample.dir;

    return any(ls.Li > 0.f);
}

/** Return the probabilities for selecting different light types.
    \param[out] p Probabilities.
*/
void getLightTypeSelectionProbabilities(out float p[3])
{
    // Set relative probabilities of the different sampling techniques.
    p[uint(LightSampleType::EnvMap)] = kUseEnvLight ? 1.f : 0.f;
    p[uint(LightSampleType::Emissive)] = kUseEmissiveLights ? 1.f : 0.f;
    p[uint(LightSampleType::Analytic)] = kUseAnalyticLights ? 1.f : 0.f;

    // Normalize probabilities. Early out if zero.
    float sum = p[0] + p[1] + p[2];
    if (sum == 0.f) return;

    float invSum = 1.f / sum;
    p[0] *= invSum;
    p[1] *= invSum;
    p[2] *= invSum;
}

float getEnvMapSelectionProbability()   { float p[3]; getLightTypeSelectionProbabilities(p); return p[uint(LightSampleType::EnvMap)]; }
float getEmissiveSelectionProbability() { float p[3]; getLightTypeSelectionProbabilities(p); return p[uint(LightSampleType::Emissive)]; }
float getAnalyicSelectionProbability()  { float p[3]; getLightTypeSelectionProbabilities(p); return p[uint(LightSampleType::Analytic)]; }

/** Select a light type for sampling.
    \param[out] lightType Selected light type.
    \param[out] pdf Probability for selected type.
    \param[in,out] sg Sample generator.
    \return Return true if selection is valid.
*/
bool selectLightType(out uint lightType, out float pdf, inout SampleGenerator sg)
{
    float p[3];
    getLightTypeSelectionProbabilities(p);

    float u = sampleNext1D(sg);

    [unroll]
    for (lightType = 0; lightType < 3; ++lightType)
    {
        if (u < p[lightType])
        {
            pdf = p[lightType];
            return true;
        }
        u -= p[lightType];
    }

    lightType = {};
    pdf = {};

    return false;
}

/** Samples a light source in the scene.
    This function first stochastically selects a type of light source to sample,
    and then calls that the sampling function for the chosen light type.
    \param[in] vertex Path vertex.
    \param[in] sampleUpperHemisphere True if the upper hemisphere should be sampled.
    \param[in] sampleLowerHemisphere True if the lower hemisphere should be sampled.
    \param[in,out] sg Sample generator.
    \param[out] ls Struct describing valid samples.
    \return True if the sample is valid and has nonzero contribution, false otherwise.
*/
bool generateLightSample(const PathVertex vertex, const bool sampleUpperHemisphere, const bool sampleLowerHemisphere, inout SampleGenerator sg, out LightSample ls)
{
    ls = {};

    uint lightType;
    float selectionPdf;
    if (!selectLightType(lightType, selectionPdf, sg)) return false;

    bool valid = false;
    if (kUseEnvLight && lightType == uint(LightSampleType::EnvMap)) valid = generateEnvMapSample(vertex, sg, ls);
    if (kUseEmissiveLights && lightType == uint(LightSampleType::Emissive))
    {
        // Emissive light samplers have an option to exclusively sample the upper hemisphere.
        bool upperHemisphere = sampleUpperHemisphere && !sampleLowerHemisphere;
        valid = generateEmissiveSample(vertex, upperHemisphere, sg, ls);
    }
    if (kUseAnalyticLights && lightType == uint(LightSampleType::Analytic))
    {
        valid = generateAnalyticLightSample(vertex, sg, ls);
    }
    if (!valid) return false;

    // Reject samples in non-requested hemispheres.
    float NdotL = dot(vertex.getOrientedFaceNormal(), ls.dir);
    if ((!sampleUpperHemisphere && NdotL >= -kMinCosTheta) || (!sampleLowerHemisphere && NdotL <= kMinCosTheta))
        return false;

    // Account for light type selection.
    ls.lightType = lightType;
    ls.pdf *= selectionPdf;
    ls.Li /= selectionPdf;

    return true;
}

/**
 * Traces a shadow ray towards a light source.
 * @param[in] origin Ray origin for the shadow ray.
 * @param[in] dir Direction from shading point towards the light source (normalized).
 * @param[in] distance Distance to the light source.
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
        // Handle delta events.
        if (bsdfSample.isLobe(LobeType::Delta)) rayData.setDeltaLobe(true);
        else rayData.setDeltaLobe(false);

        rayData.origin = sd.computeRayOrigin(true);
        if (bsdfSample.isLobe(LobeType::Transmission))
        {
            rayData.origin = sd.computeRayOrigin(false);
        }
        rayData.direction = bsdfSample.wo;
        rayData.pdf = bsdfSample.pdf;
        rayData.normal = sd.getOrientedFaceNormal();
        rayData.thp *= bsdfSample.weight;
        float survival_prob = (kRRProbStartValue * pow(kRRProbReductionFactor, rayData.numBounces));
        if (sampleNext1D(rayData.sg) < survival_prob)
        {
            // rr survived
            rayData.thp /= min(survival_prob, 1.0);
            return any(rayData.thp > 0.f);
        }
    }
    return false;
}

bool handleHit(const HitInfo hit, const Ray ray, inout ScatterRayData rayData)
{
    bool kUseNEE = true;
    bool kUseMIS = true;
    if (hit.getType() != HitType::Triangle) return false;
    let triangleHit = hit.getTriangleHit();
    let lod = ExplicitLodTextureSampler(0.f);
    ShadingData sd = loadShadingData(triangleHit, ray.dir, lod);
    IMaterialInstance mi = gScene.materials.getMaterialInstance(sd, lod);
    BSDFProperties bsdfProperties = mi.getProperties(sd);
#if 1
    // light hit by chance
    bool computeEmissive = rayData.numBounces == 0 || kUseEmissiveLights && (!kUseNEE || kUseMIS || !rayData.isLightSampled() || rayData.isDeltaLobe());
    if (computeEmissive && any(bsdfProperties.emission > 0.f))
    {
        float misWeight = 1.0;
        if (kUseEmissiveLights && kUseNEE && kUseMIS && rayData.numBounces > 0 && rayData.isLightSampled() && !rayData.isDeltaLobe())
        {
            TriangleLightHit lightHit;
            lightHit.triangleIndex = gScene.lightCollection.getTriangleIndex(triangleHit.instanceID, triangleHit.primitiveIndex);
            lightHit.posW = sd.posW;
            lightHit.normalW = sd.getOrientedFaceNormal();
            // Evaluate PDF at the hit, had it been generated with light sampling.
            // Emissive light samplers have an option to exclusively sample the upper hemisphere.
            bool upperHemisphere = rayData.isLightSampledUpper() && !rayData.isLightSampledLower();
            float lightPdf = getEmissiveSelectionProbability() * gPathTracer.emissiveSampler.evalPdf(rayData.origin, rayData.normal, upperHemisphere, lightHit);
            // Compute MIS weight by combining this with BSDF sampling.
            // Note we can assume path.pdf > 0.f since we shouldn't have got here otherwise.
            misWeight = evalMIS(1, rayData.pdf, 1, lightPdf);
        }
        // Accumulate emitted radiance weighted by path throughput and MIS weight.
        rayData.radiance += rayData.thp * misWeight * bsdfProperties.emission;
    }
    // nee
    LightSample ls;
    // Setup path vertex.
    PathVertex vertex = PathVertex(sd.posW, sd.faceN, sd.frontFacing);
    const uint lobeTypes = mi.getLobeTypes(sd);
    const bool hasNonDeltaLobes = (lobeTypes & uint(LobeType::NonDelta)) != 0;

    // Check if we should apply NEE.
    const bool applyNEE = kUseNEE && hasNonDeltaLobes;

    rayData.setLightSampled(false, false);
    if (applyNEE && ((rayData.numBounces + 1) <= kMaxBounces))
    {
        bool sampleUpperHemisphere = ((lobeTypes & uint(LobeType::NonDeltaReflection)) != 0);
        bool sampleLowerHemisphere = ((lobeTypes & uint(LobeType::NonDeltaTransmission)) != 0);
        // Sample a light.
        bool validSample = generateLightSample(vertex, sampleUpperHemisphere, sampleLowerHemisphere, rayData.sg, ls);
        rayData.setLightSampled(sampleUpperHemisphere, sampleLowerHemisphere);
        if (validSample)
        {
            // Apply MIS weight.
            if (ls.lightType != uint(LightSampleType::Analytic))
            {
                float scatterPdf = mi.evalPdf(sd, ls.dir, kUseImportanceSampling);
                ls.Li *= evalMIS(1, ls.pdf, 1, scatterPdf);
            }

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
#else
    // probably obsolete TODO
    if (kUseEmissiveLights && (kComputeDirect || rayData.numBounces > 0))
    {
        rayData.radiance += rayData.thp * bsdfProperties.emission;
    }
#endif
    // Generate scatter ray for the next path segment.
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
        bool computeEnv = kUseEnvLight && (/*!kUseNEE || kUseMIS || */true || !rayData.isLightSampled() || rayData.isDeltaLobe());
        if (computeEnv)
        {
            float misWeight = 1.f;
            if (/*kUseNEE && kUseMIS && */rayData.isLightSampled() && !rayData.isDeltaLobe())
            {
                // If NEE and MIS are enabled, and we've already sampled the env map,
                // then we need to evaluate the MIS weight here to account for the remaining contribution.

                // Evaluate PDF, had it been generated with light sampling.
                float lightPdf = getEnvMapSelectionProbability() * gPathTracer.envMapSampler.evalPdf(rayData.direction);

                // Compute MIS weight by combining this with BSDF sampling.
                // Note we can assume path.pdf > 0.f since we shouldn't have got here otherwise.
                misWeight = evalMIS(1, rayData.pdf, 1, lightPdf);
            }

            float3 Le = gPathTracer.envMapSampler.eval(rayData.direction);
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
        if (kComputeDirect)
        {
            if (!handleHit(hit, Ray(gScene.camera.getPosition(), -gViewW[pixel].xyz), rayData)) return rayData.radiance;
        }
        rayData.numBounces++;
        // Follow path into the scene and compute its total contribution.
        while (rayData.numBounces <= kMaxBounces)
        {
            if (!traceScatterRay(rayData)) break;
            rayData.numBounces++;
        }
        // Store contribution from scatter ray.
        outColor += rayData.radiance;
        if (kDebugPathLength) return colormapViridis(float(rayData.numBounces) / float(kPathLengthUpperLimit));
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


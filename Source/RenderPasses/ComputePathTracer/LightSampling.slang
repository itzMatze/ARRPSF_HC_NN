#include "Utils/Math/MathConstants.slangh"

import Scene.RaytracingInline;
import Rendering.Lights.EnvMapSampler;
import Rendering.Lights.EmissiveLightSampler;
import Rendering.Lights.EmissiveLightSamplerHelpers;
import Rendering.Lights.LightHelpers;
import Utils.Sampling.SampleGenerator;
import Utils.Geometry.GeometryHelpers;

static const bool kUseAnalyticLights = USE_ANALYTIC_LIGHTS;
static const bool kUseEmissiveLights = USE_EMISSIVE_LIGHTS;
static const bool kUseEnvLight = USE_ENV_LIGHT;
static const float kRayTMax = FLT_MAX;

struct Sampler
{
    // Samplers
    EnvMapSampler envMapSampler;                    ///< Environment map sampler. Only valid when kUseEnvLight == true.
    EmissiveLightSampler emissiveSampler;           ///< Emissive light sampler. Only valid when kUseEmissiveLights == true.
}
ParameterBlock<Sampler> gSampler;

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
}

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
    if (!gSampler.envMapSampler.sample(sampleNext2D(sg), lightSample)) return false;
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
    if (!gSampler.emissiveSampler.sampleLight(vertex.pos, vertex.getOrientedFaceNormal(), upperHemisphere, sg, tls))
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
    if ((!sampleUpperHemisphere && NdotL >= -kMinCosTheta) || (!sampleLowerHemisphere && NdotL <= kMinCosTheta)) return false;
    // Account for light type selection.
    ls.lightType = lightType;
    ls.pdf *= selectionPdf;
    ls.Li /= selectionPdf;
    return true;
}


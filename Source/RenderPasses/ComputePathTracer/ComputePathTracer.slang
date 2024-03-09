import Scene.Shading;
import Scene.HitInfo;
import Utils.Sampling.SampleGenerator;
import Scene.Material.ShadingUtils;
import Utils.Math.MathHelpers;
import Scene.Scene;
import Scene.RaytracingInline;
import Utils.Color.ColorHelpers;
import Utils.Debug.PixelDebug;

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

// Static configuration based on defines set from the host.
#define is_valid(name) (is_valid_##name != 0)
static const uint kMaxBounces = MAX_BOUNCES;
static const bool kComputeDirect = COMPUTE_DIRECT;
static const bool kUseImportanceSampling = USE_IMPORTANCE_SAMPLING;
static const bool kTest = TEST;
static const bool kUseAnalyticLights = USE_ANALYTIC_LIGHTS;
static const bool kUseEmissiveLights = USE_EMISSIVE_LIGHTS;
static const bool kUseEnvLight = USE_ENV_LIGHT;
static const bool kUseEnvBackground = USE_ENV_BACKGROUND;
static const float3 kDefaultBackgroundColor = float3(0, 0, 0);
//static const float kRayTMax = FLT_MAX;

/**
 * Setup ShadingData based on loaded vertex/material attributes for a hit point.
 * @param[in] hit Hit information.
 * @param[in] rayOrigin Ray origin.
 * @param[in] rayDir Normalized ray direction.
 * @param[in] lod Method for computing texture level-of-detail.
 * @return ShadingData struct.
 */
ShadingData loadShadingData(const HitInfo hit, const float3 rayOrigin, const float3 rayDir, const ITextureSampler lod)
{
    VertexData v = {};
    uint materialID = {};

    if (hit.getType() == HitType::Triangle)
    {
        const TriangleHit triangleHit = hit.getTriangleHit();
        v = gScene.getVertexData(triangleHit);
        materialID = gScene.getMaterialID(triangleHit.instanceID);
    }

    ShadingData sd = gScene.materials.prepareShadingData(v, materialID, -rayDir, lod);
    return sd;
}

[numthreads(16, 16, 1)]
void main(uint3 dispatchThreadId: SV_DispatchThreadID)
{
    uint2 pixel = dispatchThreadId.xy;
    if (any(pixel >= gFrameDim)) return;

    float3 color = float3(0.0f);
    float3 emission = float3(0.0f);
    let lod = ExplicitLodTextureSampler(0.f); // TODO: Implement texture level-of-detail.
    const HitInfo primary_hit = HitInfo(gVBuffer[pixel]);
    if (primary_hit.isValid() && primary_hit.getType() == HitType::Triangle)
    {
        const TriangleHit triangleHit = primary_hit.getTriangleHit();
        ShadingData sd = loadShadingData(primary_hit, gScene.camera.getPosition(), -gViewW[pixel].xyz, lod);
        if (kTest)
        {
            IMaterialInstance mi = gScene.materials.getMaterialInstance(sd, lod);
            SampleGenerator sg = SampleGenerator(pixel, gFrameCount);
            color = mi.getProperties(sd).diffuseReflectionAlbedo;
            emission = mi.getProperties(sd).emission;
            for (uint i = 0; i < 10; ++i)
            {
                float3 dir = normalize(sampleNext3D(sg) - 0.5);
                SceneRayQuery<1> rayQuery;
                const Ray ray = Ray(sd.computeRayOrigin(), dir, 0.0f, 1000.0f);
                HitInfo hit;
                float t = 0.0f;
                if (rayQuery.traceRay(ray, hit, t, RAY_FLAG_NONE, 0xff))
                {
                    sd = loadShadingData(hit, ray.origin, ray.dir, lod);
                    mi = gScene.materials.getMaterialInstance(sd, lod);
                    emission += color * mi.getProperties(sd).emission;
                    color *= mi.getProperties(sd).diffuseReflectionAlbedo;
                }
                else
                {
                    break;
                }
            }
        }
        else
        {
            emission = float3(triangleHit.getBarycentricWeights());
        }
    }
    else
    {
        emission = float3(0.0f, 0.0f, 0.05f);
    }
    gOutputColor[pixel] = float4(emission, 1.0f);
}

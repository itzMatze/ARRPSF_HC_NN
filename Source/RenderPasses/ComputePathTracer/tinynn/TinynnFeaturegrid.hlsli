#ifndef _SRENDERER_ADDON_HALF_TINYNN_FEATUREGRID_HLSLI_HEADER_
#define _SRENDERER_ADDON_HALF_TINYNN_FEATUREGRID_HLSLI_HEADER_

#include "TinynnHalfLinear.hlsli"

struct FeatureGrid2DIndex {
    uint2 cellId;
    float2 weights;
    __init(uint2 frameDim, int2 pixelId, int2 feature_grid) {
        const float2 cellSize = float2(float(frameDim.x) / float(feature_grid.x), float(frameDim.y) / float(feature_grid.y));
        this.cellId = uint2(floor(pixelId / cellSize));
        this.weights = (float2(pixelId) / cellSize) - float2(cellId);
    }
};

[Differentiable]
HalfFeature<32> computeInterpolatedFeature(
    no_diff TensorView featureGrid,
    no_diff FeatureGrid2DIndex index,
    no_diff float3 pos,
    no_diff float3 dir,
    no_diff float3 albedo
) {
    const uint2 cellId = index.cellId;
    const float2 weights = index.weights;
    HalfFeature<32> feature;
    [ForceUnroll]
    for (uint i = 0; i < 32; i++) feature.vals[i] = 1.0h;
    feature.vals[0] = float16_t(pos.x);
    feature.vals[1] = float16_t(pos.y);
    feature.vals[2] = float16_t(pos.z);
    [ForceUnroll]
    for (uint i = 0; i < 4; i++)
    {
        feature.vals[(i * 3) + 3] = sin(float16_t(pos.x) * float16_t(3.1415926f * pow(2.0, (i * 2.0))));
        feature.vals[(i * 3) + 4] = sin(float16_t(pos.y) * float16_t(3.1415926f * pow(2.0, (i * 2.0))));
        feature.vals[(i * 3) + 5] = sin(float16_t(pos.z) * float16_t(3.1415926f * pow(2.0, (i * 2.0))));
    }
    feature.vals[15] = float16_t(dir.x);
    feature.vals[16] = float16_t(dir.y);
    feature.vals[17] = float16_t(dir.z);
    [ForceUnroll]
    for (uint i = 0; i < 3; i++)
    {
        feature.vals[(i * 3) + 18] = sin(float16_t(dir.x) * float16_t(3.1415926f * pow(2.0, (i * 2.0))));
        feature.vals[(i * 3) + 19] = sin(float16_t(dir.y) * float16_t(3.1415926f * pow(2.0, (i * 2.0))));
        feature.vals[(i * 3) + 20] = sin(float16_t(dir.z) * float16_t(3.1415926f * pow(2.0, (i * 2.0))));
    }
    //feature.vals[29] = float16_t(albedo.x);
    //feature.vals[30] = float16_t(albedo.y);
    //feature.vals[31] = float16_t(albedo.z);
    return feature;
}

#endif // !_SRENDERER_ADDON_HALF_TINYNN_FEATUREGRID_HLSLI_HEADER_


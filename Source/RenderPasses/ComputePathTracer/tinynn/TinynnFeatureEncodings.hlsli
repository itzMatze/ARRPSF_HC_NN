#ifndef _SRENDERER_ADDON_HALF_TINYNN_FEATUREGRID_HLSLI_HEADER_
#define _SRENDERER_ADDON_HALF_TINYNN_FEATUREGRID_HLSLI_HEADER_

import Utils.Debug.PixelDebug;

#include "TinynnHalfLinear.hlsli"
#include "HashEncHashGridCommon.slang"

[Differentiable]
HalfFeature<32> computeFreqEncFeature(
    no_diff float3 pos,
    no_diff float3 dir,
) {
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
    return feature;
}

[Differentiable]
HalfFeature<32> computeHashEncFeature(
    no_diff float3 pos,
    no_diff float3 dir,
    no_diff float3 normal,
    no_diff FeatureHashGridData featureHashGrid,
) {
    HalfFeature<32> feature;
    [ForceUnroll]
    for (uint i = 0; i < 32; i++) feature.vals[i] = 1.0h;
    [ForceUnroll]
    for (uint i = 0; i < 8; i++)
    {
        uint idx = no_diff featureHashGrid.FindEntry(pos, normal, i);
        feature.vals[i * 2] = float16_t(featureHashGrid.dataView.load_prim(idx));
        feature.vals[i * 2 + 1] = float16_t(featureHashGrid.dataView.load_prim(idx + 1));
    }
    feature.vals[16] = float16_t(dir.x);
    feature.vals[17] = float16_t(dir.y);
    feature.vals[18] = float16_t(dir.z);
    [ForceUnroll]
    for (uint i = 0; i < 2; i++)
    {
        feature.vals[(i * 3) + 19] = sin(float16_t(dir.x) * float16_t(3.1415926f * pow(2.0, (i * 2.0))));
        feature.vals[(i * 3) + 20] = sin(float16_t(dir.y) * float16_t(3.1415926f * pow(2.0, (i * 2.0))));
        feature.vals[(i * 3) + 21] = sin(float16_t(dir.z) * float16_t(3.1415926f * pow(2.0, (i * 2.0))));
    }
    feature.vals[25] = float16_t(pos.x);
    feature.vals[26] = float16_t(pos.y);
    feature.vals[27] = float16_t(pos.z);
    [ForceUnroll]
    for (uint i = 0; i < 1; i++)
    {
        feature.vals[(i * 3) + 28] = sin(float16_t(pos.x) * float16_t(3.1415926f * pow(2.0, (i * 2.0))));
        feature.vals[(i * 3) + 29] = sin(float16_t(pos.y) * float16_t(3.1415926f * pow(2.0, (i * 2.0))));
        feature.vals[(i * 3) + 30] = sin(float16_t(pos.z) * float16_t(3.1415926f * pow(2.0, (i * 2.0))));
    }
    return feature;
}
#endif // !_SRENDERER_ADDON_HALF_TINYNN_FEATUREGRID_HLSLI_HEADER_


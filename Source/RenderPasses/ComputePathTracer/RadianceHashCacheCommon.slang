import Utils.Debug.PixelDebug;
import RadianceHashCacheHashGridCommon;

#if R_HC_UPDATE || R_HC_QUERY
static const uint kHashCacheLevelTrainingSpread = 3;
static const uint kHashCachePropagationDepth = 8;
static const float kHashCacheRadianceScale = 1e4f;

RWByteAddressBuffer gRHCVoxelDataBufferPrev;
RWByteAddressBuffer gRHCVoxelDataBuffer;

struct HashCacheVoxelData
{
    float3 radiance = float3(0.0);
    uint sampleNum = 0;
}

HashCacheVoxelData hashCacheGetVoxelData(bool usePrev, uint idx)
{
    if (idx == kHashGridInvalidIdx) return HashCacheVoxelData();
    uint4 data;
    if (usePrev) data = gRHCVoxelDataBufferPrev.Load4(idx * 16);
    else data = gRHCVoxelDataBuffer.Load4(idx * 16);
    HashCacheVoxelData voxelData;
    voxelData.radiance = data.xyz / kHashCacheRadianceScale;
    voxelData.sampleNum = data.w;
    return voxelData;
}

void hashCacheSetVoxelData(bool usePrev, uint idx, HashCacheVoxelData data)
{
    uint4 voxelData = uint4(data.radiance * kHashCacheRadianceScale, data.sampleNum);
    if (usePrev) gRHCVoxelDataBufferPrev.Store4(idx * 16, voxelData);
    else gRHCVoxelDataBuffer.Store4(idx * 16, voxelData);
}

void hashCacheAddVoxelData(uint idx[kHashCacheLevelTrainingSpread], float3 value, bool newSample)
{
    uint3 voxelData = uint3(value * kHashCacheRadianceScale);
    for (uint i = 0; i < kHashCacheLevelTrainingSpread; i++)
    {
        if (idx[i] == kHashGridInvalidIdx) continue;
        if (value.x > 0.0) gRHCVoxelDataBuffer.InterlockedAdd(idx[i] * 16, voxelData.x);
        if (value.y > 0.0) gRHCVoxelDataBuffer.InterlockedAdd(idx[i] * 16 + 4, voxelData.y);
        if (value.z > 0.0) gRHCVoxelDataBuffer.InterlockedAdd(idx[i] * 16 + 8, voxelData.z);
        if (newSample) gRHCVoxelDataBuffer.InterlockedAdd(idx[i] * 16 + 12, 1);
    }
}

struct HashCacheState
{
    HashMapData hashMapData;
#if R_HC_UPDATE
    uint voxelIndices[kHashCachePropagationDepth][kHashCacheLevelTrainingSpread];
    float3 sampleWeights[kHashCachePropagationDepth];
    uint pathLength;
#endif // R_HC_UPDATE
};

struct HashCacheHitData
{
    float distance;
    float3 positionWorld;
    float3 normalWorld;
};

void hashCacheUpdateCache(inout HashCacheState hashCacheState, float3 radiance, uint startIdx)
{
#if R_HC_UPDATE
    for (int i = startIdx; i < hashCacheState.pathLength; ++i)
    {
        radiance *= hashCacheState.sampleWeights[i];
        hashCacheAddVoxelData(hashCacheState.voxelIndices[i], radiance, false);
    }
#endif // R_HC_UPDATE
}

void hashCacheUpdateMiss(inout HashCacheState hashCacheState, float3 radiance)
{
    hashCacheUpdateCache(hashCacheState, radiance, 0);
}

void hashCacheUpdateHit(inout HashCacheState hashCacheState, HashCacheHitData hashCacheHitData, float3 radiance)
{
#if R_HC_UPDATE
    for (uint i = hashCacheState.pathLength; i > 0; --i)
    {
        hashCacheState.voxelIndices[i] = hashCacheState.voxelIndices[i - 1];
        hashCacheState.sampleWeights[i] = hashCacheState.sampleWeights[i - 1];
    }
    for (int i = 0; i < kHashCacheLevelTrainingSpread; i++)
    {
        hashCacheState.voxelIndices[0][i] = hashCacheState.hashMapData.InsertEntry(hashCacheHitData.distance, hashCacheHitData.positionWorld, hashCacheHitData.normalWorld, i - (kHashCacheLevelTrainingSpread / 2));
    }
    hashCacheState.pathLength += 1;
    hashCacheState.pathLength = min(hashCacheState.pathLength, kHashCachePropagationDepth - 1);
    hashCacheAddVoxelData(hashCacheState.voxelIndices[0], radiance, true);
    hashCacheUpdateCache(hashCacheState, radiance, 1);
#endif // R_HC_UPDATE
}

void hashCacheSetThroughput(inout HashCacheState hashCacheState, float3 throughput)
{
#if R_HC_UPDATE
    hashCacheState.sampleWeights[0] = throughput;
#endif // R_HC_UPDATE
}

bool hashCacheGetCachedRadiance(inout HashCacheState hashCacheState, HashCacheHitData hashCacheHitData, out float3 radiance, bool debug = false)
{
    radiance = 0;
    if (debug)
    {
        radiance = hashCacheState.hashMapData.HashGridDebugColoredHash(hashCacheHitData.distance, hashCacheHitData.positionWorld);
        return true;
    }
    uint idx = hashCacheState.hashMapData.FindEntry(hashCacheHitData.distance, hashCacheHitData.positionWorld, hashCacheHitData.normalWorld);
    if (idx == kHashGridInvalidIdx) return false;
    HashCacheVoxelData voxelData = hashCacheGetVoxelData(false, idx);
    if (voxelData.sampleNum > 0)
    {
        radiance = voxelData.radiance;
        return true;
    }
    return false;
}

void hashCacheCombine(uint idx)
{
    HashCacheVoxelData voxelData = hashCacheGetVoxelData(false, idx);
    HashCacheVoxelData voxelDataPrev = hashCacheGetVoxelData(true, idx);
    uint newSampleNum = voxelData.sampleNum - voxelDataPrev.sampleNum;
    if (newSampleNum == 0)
    {
        voxelData.radiance = voxelDataPrev.radiance;
    }
    else
    {
        voxelData.radiance /= float(newSampleNum);
        if (voxelDataPrev.sampleNum > 0)
        {
            float weight = min(newSampleNum * 0.0005, 0.025);
            voxelData.radiance = (1 - weight) * voxelDataPrev.radiance + weight * voxelData.radiance;
        }
    }
    if (voxelData.sampleNum > 0 || voxelDataPrev.sampleNum > 0)
    {
        hashCacheSetVoxelData(false, idx, voxelData);
        voxelData.radiance = float3(0.0);
        hashCacheSetVoxelData(true, idx, voxelData);
    }
}
#endif // R_HC_UPDATE || R_HC_QUERY


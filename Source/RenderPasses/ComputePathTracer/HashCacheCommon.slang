import Utils.Debug.PixelDebug;
import HashGridCommon;

#if HC_UPDATE || HC_QUERY
static const uint kHashCachePropagationDepth = 8;
static const float kHashCacheRadianceScale = 1e4f;

RWByteAddressBuffer gHashCacheVoxelDataBufferPrev;
RWByteAddressBuffer gHashCacheVoxelDataBuffer;

struct HashCacheVoxelData
{
    float3 radiance = float3(0.0);
    uint sampleNum = 0;
}

HashCacheVoxelData hashCacheGetVoxelData(bool usePrev, uint idx)
{
    if (idx == kHashGridInvalidIdx) return HashCacheVoxelData();
    uint4 data;
    if (usePrev) data = gHashCacheVoxelDataBufferPrev.Load4(idx * 16);
    else data = gHashCacheVoxelDataBuffer.Load4(idx * 16);
    HashCacheVoxelData voxelData;
    voxelData.radiance = data.xyz / kHashCacheRadianceScale;
    voxelData.sampleNum = data.w;
    return voxelData;
}

void hashCacheSetVoxelData(bool usePrev, uint idx, HashCacheVoxelData data)
{
    uint4 voxelData = uint4(data.radiance * kHashCacheRadianceScale, data.sampleNum);
    if (usePrev) gHashCacheVoxelDataBufferPrev.Store4(idx * 16, voxelData);
    else gHashCacheVoxelDataBuffer.Store4(idx * 16, voxelData);
}

void hashCacheAddVoxelData(uint idx, float3 value, bool newSample)
{
    if (idx == kHashGridInvalidIdx) return;
    uint3 voxelData = uint3(value * kHashCacheRadianceScale);
    if (value.x > 0.0) gHashCacheVoxelDataBuffer.InterlockedAdd(idx * 16, voxelData.x);
    if (value.y > 0.0) gHashCacheVoxelDataBuffer.InterlockedAdd(idx * 16 + 4, voxelData.y);
    if (value.z > 0.0) gHashCacheVoxelDataBuffer.InterlockedAdd(idx * 16 + 8, voxelData.z);
    if (newSample) gHashCacheVoxelDataBuffer.InterlockedAdd(idx * 16 + 12, 1);
}

struct HashCacheState
{
    GridParameters gridParameters;
    HashMapData hashMapData;
#if HC_UPDATE
    uint voxelIndices[kHashCachePropagationDepth];
    float3 sampleWeights[kHashCachePropagationDepth];
    uint pathLength;
#endif // HC_UPDATE
};

struct HashCacheHitData
{
    float3 positionWorld;
    float3 normalWorld;
};

void hashCacheUpdateCache(inout HashCacheState hashCacheState, float3 radiance, uint startIdx)
{
#if HC_UPDATE
    for (int i = startIdx; i < hashCacheState.pathLength; ++i)
    {
        radiance *= hashCacheState.sampleWeights[i];
        hashCacheAddVoxelData(hashCacheState.voxelIndices[i], radiance, false);
    }
#endif // HC_UPDATE
}

void hashCacheUpdateMiss(inout HashCacheState hashCacheState, float3 radiance)
{
    hashCacheUpdateCache(hashCacheState, radiance, 0);
}

void hashCacheUpdateHit(inout HashCacheState hashCacheState, HashCacheHitData hashCacheHitData, float3 radiance)
{
#if HC_UPDATE
    for (uint i = hashCacheState.pathLength; i > 0; --i)
    {
        hashCacheState.voxelIndices[i] = hashCacheState.voxelIndices[i - 1];
        hashCacheState.sampleWeights[i] = hashCacheState.sampleWeights[i - 1];
    }
    hashCacheState.voxelIndices[0] = hashCacheState.hashMapData.InsertEntry(hashCacheHitData.positionWorld, hashCacheHitData.normalWorld, hashCacheState.gridParameters);
    hashCacheState.pathLength += 1;
    hashCacheState.pathLength = min(hashCacheState.pathLength, kHashCachePropagationDepth - 1);
    hashCacheAddVoxelData(hashCacheState.voxelIndices[0], radiance, true);
    hashCacheUpdateCache(hashCacheState, radiance, 1);
#endif // HC_UPDATE
}

void hashCacheSetThroughput(inout HashCacheState hashCacheState, float3 throughput)
{
#if HC_UPDATE
    hashCacheState.sampleWeights[0] = throughput;
#endif // HC_UPDATE
}

bool hashCacheGetCachedRadiance(inout HashCacheState hashCacheState, HashCacheHitData hashCacheHitData, out float3 radiance, bool debug = false)
{
    radiance = 0;
    if (debug)
    {
        radiance = HashGridDebugColoredHash(hashCacheHitData.positionWorld, hashCacheState.gridParameters);
        return true;
    }
    uint idx = hashCacheState.hashMapData.FindEntry(hashCacheHitData.positionWorld, hashCacheHitData.normalWorld, hashCacheState.gridParameters);
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
            float weight = min(newSampleNum * 0.002, 0.1);
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
#endif // HC_UPDATE || HC_QUERY


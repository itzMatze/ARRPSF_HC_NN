import Utils.Debug.PixelDebug;
import RadianceHashCacheHashGridCommon;

#if HC_UPDATE || HC_QUERY
namespace hc
{
static const uint kHashCacheLevelTrainingSpread = 3;
static const uint kHashCachePropagationDepth = 8;
// prevent overflow by only counting samples to this limit
static const uint kMaxSampleCount = 65536;

RWByteAddressBuffer gHCVoxelDataBufferPrev;
RWByteAddressBuffer gHCVoxelDataBuffer;

struct HashCacheVoxelData
{
    float3 radiance = float3(0.0);
    uint sampleNum = 0;
}
static const uint sizeofHashCacheVoxelData = 16;

HashCacheVoxelData hashCacheGetVoxelData(bool usePrev, uint idx)
{
    HashCacheVoxelData voxelData;
    if (idx == kHashGridInvalidIdx) return HashCacheVoxelData();
    if (usePrev) voxelData = gHCVoxelDataBufferPrev.Load<HashCacheVoxelData>(idx * sizeofHashCacheVoxelData);
    else voxelData = gHCVoxelDataBuffer.Load<HashCacheVoxelData>(idx * sizeofHashCacheVoxelData);
    return voxelData;
}

void hashCacheResetVoxelData(uint idx)
{
    HashCacheVoxelData data = HashCacheVoxelData();
    gHCVoxelDataBuffer.Store(idx * sizeofHashCacheVoxelData, data);
    gHCVoxelDataBufferPrev.Store(idx * sizeofHashCacheVoxelData, data);
}

void hashCacheSetVoxelData(bool usePrev, uint idx, HashCacheVoxelData data)
{
    if (usePrev) gHCVoxelDataBufferPrev.Store(idx * sizeofHashCacheVoxelData, data);
    else gHCVoxelDataBuffer.Store(idx * sizeofHashCacheVoxelData, data);
}

void hashCacheAddVoxelData(uint idx[kHashCacheLevelTrainingSpread], float3 value, bool newSample)
{
    for (uint i = 0; i < kHashCacheLevelTrainingSpread; i++)
    {
        if (idx[i] == kHashGridInvalidIdx) continue;
        if (value.x > 0.0) gHCVoxelDataBuffer.InterlockedAddF32(idx[i] * sizeofHashCacheVoxelData, value.x);
        if (value.y > 0.0) gHCVoxelDataBuffer.InterlockedAddF32(idx[i] * sizeofHashCacheVoxelData + 4, value.y);
        if (value.z > 0.0) gHCVoxelDataBuffer.InterlockedAddF32(idx[i] * sizeofHashCacheVoxelData + 8, value.z);
        if (newSample) gHCVoxelDataBuffer.InterlockedAdd(idx[i] * sizeofHashCacheVoxelData + 12, 1);
    }
}

struct HashCacheState
{
    HashMapData hashMapData;
#if HC_UPDATE
    uint voxelIndices[kHashCachePropagationDepth][kHashCacheLevelTrainingSpread];
    float3 sampleWeights[kHashCachePropagationDepth];
    uint pathLength;
#endif // HC_UPDATE
}

struct HashCacheHitData
{
    float distance;
    float3 positionWorld;
    float3 normalWorld;
    float3 direction;
}

void hashCacheUpdateCache(inout HashCacheState hashCacheState, float3 radiance)
{
#if HC_UPDATE
    for (int i = 0; i < hashCacheState.pathLength; ++i)
    {
#if USE_RHC
        radiance *= hashCacheState.sampleWeights[i];
        hashCacheAddVoxelData(hashCacheState.voxelIndices[i], radiance, false);
#elif USE_IRHC
        hashCacheAddVoxelData(hashCacheState.voxelIndices[i], radiance, false);
        radiance *= hashCacheState.sampleWeights[i];
#endif
    }
#endif // HC_UPDATE
}

void hashCacheUpdateMiss(inout HashCacheState hashCacheState, float3 radiance)
{
    hashCacheUpdateCache(hashCacheState, radiance);
}

void hashCacheUpdateHit(inout HashCacheState hashCacheState, HashCacheHitData hashCacheHitData, float3 radiance)
{
#if HC_UPDATE
    hashCacheUpdateCache(hashCacheState, radiance);
    for (uint i = hashCacheState.pathLength; i > 0; --i)
    {
        hashCacheState.voxelIndices[i] = hashCacheState.voxelIndices[i - 1];
        hashCacheState.sampleWeights[i] = hashCacheState.sampleWeights[i - 1];
    }
    for (int i = 0; i < kHashCacheLevelTrainingSpread; i++)
    {
        hashCacheState.voxelIndices[0][i] = hashCacheState.hashMapData.InsertEntry(hashCacheHitData.distance, hashCacheHitData.positionWorld, hashCacheHitData.direction, hashCacheHitData.normalWorld, i - (kHashCacheLevelTrainingSpread / 2));
    }
    hashCacheState.pathLength += 1;
    hashCacheState.pathLength = min(hashCacheState.pathLength, kHashCachePropagationDepth - 1);
    hashCacheAddVoxelData(hashCacheState.voxelIndices[0], float3(0.0), true);
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
        radiance = hashCacheState.hashMapData.HashGridDebugColoredHash(hashCacheHitData.distance, hashCacheHitData.positionWorld, hashCacheHitData.direction);
        return true;
    }
    uint idx = hashCacheState.hashMapData.FindEntry(hashCacheHitData.distance, hashCacheHitData.positionWorld, hashCacheHitData.direction, hashCacheHitData.normalWorld);
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
        voxelData.sampleNum = min(kMaxSampleCount, voxelData.sampleNum);
        hashCacheSetVoxelData(false, idx, voxelData);
        voxelData.radiance = float3(0.0);
        hashCacheSetVoxelData(true, idx, voxelData);
    }
}
}
#endif // HC_UPDATE || HC_QUERY


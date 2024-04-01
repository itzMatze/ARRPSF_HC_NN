import Utils.Debug.PixelDebug;

#if HASH_CACHE_UPDATE || HASH_CACHE_QUERY
static const uint kHashGridPositionBitNum = 8;
static const uint kHashGridPositionBitMask = ((1u << kHashGridPositionBitNum) - 1);
static const uint kHashGridLevelBitNum = 5;
static const uint kHashGridLevelBitMask = ((1u << kHashGridLevelBitNum) - 1);
static const uint kHashGridNormalBitNum = 3;
static const uint kHashGridNormalBitMask = ((1u << kHashGridNormalBitNum) - 1);
static const uint kHashGridHashMapBucketSize = 32;
static const uint kHashGridInvalidHashKey = 0;
static const uint kHashGridInvalidIdx = 0xFFFF;
static const bool kHashGridUseNormals = true;
static const uint kHashGridLevelBias = 2; // positive bias adds extra levels with content magnification
static const float3 kHashGridPositionOffset = float3(0.0f, 0.0f, 0.0f);
static const float kHashCacheGridLogarithmBase = 2.0f;
static const float kHashCacheSceneScale = 30.0f;

typedef uint HashKey;

RWStructuredBuffer<uint> gHashEntriesBuffer;

struct GridParameters
{
    float3 cameraPosition;
};

float LogBase(float x, float base)
{
    return log(x) / log(base);
}

// http://burtleburtle.net/bob/hash/integer.html
uint HashJenkins32(uint a)
{
    a = (a + 0x7ed55d16) + (a << 12);
    a = (a ^ 0xc761c23c) ^ (a >> 19);
    a = (a + 0x165667b1) + (a << 5);
    a = (a + 0xd3a2646c) ^ (a << 9);
    a = (a + 0xfd7046c5) + (a << 3);
    a = (a ^ 0xb55a4f09) ^ (a >> 16);
    return a;
}

uint Hash32(HashKey hashKey)
{
    return HashJenkins32(uint((hashKey >> 0) & 0xffffffff))
         ^ HashJenkins32(uint((hashKey >> 32) & 0xffffffff));
}

uint GetGridLevel(float3 samplePosition, GridParameters gridParameters)
{
    const float distance = length(gridParameters.cameraPosition - samplePosition);
    return clamp(floor(LogBase(distance, kHashCacheGridLogarithmBase) + kHashGridLevelBias), 1, kHashGridLevelBitMask);
}

float GetVoxelSize(uint gridLevel, GridParameters gridParameters)
{
    return pow(kHashCacheGridLogarithmBase, gridLevel) / (kHashCacheSceneScale * pow(kHashCacheGridLogarithmBase, kHashGridLevelBias));
}

// Based on logarithmic caching by Johannes Jendersie
int4 CalculateGridPositionLog(float3 samplePosition, GridParameters gridParameters)
{
    samplePosition += kHashGridPositionOffset;
    uint  gridLevel    = GetGridLevel(samplePosition, gridParameters);
    float voxelSize    = GetVoxelSize(gridLevel, gridParameters);
    int3  gridPosition = floor(samplePosition / voxelSize);
    return int4(gridPosition.xyz, gridLevel);
}

HashKey ComputeSpatialHash(float3 samplePosition, float3 sampleNormal, GridParameters gridParameters)
{
    const float distance = length(gridParameters.cameraPosition - samplePosition);
    uint4 gridPosition = asuint(CalculateGridPositionLog(samplePosition, gridParameters));
    HashKey hashKey = (((uint)gridPosition.x & kHashGridPositionBitMask) << (kHashGridPositionBitNum * 0))
                    | (((uint)gridPosition.y & kHashGridPositionBitMask) << (kHashGridPositionBitNum * 1))
                    | (((uint)gridPosition.z & kHashGridPositionBitMask) << (kHashGridPositionBitNum * 2))
                    | (((uint)gridPosition.w & kHashGridLevelBitMask) << (kHashGridPositionBitNum * 3));

    if (kHashGridUseNormals)
    {
        uint normalBits =
            (sampleNormal.x >= 0 ? 1 : 0) +
            (sampleNormal.y >= 0 ? 2 : 0) +
            (sampleNormal.z >= 0 ? 4 : 0);

        hashKey |= ((uint)normalBits << (kHashGridPositionBitNum * 3 + kHashGridLevelBitNum));
    }
    return hashKey;
}

struct HashMapData
{
    uint capacity;

    void AtomicCompareExchange(in uint dstOffset, in uint compareValue, in uint value, out uint originalValue)
    {
        InterlockedCompareExchange(gHashEntriesBuffer[dstOffset], compareValue, value, originalValue);
    }

    bool HashMapInsert(const HashKey hashKey, out uint cacheEntry)
    {
        uint    hash        = Hash32(hashKey);
        uint    slot        = hash % capacity;
        uint    initSlot    = slot;
        HashKey prevHashKey = kHashGridInvalidHashKey;
        for (uint bucketOffset = 0; bucketOffset < kHashGridHashMapBucketSize && slot < capacity; ++bucketOffset)
        {
            AtomicCompareExchange(slot + bucketOffset, kHashGridInvalidHashKey, hashKey, prevHashKey);

            if (prevHashKey == kHashGridInvalidHashKey || prevHashKey == hashKey)
            {
                cacheEntry = slot + bucketOffset;
                return true;
            }
        }
        cacheEntry = 0;
        return false;
    }

    bool HashMapFind(const HashKey hashKey, inout uint cacheEntry)
    {
        uint hash = Hash32(hashKey);
        uint slot = hash % capacity;
        for (uint bucketOffset = 0; bucketOffset < kHashGridHashMapBucketSize; ++bucketOffset)
        {
            HashKey storedHashKey = gHashEntriesBuffer[slot + bucketOffset];
            if (storedHashKey == hashKey)
            {
                cacheEntry = slot + bucketOffset;
                return true;
            }
        }
        return false;
    }

    uint InsertEntry(float3 samplePosition, float3 sampleNormal, GridParameters gridParameters)
    {
        uint idx = kHashGridInvalidIdx;
        const HashKey hashKey = ComputeSpatialHash(samplePosition, sampleNormal, gridParameters);
        const bool successful = HashMapInsert(hashKey, idx);
        return idx;
    }

    uint FindEntry(float3 samplePosition, float3 sampleNormal, GridParameters gridParameters)
    {
        uint idx = kHashGridInvalidIdx;
        const HashKey hashKey = ComputeSpatialHash(samplePosition, sampleNormal, gridParameters);
        const bool successful = HashMapFind(hashKey, idx);
        return idx;
    }
};

// Debug functions
float3 GetColorFromHash32(uint hash)
{
    float3 color;
    color.x = ((hash >>  0) & 0x3ff) / 1023.0f;
    color.y = ((hash >> 11) & 0x7ff) / 2047.0f;
    color.z = ((hash >> 22) & 0x7ff) / 2047.0f;
    return color;
}

// Debug visualization
float3 HashGridDebugColoredHash(float3 samplePosition, GridParameters gridParameters)
{
    HashKey hashKey = ComputeSpatialHash(samplePosition, float3(0, 0, 0), gridParameters);
    return GetColorFromHash32(Hash32(hashKey));
}
#endif // HASH_CACHE_UPDATE || HASH_CACHE_QUERY


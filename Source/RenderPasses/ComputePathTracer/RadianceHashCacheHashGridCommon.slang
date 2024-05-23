import Utils.Debug.PixelDebug;

#if R_HC_UPDATE || R_HC_QUERY
namespace rhc
{
typedef uint64_t HashKey;
static const uint sizeofHashKey = 8;

static const HashKey kHashGridPositionBitNum = 17;
static const HashKey kHashGridPositionBitMask = ((1u << kHashGridPositionBitNum) - 1);
static const HashKey kHashGridLevelBitNum = 10;
static const HashKey kHashGridLevelBitMask = ((1u << kHashGridLevelBitNum) - 1);
static const HashKey kHashGridNormalBitNum = 3;
static const HashKey kHashGridNormalBitMask = ((1u << kHashGridNormalBitNum) - 1);
static const HashKey kHashGridHashMapBucketSize = 32;
static const HashKey kHashGridInvalidHashKey = 0;
static const HashKey kHashGridInvalidIdx = 0xffffffff;
static const bool kHashGridUseNormals = true;
static const uint kHashGridLevelBias = 2; // positive bias adds extra levels with content magnification
static const float3 kHashGridPositionOffset = float3(0.0f, 0.0f, 0.0f);
static const float kHashCacheGridLogarithmBase = 2.0f;
static const float kHashCacheSceneScale = 60.0f;

RWByteAddressBuffer gRHCHashGridEntriesBuffer;

float LogBase(float x, float base)
{
    return log(x) / log(base);
}

struct HashMapData
{
    uint capacity;

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

    uint GetGridLevel(float distance)
    {
        return clamp(floor(LogBase(distance, kHashCacheGridLogarithmBase) + kHashGridLevelBias), 1, kHashGridLevelBitMask);
    }

    float GetVoxelSize(uint gridLevel)
    {
        return pow(kHashCacheGridLogarithmBase, gridLevel) / (kHashCacheSceneScale * pow(kHashCacheGridLogarithmBase, kHashGridLevelBias));
    }

    // Based on logarithmic caching by Johannes Jendersie
    int4 CalculateGridPositionLog(float distance, float3 samplePosition, int levelOffset)
    {
        samplePosition += kHashGridPositionOffset;
        uint  gridLevel    = GetGridLevel(distance) + levelOffset;
        float voxelSize    = GetVoxelSize(gridLevel);
        int3  gridPosition = floor(samplePosition / voxelSize);
        return int4(gridPosition.xyz, gridLevel);
    }

    HashKey ComputeSpatialHash(float distance, float3 samplePosition, float3 sampleNormal, int levelOffset = 0)
    {
        uint4 gridPosition = asuint(CalculateGridPositionLog(distance, samplePosition, levelOffset));
        HashKey hashKey = (((HashKey)gridPosition.x & kHashGridPositionBitMask) << (kHashGridPositionBitNum * 0))
                        | (((HashKey)gridPosition.y & kHashGridPositionBitMask) << (kHashGridPositionBitNum * 1))
                        | (((HashKey)gridPosition.z & kHashGridPositionBitMask) << (kHashGridPositionBitNum * 2))
                        | (((HashKey)gridPosition.w & kHashGridLevelBitMask) << (kHashGridPositionBitNum * 3));

        if (kHashGridUseNormals)
        {
            HashKey normalBits =
                (sampleNormal.x >= 0 ? 1 : 0) +
                (sampleNormal.y >= 0 ? 2 : 0) +
                (sampleNormal.z >= 0 ? 4 : 0);

            hashKey |= ((HashKey)normalBits << (kHashGridPositionBitNum * 3 + kHashGridLevelBitNum));
        }
        return hashKey;
    }

    void AtomicCompareExchange(in uint dstOffset, in HashKey compareValue, in HashKey value, out HashKey originalValue)
    {
        gRHCHashGridEntriesBuffer.InterlockedCompareExchangeU64(dstOffset * sizeofHashKey, compareValue, value, originalValue);
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
            HashKey storedHashKey = gRHCHashGridEntriesBuffer.Load<HashKey>((slot + bucketOffset) * sizeofHashKey);
            if (storedHashKey == hashKey)
            {
                cacheEntry = slot + bucketOffset;
                return true;
            }
        }
        return false;
    }

    uint InsertEntry(float distance, float3 samplePosition, float3 sampleNormal, int levelOffset = 0)
    {
        uint idx = kHashGridInvalidIdx;
        const HashKey hashKey = ComputeSpatialHash(distance, samplePosition, sampleNormal, levelOffset);
        const bool successful = HashMapInsert(hashKey, idx);
        return idx;
    }

    uint FindEntry(float distance, float3 samplePosition, float3 sampleNormal)
    {
        uint idx = kHashGridInvalidIdx;
        const HashKey hashKey = ComputeSpatialHash(distance, samplePosition, sampleNormal);
        const bool successful = HashMapFind(hashKey, idx);
        return idx;
    }

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
    float3 HashGridDebugColoredHash(float distance, float3 samplePosition)
    {
        HashKey hashKey = ComputeSpatialHash(distance, samplePosition, float3(0, 0, 0));
        return GetColorFromHash32(Hash32(hashKey));
    }
}
}
#endif // R_HC_UPDATE || R_HC_QUERY


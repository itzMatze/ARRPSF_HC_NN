import Utils.Debug.PixelDebug;

#if HC_UPDATE || HC_QUERY
namespace hc
{
typedef uint64_t HashKey;
static const uint sizeofHashKey = 8;

#if USE_IRHC
static const HashKey kHashGridPositionBitNum = 15;
static const HashKey kHashGridPositionBitMask = ((1u << kHashGridPositionBitNum) - 1);
static const HashKey kHashGridDirectionBitNum = 3;
static const HashKey kHashGridDirectionBitMask = ((1u << kHashGridDirectionBitNum) - 1);
static const HashKey kHashGridLevelBitNum = 7;
static const HashKey kHashGridLevelBitMask = ((1u << kHashGridLevelBitNum) - 1);
static const HashKey kHashGridNormalBitNum = 3;
static const HashKey kHashGridNormalBitMask = ((1u << kHashGridNormalBitNum) - 1);
#elif USE_RHC
static const HashKey kHashGridPositionBitNum = 17;
static const HashKey kHashGridPositionBitMask = ((1u << kHashGridPositionBitNum) - 1);
static const HashKey kHashGridDirectionBitNum = 0;
static const HashKey kHashGridDirectionBitMask = ((1u << kHashGridDirectionBitNum) - 1);
static const HashKey kHashGridLevelBitNum = 10;
static const HashKey kHashGridLevelBitMask = ((1u << kHashGridLevelBitNum) - 1);
static const HashKey kHashGridNormalBitNum = 3;
static const HashKey kHashGridNormalBitMask = ((1u << kHashGridNormalBitNum) - 1);
#endif
static const HashKey kHashGridHashMapBucketSize = 32;
static const HashKey kHashGridInvalidHashKey = 0;
static const HashKey kHashGridInvalidIdx = 0xffffffff;
static const uint kHashGridLevelBias = 2; // positive bias adds extra levels with content magnification
static const float3 kHashGridPositionOffset = float3(0.0f, 0.0f, 0.0f);
static const float kHashCacheGridLogarithmBase = 2.0f;
static const float kHashCacheSceneScale = 60.0f;
static const uint kHashCacheCapacity = HC_HASHMAP_SIZE;

RWByteAddressBuffer gHCHashGridEntriesBuffer;

float LogBase(float x, float base)
{
    return log(x) / log(base);
}

void HashMapReset(const uint idx)
{
    gHCHashGridEntriesBuffer.Store(idx * sizeofHashKey, HashKey(0));
}

struct HashMapData
{
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
        uint gridLevel = GetGridLevel(distance) + levelOffset;
        float voxelSize = GetVoxelSize(gridLevel);
        int3 gridPosition = floor(samplePosition / voxelSize);
        return int4(gridPosition.xyz, gridLevel);
    }

    float2 convertDirToPolar(float3 input)
    {
        float2 angles = float2(atan(input.y / input.x), acos(input.z));
        return angles;
    }

    uint2 quantizeToBitRange(float2 input, uint max)
    {
        return uint2(((input + 1.0) / 2.0) * float((max + 1) - 0.001));
    }

    uint3 quantizeToBitRange(float3 input, uint max)
    {
        return uint3(((input + 1.0) / 2.0) * float((max + 1) - 0.001));
    }

    HashKey ComputeSpatialHash(float distance, float3 samplePosition, float3 sampleDirection, float3 sampleNormal, int levelOffset = 0)
    {
        uint4 gridPosition = asuint(CalculateGridPositionLog(distance, samplePosition, levelOffset));
        uint3 quantizedDir = quantizeToBitRange(sampleDirection, kHashGridDirectionBitMask);
        // | normal bits | level | pos.z | pos.y | pos.x | dir.z | dir.y | dir.x |
        HashKey hashKey = (sampleNormal.x >= 0 ? 1 : 0) + (sampleNormal.y >= 0 ? 2 : 0) + (sampleNormal.z >= 0 ? 4 : 0);
        hashKey <<= kHashGridLevelBitNum;
        hashKey |= ((HashKey)gridPosition.w & kHashGridLevelBitMask);
        hashKey <<= kHashGridPositionBitNum;
        hashKey |= ((HashKey)gridPosition.z & kHashGridPositionBitMask);
        hashKey <<= kHashGridPositionBitNum;
        hashKey |= ((HashKey)gridPosition.y & kHashGridPositionBitMask);
        hashKey <<= kHashGridPositionBitNum;
        hashKey |= ((HashKey)gridPosition.x & kHashGridPositionBitMask);
        hashKey <<= kHashGridDirectionBitNum;
        hashKey |= ((HashKey)quantizedDir.z);
        hashKey <<= kHashGridDirectionBitNum;
        hashKey |= ((HashKey)quantizedDir.y);
        hashKey <<= kHashGridDirectionBitNum;
        hashKey |= ((HashKey)quantizedDir.x);
        return hashKey;
    }

    uint InsertEntry(float distance, float3 samplePosition, float3 sampleDirection, float3 sampleNormal, int levelOffset = 0)
    {
        const HashKey hashKey = ComputeSpatialHash(distance, samplePosition, sampleDirection, sampleNormal, levelOffset);
        const uint hash = Hash32(hashKey);
        const uint slot = hash % kHashCacheCapacity;
        HashKey prevHashKey = kHashGridInvalidHashKey;
        for (uint bucketOffset = 0; bucketOffset < kHashGridHashMapBucketSize && slot < kHashCacheCapacity; ++bucketOffset)
        {
            gHCHashGridEntriesBuffer.InterlockedCompareExchangeU64((slot + bucketOffset) * sizeofHashKey, kHashGridInvalidHashKey, hashKey, prevHashKey);
            if (prevHashKey == kHashGridInvalidHashKey || prevHashKey == hashKey)
            {
                return slot + bucketOffset;
            }
        }
        return kHashGridInvalidIdx;
    }

    uint FindEntry(float distance, float3 samplePosition, float3 sampleDirection, float3 sampleNormal)
    {
        const HashKey hashKey = ComputeSpatialHash(distance, samplePosition, sampleDirection, sampleNormal, 0);
        const uint hash = Hash32(hashKey);
        const uint slot = hash % kHashCacheCapacity;
        for (uint bucketOffset = 0; bucketOffset < kHashGridHashMapBucketSize; ++bucketOffset)
        {
            HashKey storedHashKey = gHCHashGridEntriesBuffer.Load<HashKey>((slot + bucketOffset) * sizeofHashKey);
            if (storedHashKey == hashKey)
            {
                return slot + bucketOffset;
            }
        }
        return kHashGridInvalidIdx;
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
    float3 HashGridDebugColoredHash(float distance, float3 samplePosition, float3 sampleDirection)
    {
        HashKey hashKey = ComputeSpatialHash(distance, samplePosition, sampleDirection, float3(0, 0, 0));
        return GetColorFromHash32(Hash32(hashKey));
    }
}
}
#endif // HC_UPDATE || HC_QUERY


import Utils.Debug.PixelDebug;

typedef uint64_t HashKey;
static const uint sizeofHashKey = 8;

#if USE_NIRC
static const HashKey kHashGridPositionBitNum = 14;
static const HashKey kHashGridPositionBitMask = ((1u << kHashGridPositionBitNum) - 1);
static const HashKey kHashGridDirectionBitNum = 4;
static const HashKey kHashGridDirectionBitMask = ((1u << kHashGridDirectionBitNum) - 1);
static const HashKey kHashGridLevelBitNum = 7;
static const HashKey kHashGridLevelBitMask = ((1u << kHashGridLevelBitNum) - 1);
static const HashKey kHashGridNormalBitNum = 3;
static const HashKey kHashGridNormalBitMask = ((1u << kHashGridNormalBitNum) - 1);
#elif USE_NRC
static const HashKey kHashGridPositionBitNum = 17;
static const HashKey kHashGridPositionBitMask = ((1u << kHashGridPositionBitNum) - 1);
static const HashKey kHashGridDirectionBitNum = 0;
static const HashKey kHashGridDirectionBitMask = ((1u << kHashGridDirectionBitNum) - 1);
static const HashKey kHashGridLevelBitNum = 10;
static const HashKey kHashGridLevelBitMask = ((1u << kHashGridLevelBitNum) - 1);
static const HashKey kHashGridNormalBitNum = 3;
static const HashKey kHashGridNormalBitMask = ((1u << kHashGridNormalBitNum) - 1);
#endif
static const float kHashCacheGridLogarithmBase = 2.0f;
static const float kHashCacheSceneScale = 60.0f;
static const uint kFeatureHashGridProbingSize = FEATURE_HASH_GRID_PROBING_SIZE;
static const HashKey kHashGridInvalidHashKey = 0;

#if FEATURE_HASH_GRID_PROBING_SIZE > 0
RWByteAddressBuffer gFeatureHashGridEntriesBuffer;
#endif

void FeatureHashMapReset(const uint idx)
{
#if FEATURE_HASH_GRID_PROBING_SIZE > 0
    gFeatureHashGridEntriesBuffer.Store(idx * sizeofHashKey, HashKey(0));
#endif
}

struct FeatureHashGridData
{
    uint capacity;

    __init(uint capacity)
    {
        this.capacity = capacity;
    }

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

    float GetVoxelSize(uint gridLevel)
    {
        return pow(kHashCacheGridLogarithmBase, gridLevel) / (kHashCacheSceneScale);
    }

    // Based on logarithmic caching by Johannes Jendersie
    int3 CalculateGridPositionLog(float3 samplePosition, uint level)
    {
        float voxelSize = GetVoxelSize(level);
        int3 gridPosition = floor(samplePosition / voxelSize);
        return gridPosition.xyz;
    }

    uint3 quantizeToBitRange(float3 input, uint max)
    {
        return uint3(((input + 1.0) / 2.0) * float((max + 1) - 0.001));
    }

    HashKey ComputeSpatialHash(float3 samplePosition, float3 sampleDirection, float3 sampleNormal, uint level)
    {
        uint3 gridPosition = asuint(CalculateGridPositionLog(samplePosition, level));
        uint3 quantizedDir = quantizeToBitRange(sampleDirection, kHashGridDirectionBitMask);
        // | normal bits | level | pos.z | pos.y | pos.x | dir.z | dir.y | dir.x |
        HashKey hashKey = (sampleNormal.x >= 0 ? 1 : 0) + (sampleNormal.y >= 0 ? 2 : 0) + (sampleNormal.z >= 0 ? 4 : 0);
        hashKey <<= kHashGridLevelBitNum;
        hashKey |= ((HashKey)level & kHashGridLevelBitMask);
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

    uint InsertEntry(float3 samplePosition, float3 sampleDirection, float3 sampleNormal, uint level)
    {
        const HashKey hashKey = ComputeSpatialHash(samplePosition, sampleDirection, sampleNormal, level);
        uint hash = Hash32(hashKey);
        uint slot = (hash % capacity);
#if FEATURE_HASH_GRID_PROBING_SIZE > 0
        HashKey prevHashKey = kHashGridInvalidHashKey;
        // search for slot that is empty or occupied with the same hash
        // if no such slot is found, produce collision on the first slot
        for (uint bucketOffset = 0; (bucketOffset <= kFeatureHashGridProbingSize) && (slot + bucketOffset < capacity); ++bucketOffset)
        {
            gFeatureHashGridEntriesBuffer.InterlockedCompareExchangeU64((slot + bucketOffset) * sizeofHashKey, kHashGridInvalidHashKey, hashKey, prevHashKey);
            if (prevHashKey == kHashGridInvalidHashKey || prevHashKey == hashKey)
            {
                slot += bucketOffset;
                break;
            }
        }
#endif
        return slot;
    }

    uint FindEntry(float3 samplePosition, float3 sampleDirection, float3 sampleNormal, uint level)
    {
        const HashKey hashKey = ComputeSpatialHash(samplePosition, sampleDirection, sampleNormal, level);
        uint hash = Hash32(hashKey);
        uint slot = (hash % capacity);
#if FEATURE_HASH_GRID_PROBING_SIZE > 0
        HashKey prevHashKey = kHashGridInvalidHashKey;
        // search for slot that is empty or occupied with the same hash
        // if no such slot is found, produce collision on the first slot
        for (uint bucketOffset = 0; (bucketOffset <= kFeatureHashGridProbingSize) && (slot + bucketOffset < capacity); ++bucketOffset)
        {
            HashKey storedHashKey = gFeatureHashGridEntriesBuffer.Load<HashKey>((slot + bucketOffset) * sizeofHashKey);
            if (storedHashKey == hashKey)
            {
                slot += bucketOffset;
                break;
            }
        }
#endif
        return slot;
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
    float3 DebugColoredHash(float3 pos, float3 dir, float3 normal, uint level)
    {
        HashKey hashKey = ComputeSpatialHash(pos, dir, normal, level);
        return GetColorFromHash32(Hash32(hashKey));
    }
}


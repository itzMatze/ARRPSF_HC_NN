import Utils.Debug.PixelDebug;
#include "Utils/Math/MathConstants.slangh"
#include "TinynnTensorview.slang"

static const uint kFeatureHashGridSize = FEATURE_HASH_GRID_SIZE;
static const uint kFeatureHashGridPlacesPerElement = FEATURE_HASH_GRID_PLACES_PER_ELEMENT;
static const uint kFeatureHashGridCapacity = kFeatureHashGridSize / kFeatureHashGridPlacesPerElement;

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

struct FeatureHashGridData
{
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

    float2 convertDirToPolar(float3 input)
    {
        float2 angles = float2(atan(input.y / input.x), acos(input.z));
        return angles;
    }

    uint2 quantizePolarToBitRange(float2 input, uint newMax)
    {
        uint2 result;
        result.x = uint(((input.x + M_PI) / (M_PI + M_PI)) * float((newMax + 1) - 0.001));
        result.y = uint((input.y / M_PI) * float((newMax + 1) - 0.001));
        return result;
    }

    uint3 quantizeCartesianToBitRange(float3 input, uint newMax)
    {
        return uint3(((input + 1.0) / 2.0) * float((newMax + 1) - 0.001));
    }

    HashKey ComputeSpatialHash(float3 samplePosition, float3 sampleDirection, float3 sampleNormal, uint level, bool multiLevelDir = false)
    {
#if USE_NRC
        uint3 gridPosition = asuint(CalculateGridPositionLog(samplePosition, level));
        uint3 quantizedDir = quantizeCartesianToBitRange(sampleDirection, kHashGridDirectionBitMask);
        const HashKey positionBitNum = kHashGridPositionBitNum;
        const HashKey positionBitMask = kHashGridPositionBitMask;
        const HashKey directionBitNum = kHashGridDirectionBitNum;
        const HashKey directionBitMask = kHashGridDirectionBitMask;
#elif USE_NIRC
        // range of used bits for directional encoding: [1, 8]
        const HashKey directionBitNum = multiLevelDir ? level + 1 : kHashGridDirectionBitNum;
        const HashKey directionBitMask = ((1u << directionBitNum) - 1);
        const HashKey positionBitNum = 18 - directionBitNum;
        const HashKey positionBitMask = ((1u << positionBitNum) - 1);
        uint3 gridPosition = multiLevelDir ? asuint(CalculateGridPositionLog(samplePosition, 5)) : asuint(CalculateGridPositionLog(samplePosition, level));
        uint2 quantizedDir = quantizePolarToBitRange(convertDirToPolar(sampleDirection), directionBitMask);
#endif
        // | normal bits | level | pos.z | pos.y | pos.x | dir.z | dir.y | dir.x |
        HashKey hashKey = (sampleNormal.x >= 0 ? 1 : 0) + (sampleNormal.y >= 0 ? 2 : 0) + (sampleNormal.z >= 0 ? 4 : 0);
        hashKey <<= kHashGridLevelBitNum;
        hashKey |= ((HashKey)level & kHashGridLevelBitMask);
        hashKey <<= positionBitNum;
        hashKey |= ((HashKey)gridPosition.z & positionBitMask);
        hashKey <<= positionBitNum;
        hashKey |= ((HashKey)gridPosition.y & positionBitMask);
        hashKey <<= positionBitNum;
        hashKey |= ((HashKey)gridPosition.x & positionBitMask);
        //hashKey <<= directionBitNum;
        //hashKey |= ((HashKey)quantizedDir.z & directionBitMask);
        hashKey <<= directionBitNum;
        hashKey |= ((HashKey)quantizedDir.y & directionBitMask);
        hashKey <<= directionBitNum;
        hashKey |= ((HashKey)quantizedDir.x & directionBitMask);
        return hashKey;
    }

    uint InsertEntry(float3 samplePosition, float3 sampleDirection, float3 sampleNormal, uint level, bool multiLevelDir)
    {
        const HashKey hashKey = ComputeSpatialHash(samplePosition, sampleDirection, sampleNormal, level, multiLevelDir);
        uint hash = Hash32(hashKey);
        uint slot = (hash % kFeatureHashGridCapacity);
#if FEATURE_HASH_GRID_PROBING_SIZE > 0
        HashKey prevHashKey = kHashGridInvalidHashKey;
        // search for slot that is empty or occupied with the same hash
        // if no such slot is found, produce collision on the first slot
        for (uint bucketOffset = 0; (bucketOffset <= kFeatureHashGridProbingSize) && (slot + bucketOffset < kFeatureHashGridCapacity); ++bucketOffset)
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

    uint FindEntry(float3 samplePosition, float3 sampleDirection, float3 sampleNormal, uint level, bool multiLevelDir)
    {
        const HashKey hashKey = ComputeSpatialHash(samplePosition, sampleDirection, sampleNormal, level, multiLevelDir);
        uint hash = Hash32(hashKey);
        uint slot = (hash % kFeatureHashGridCapacity);
#if FEATURE_HASH_GRID_PROBING_SIZE > 0
        HashKey prevHashKey = kHashGridInvalidHashKey;
        // search for slot that is empty or occupied with the same hash
        // if no such slot is found, produce collision on the first slot
        for (uint bucketOffset = 0; (bucketOffset <= kFeatureHashGridProbingSize) && (slot + bucketOffset < kFeatureHashGridCapacity); ++bucketOffset)
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

struct FeatureHashGrid
{
    FeatureHashGridData hashmap;
    TensorView dataView;

    __init(inout uint param_offset, inout uint grad_offset)
    {
        hashmap = FeatureHashGridData();
        dataView = TensorView(param_offset, grad_offset, 32, 32);
        param_offset += kFeatureHashGridSize;
        grad_offset += kFeatureHashGridSize;
    }

    uint InsertEntry(float3 samplePosition, float3 sampleDirection, float3 sampleNormal, uint level, bool multiLevelDir)
    {
        return hashmap.InsertEntry(samplePosition, sampleDirection, sampleNormal, level, multiLevelDir) * kFeatureHashGridPlacesPerElement;
    }

    uint FindEntry(float3 samplePosition, float3 sampleDirection, float3 sampleNormal, uint level, bool multiLevelDir)
    {
        return hashmap.FindEntry(samplePosition, sampleDirection, sampleNormal, level, multiLevelDir) * kFeatureHashGridPlacesPerElement;
    }
}

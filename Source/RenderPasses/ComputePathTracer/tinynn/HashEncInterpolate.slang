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
    void CalculateGridPositionLog(float3 samplePosition, uint level, out int3 gridPosition, out float3 weights)
    {
        float voxelSize = GetVoxelSize(level);
        float3 scaledSamplePosition = samplePosition / voxelSize;
        gridPosition = int3(scaledSamplePosition);
        weights = scaledSamplePosition - gridPosition;
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

    void ComputeSpatialHash(float3 samplePosition, float3 sampleDirection, float3 sampleNormal, uint level, bool multiLevelDir, out HashKey[8] hashKeys, out float[8] weights)
    {
        int3 gridPosition;
        float3 weightsAxis;
#if USE_NRC
        CalculateGridPositionLog(samplePosition, level, gridPosition, weightsAxis);
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
        if (multiLevelDir) level = 5;
        CalculateGridPositionLog(samplePosition, level, gridPosition, weightsAxis);
        uint2 quantizedDir = quantizePolarToBitRange(convertDirToPolar(sampleDirection), directionBitMask);
#endif
        uint3[8] gridPositionCorners;
        gridPositionCorners[0] = asuint(gridPosition);
        weights[0] = (1.0 - weightsAxis.x) * (1.0 - weightsAxis.y) * (1.0 - weightsAxis.z);
        gridPositionCorners[1] = asuint(gridPosition + int3(1, 1, 1));
        weights[1] = weightsAxis.x * weightsAxis.y * weightsAxis.z;
        gridPositionCorners[2] = asuint(gridPosition + int3(1, 0, 0));
        weights[2] = weightsAxis.x * (1.0 - weightsAxis.y) * (1.0 - weightsAxis.z);
        gridPositionCorners[3] = asuint(gridPosition + int3(0, 1, 0));
        weights[3] = (1.0 - weightsAxis.x) * weightsAxis.y * (1.0 - weightsAxis.z);
        gridPositionCorners[4] = asuint(gridPosition + int3(0, 0, 1));
        weights[4] = (1.0 - weightsAxis.x) * (1.0 - weightsAxis.y) * weightsAxis.z;
        gridPositionCorners[5] = asuint(gridPosition + int3(1, 1, 0));
        weights[5] = weightsAxis.x * weightsAxis.y * (1.0 - weightsAxis.z);
        gridPositionCorners[6] = asuint(gridPosition + int3(1, 0, 1));
        weights[6] = weightsAxis.x * (1.0 - weightsAxis.y) * weightsAxis.z;
        gridPositionCorners[7] = asuint(gridPosition + int3(0, 1, 1));
        weights[7] = (1.0 - weightsAxis.x) * weightsAxis.y * weightsAxis.z;
        for (uint i = 0; i < 8; i++)
        {
            // | normal bits | level | pos.z | pos.y | pos.x | dir.z | dir.y | dir.x |
            hashKeys[i] = (sampleNormal.x >= 0 ? 1 : 0) + (sampleNormal.y >= 0 ? 2 : 0) + (sampleNormal.z >= 0 ? 4 : 0);
            hashKeys[i] <<= kHashGridLevelBitNum;
            hashKeys[i] |= ((HashKey)level & kHashGridLevelBitMask);
            hashKeys[i] <<= positionBitNum;
            hashKeys[i] |= ((HashKey)gridPositionCorners[i].z & positionBitMask);
            hashKeys[i] <<= positionBitNum;
            hashKeys[i] |= ((HashKey)gridPositionCorners[i].y & positionBitMask);
            hashKeys[i] <<= positionBitNum;
            hashKeys[i] |= ((HashKey)gridPositionCorners[i].x & positionBitMask);
            hashKeys[i] <<= directionBitNum;
            hashKeys[i] |= ((HashKey)quantizedDir.y & directionBitMask);
            hashKeys[i] <<= directionBitNum;
            hashKeys[i] |= ((HashKey)quantizedDir.x & directionBitMask);
        }
    }

    void InsertEntry(float3 samplePosition, float3 sampleDirection, float3 sampleNormal, uint level, bool multiLevelDir, out uint[8] indices, out float[8] weights)
    {
        HashKey[8] hashKeys;
        ComputeSpatialHash(samplePosition, sampleDirection, sampleNormal, level, multiLevelDir, hashKeys, weights);
        for (uint i = 0; i < 8; i++)
        {
            uint hash = Hash32(hashKeys[i]);
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
            indices[i] = slot;
        }
    }

    void FindEntry(float3 samplePosition, float3 sampleDirection, float3 sampleNormal, uint level, bool multiLevelDir, out uint[8] indices, out float[8] weights)
    {
        HashKey[8] hashKeys;
        ComputeSpatialHash(samplePosition, sampleDirection, sampleNormal, level, multiLevelDir, hashKeys, weights);
        for (uint i = 0; i < 8; i++)
        {
            uint hash = Hash32(hashKeys[i]);
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
            indices[i] = slot;
        }
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

    void InsertEntry(float3 samplePosition, float3 sampleDirection, float3 sampleNormal, uint level, bool multiLevelDir, out uint[8] indices, out float[8] weights)
    {
        hashmap.InsertEntry(samplePosition, sampleDirection, sampleNormal, level, multiLevelDir, indices, weights);
        for (uint i = 0; i < 8; i++) indices[i] *= kFeatureHashGridPlacesPerElement;
    }

    void FindEntry(float3 samplePosition, float3 sampleDirection, float3 sampleNormal, uint level, bool multiLevelDir, out uint[8] indices, out float[8] weights)
    {
        hashmap.FindEntry(samplePosition, sampleDirection, sampleNormal, level, multiLevelDir, indices, weights);
        for (uint i = 0; i < 8; i++) indices[i] *= kFeatureHashGridPlacesPerElement;
    }
}

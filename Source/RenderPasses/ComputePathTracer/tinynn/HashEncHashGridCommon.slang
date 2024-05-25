#include "TinynnTensorview.hlsli"
import Utils.Debug.PixelDebug;

typedef uint64_t HashKey;

static const uint kHashGridPositionBitNum = 17;
static const uint kHashGridPositionBitMask = ((1u << kHashGridPositionBitNum) - 1);
static const uint kHashGridLevelBitNum = 10;
static const uint kHashGridLevelBitMask = ((1u << kHashGridLevelBitNum) - 1);
static const uint kHashGridNormalBitNum = 3;
static const uint kHashGridNormalBitMask = ((1u << kHashGridNormalBitNum) - 1);
static const float kHashCacheGridLogarithmBase = 2.0f;
static const float kHashCacheSceneScale = 60.0f;
static const uint kFeatureHashGridSize = FEATURE_HASH_GRID_SIZE;
static const uint kFeatureHashGridPlacesPerElement = FEATURE_HASH_GRID_PLACES_PER_ELEMENT;

struct FeatureHashGridData
{
    uint capacity;
    TensorView dataView;

    __init(inout uint param_offset, inout uint grad_offset)
    {
        this.capacity = kFeatureHashGridSize / kFeatureHashGridPlacesPerElement;
        dataView = TensorView(param_offset, grad_offset, 32, 32);
        param_offset += kFeatureHashGridSize * kFeatureHashGridPlacesPerElement;
        grad_offset += kFeatureHashGridSize * kFeatureHashGridPlacesPerElement;
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
    int4 CalculateGridPositionLog(float3 samplePosition, uint level)
    {
        float voxelSize    = GetVoxelSize(level);
        int3  gridPosition = floor(samplePosition / voxelSize);
        return int4(gridPosition.xyz, level);
    }

    HashKey ComputeSpatialHash(float3 samplePosition, float3 sampleNormal, uint level)
    {
        uint4 gridPosition = asuint(CalculateGridPositionLog(samplePosition, level));
        HashKey hashKey = (((HashKey)gridPosition.x & kHashGridPositionBitMask) << (kHashGridPositionBitNum * 0))
                        | (((HashKey)gridPosition.y & kHashGridPositionBitMask) << (kHashGridPositionBitNum * 1))
                        | (((HashKey)gridPosition.z & kHashGridPositionBitMask) << (kHashGridPositionBitNum * 2))
                        | (((HashKey)gridPosition.w & kHashGridLevelBitMask) << (kHashGridPositionBitNum * 3));

        HashKey normalBits =
            (sampleNormal.x >= 0 ? 1 : 0) +
            (sampleNormal.y >= 0 ? 2 : 0) +
            (sampleNormal.z >= 0 ? 4 : 0);

        hashKey |= ((HashKey)normalBits << (kHashGridPositionBitNum * 3 + kHashGridLevelBitNum));
        return hashKey;
    }

    uint FindEntry(float3 samplePosition, float3 sampleNormal, uint level)
    {
        const HashKey hashKey = ComputeSpatialHash(samplePosition, sampleNormal, level);
        uint hash = Hash32(hashKey);
        uint slot = (hash % capacity) * kFeatureHashGridPlacesPerElement;
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
    float3 DebugColoredHash(float3 pos, float3 normal, uint level)
    {
        HashKey hashKey = ComputeSpatialHash(pos, normal, level);
        return GetColorFromHash32(Hash32(hashKey));
    }
}


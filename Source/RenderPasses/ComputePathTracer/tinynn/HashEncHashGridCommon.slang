import Utils.Debug.PixelDebug;

static const uint kHashGridPositionBitNum = 8;
static const uint kHashGridPositionBitMask = ((1u << kHashGridPositionBitNum) - 1);
static const uint kHashGridLevelBitNum = 5;
static const uint kHashGridLevelBitMask = ((1u << kHashGridLevelBitNum) - 1);
static const uint kHashGridNormalBitNum = 3;
static const uint kHashGridNormalBitMask = ((1u << kHashGridNormalBitNum) - 1);
static const float kHashCacheGridLogarithmBase = 2.0f;
static const float kHashCacheSceneScale = 30.0f;

typedef uint HashKey;

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
    int4 CalculateGridPositionLog(float3 samplePosition, uint level)
    {
        float voxelSize    = GetVoxelSize(level);
        int3  gridPosition = floor(samplePosition / voxelSize);
        return int4(gridPosition.xyz, level);
    }

    HashKey ComputeSpatialHash(float3 samplePosition, float3 sampleNormal, uint level)
    {
        uint4 gridPosition = asuint(CalculateGridPositionLog(samplePosition, level));
        HashKey hashKey = (((uint)gridPosition.x & kHashGridPositionBitMask) << (kHashGridPositionBitNum * 0))
                        | (((uint)gridPosition.y & kHashGridPositionBitMask) << (kHashGridPositionBitNum * 1))
                        | (((uint)gridPosition.z & kHashGridPositionBitMask) << (kHashGridPositionBitNum * 2))
                        | (((uint)gridPosition.w & kHashGridLevelBitMask) << (kHashGridPositionBitNum * 3));

        uint normalBits =
            (sampleNormal.x >= 0 ? 1 : 0) +
            (sampleNormal.y >= 0 ? 2 : 0) +
            (sampleNormal.z >= 0 ? 4 : 0);

        hashKey |= ((uint)normalBits << (kHashGridPositionBitNum * 3 + kHashGridLevelBitNum));
        return hashKey;
    }

    uint FindEntry(float3 samplePosition, float3 sampleNormal, uint level)
    {
        const HashKey hashKey = ComputeSpatialHash(samplePosition, sampleNormal, level);
        uint hash = Hash32(hashKey);
        uint slot = (hash % capacity) * 2;
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
};


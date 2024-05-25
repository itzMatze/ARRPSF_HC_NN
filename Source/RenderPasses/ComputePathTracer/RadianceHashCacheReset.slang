import Utils.Debug.PixelDebug;
import RadianceHashCacheHashGridCommon;
import RadianceHashCacheCommon;

#if R_HC_UPDATE || R_HC_QUERY

static const uint kHashCacheHashMapSize = R_HC_HASHMAP_SIZE;

[numthreads(128, 1, 1)]
void main(uint3 dispatchThreadId: SV_DispatchThreadID)
{
    if (dispatchThreadId.x >= kHashCacheHashMapSize) return;
    rhc::hashCacheResetVoxelData(dispatchThreadId.x);
    rhc::HashMapReset(dispatchThreadId.x);
}
#endif // R_HC_UPDATE || R_HC_QUERY


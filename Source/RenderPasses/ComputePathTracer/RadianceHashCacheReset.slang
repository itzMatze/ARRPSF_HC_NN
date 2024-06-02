import Utils.Debug.PixelDebug;
import RadianceHashCacheHashGridCommon;
import RadianceHashCacheCommon;

#if HC_UPDATE || HC_QUERY

static const uint kHashCacheHashMapSize = HC_HASHMAP_SIZE;

[numthreads(128, 1, 1)]
void main(uint3 dispatchThreadId: SV_DispatchThreadID)
{
    if (dispatchThreadId.x >= kHashCacheHashMapSize) return;
    hc::hashCacheResetVoxelData(dispatchThreadId.x);
    hc::HashMapReset(dispatchThreadId.x);
}
#endif // HC_UPDATE || HC_QUERY


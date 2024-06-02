import RadianceHashCacheCommon;
import Utils.Debug.PixelDebug;

#if HC_UPDATE || HC_QUERY

static const uint kHashCacheHashMapSize = HC_HASHMAP_SIZE;

[numthreads(128, 1, 1)]
void hashCacheResolve(uint3 dispatchThreadId: SV_DispatchThreadID)
{
    if (dispatchThreadId.x >= kHashCacheHashMapSize) return;
    hc::hashCacheCombine(dispatchThreadId.x);
}
#endif // HC_UPDATE || HC_QUERY


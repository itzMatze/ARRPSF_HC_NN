import HashCacheCommon;
import Utils.Debug.PixelDebug;

#if HC_UPDATE || HC_QUERY
cbuffer CB
{
    float3 gCamPos;
    float3 gPrevCamPos;
    float gSceneScale;
}

static const uint kHashCacheHashMapSize = HC_HASHMAP_SIZE;

[numthreads(128, 1, 1)]
void hashCacheResolve(uint3 dispatchThreadId: SV_DispatchThreadID)
{
    if (dispatchThreadId.x >= kHashCacheHashMapSize) return;
    hashCacheCombine(dispatchThreadId.x);
}
#endif // HC_UPDATE || HC_QUERY


import HashCacheCommon;
import Utils.Debug.PixelDebug;

#if HASH_CACHE_UPDATE || HASH_CACHE_QUERY
cbuffer CB
{
    float3 gCamPos;
    float3 gPrevCamPos;
    float gSceneScale;
}

static const uint kHashCacheHashMapSize = HASH_CACHE_HASHMAP_SIZE;

[numthreads(128, 1, 1)]
void hashCacheResolve(uint3 dispatchThreadId: SV_DispatchThreadID)
{
    if (dispatchThreadId.x >= kHashCacheHashMapSize) return;
    hashCacheCombine(dispatchThreadId.x);
}
#endif // HASH_CACHE_UPDATE || HASH_CACHE_QUERY


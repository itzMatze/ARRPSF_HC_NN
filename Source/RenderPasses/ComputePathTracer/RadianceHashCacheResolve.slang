import RadianceHashCacheCommon;
import Utils.Debug.PixelDebug;

#if R_HC_UPDATE || R_HC_QUERY
cbuffer CB
{
    float3 gCamPos;
    float3 gPrevCamPos;
    float gSceneScale;
}

static const uint kHashCacheHashMapSize = R_HC_HASHMAP_SIZE;

[numthreads(128, 1, 1)]
void hashCacheResolve(uint3 dispatchThreadId: SV_DispatchThreadID)
{
    if (dispatchThreadId.x >= kHashCacheHashMapSize) return;
    hashCacheCombine(dispatchThreadId.x);
}
#endif // R_HC_UPDATE || R_HC_QUERY


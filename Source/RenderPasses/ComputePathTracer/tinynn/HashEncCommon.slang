#include "TinynnTensorview.hlsli"
import HashEncHashGridCommon;
import Utils.Debug.PixelDebug;

static const uint kFeatureHashGridSize = FEATURE_HASH_GRID_SIZE;
static const uint kFeatureHashGridPlacesPerElement = FEATURE_HASH_GRID_PLACES_PER_ELEMENT;

struct FeatureHashGrid
{
    FeatureHashGridData hashmap;
    TensorView dataView;

    __init(inout uint param_offset, inout uint grad_offset)
    {
        uint capacity = kFeatureHashGridSize / kFeatureHashGridPlacesPerElement;
        hashmap = FeatureHashGridData(capacity);
        dataView = TensorView(param_offset, grad_offset, 32, 32);
        param_offset += kFeatureHashGridSize * kFeatureHashGridPlacesPerElement;
        grad_offset += kFeatureHashGridSize * kFeatureHashGridPlacesPerElement;
    }

    uint FindEntry(float3 samplePosition, float3 sampleDirection, float3 sampleNormal, uint level)
    {
        return hashmap.FindEntry(samplePosition, sampleDirection, sampleNormal, level) * kFeatureHashGridPlacesPerElement;
    }
}


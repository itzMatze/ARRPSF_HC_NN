import Utils.Debug.PixelDebug;
import Utils.Sampling.SampleGenerator;
import HashEncHashGridCommon;

RWStructuredBuffer<float> PrimalBuffer;
RWStructuredBuffer<float> FilteredPrimalBuffer;
RWStructuredBuffer<float> GradientAuxBuffer;

static const uint kParamCount = NN_PARAM_COUNT;
static const uint kGradientAuxElements = NN_GRADIENT_AUX_ELEMENTS;
static const uint kFeatureHashGridSize = FEATURE_HASH_GRID_SIZE;
static const uint kFeatureHashGridPlacesPerElement = FEATURE_HASH_GRID_PLACES_PER_ELEMENT;
static const float kWeightInitLowerBound = NN_WEIGHT_INIT_LOWER_BOUND;
static const float kWeightInitUpperBound = NN_WEIGHT_INIT_UPPER_BOUND;

[shader("compute")]
[numthreads(256, 1, 1)]
void main(int3 dtid: SV_DispatchThreadID) {
    SampleGenerator sg = SampleGenerator(dtid.x, 1);
    if (dtid.x < kParamCount)
    {
        float rnd = sampleNext1D(sg);
        rnd *= (kWeightInitUpperBound - kWeightInitLowerBound);
        rnd += kWeightInitLowerBound;
        PrimalBuffer[dtid.x] = rnd;
        FilteredPrimalBuffer[dtid.x] = rnd;
    }
    if (dtid.x < kGradientAuxElements)
    {
        GradientAuxBuffer[dtid.x] = 0.0;
    }
    if (dtid.x < kFeatureHashGridSize / kFeatureHashGridPlacesPerElement)
    {
        FeatureHashMapReset(dtid.x);
    }
}


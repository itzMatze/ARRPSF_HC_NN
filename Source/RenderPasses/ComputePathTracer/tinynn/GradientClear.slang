import Utils.Debug.PixelDebug;

RWStructuredBuffer<float> GradientBuffer;
RWStructuredBuffer<float> GradientCountBuffer;

static const uint kParamCount = NN_PARAM_COUNT;

// Error sum buffer
//RWByteAddressBuffer u_error_sum;

[shader("compute")]
[numthreads(256, 1, 1)]
void main(int3 dtid: SV_DispatchThreadID) {
    //if (dtid.x == 0) u_error_sum.Store<float>(0, 0.f);
    if (any(dtid.x >= kParamCount)) return;
    printSetPixel(uint2(10000, dtid.x));
    // Clear the parameter gradients
    GradientBuffer[dtid.x] = 0.0;
    GradientCountBuffer[dtid.x] = 0.0;
}


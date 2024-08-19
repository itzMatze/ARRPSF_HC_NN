#ifndef _SRENDERER_ADDON_HALF_TINYNN_MLP_HLSLI_HEADER_
#define _SRENDERER_ADDON_HALF_TINYNN_MLP_HLSLI_HEADER_

#include "TinynnActivations.slang"
#include "TinynnHalfLinear.slang"

struct MLPHalf16X16<let N : int, Act : IActivationFn> : ISharedMem<16>
{
    typedef HalfFeature<16> Input;
    typedef HalfFeature<16> Output;
    LinearHalf16X16 linears[N];

    __init(inout uint offset_prim, inout uint offset_grad, ThreadInfo threadInfo) {
        this.threadInfo = threadInfo;
        [ForceUnroll] for (int i = 0; i < N; i++)
        linears[i] = LinearHalf16X16(offset_prim, offset_grad, threadInfo);
    }

    [Differentiable]
    Output _forward(Input input) {
        HalfFeature<16> out_feature = input;
        [ForceUnroll] for (int i = 0; i < N; i++) {
            out_feature = LinearHalf16X16.eval(linears[i], out_feature);
            [ForceUnroll] for (int j = 0; j < 16; j++)
            out_feature.vals[j] = Act.eval(out_feature.vals[j]); }
        return out_feature; }

    [Differentiable]
    static Output forward(no_diff MLPHalf16X16<N, Act> mlp, Input input) {
        return mlp._forward(input); }
}

struct MLPHalf32X32<let N : int, Act : IActivationFn>: ISharedMem<32>
{
    typedef HalfFeature<32> Input;
    typedef HalfFeature<32> Output;
    LinearHalf32X32 linears[N];
    uint64_t weights_address;
    bool     bWeightsPreloaded;


    __init(inout uint offset_prim, inout uint offset_grad, ThreadInfo threadInfo, uint64_t weights_address=0)
    {
        this.threadInfo = threadInfo;
        bWeightsPreloaded = false;
        
        this.weights_address = weights_address + offset_prim*2;
        if (weights_address == 0)
            this.weights_address = 0;
        [ForceUnroll] for (int i = 0; i < N; i++){
            uint prevOffset = offset_prim;
            linears[i] = LinearHalf32X32(offset_prim, offset_grad, threadInfo);
            linears[i].setWeightsAddress(weights_address + prevOffset * 2);
        }
    }

    
    [Differentiable]
    Output _forward(Input input) {
        HalfFeature<32> out_feature = input;
        [ForceUnroll] for (int i = 0; i < N; i++) {
            out_feature = LinearHalf32X32.eval(linears[i], out_feature);
            [ForceUnroll] for (int j = 0; j < 32; j++)
            out_feature.vals[j] = Act.eval(out_feature.vals[j]);
            }
        return out_feature;
    }


    [mutating] void preload_weights<let NWarps : int>()
    {
        [ForceUnroll] 
        for (int p = 0; p < N; p++)
        {   
            uint inputs_offset = 32 * 32 * NWarps;
            uint layer_offset = p * 32 * 32;
            uint offset = inputs_offset+layer_offset;
            linears[p].preload_weights<32, NWarps>(offset);       
        }
        GroupMemoryBarrierWithGroupSync();
        bWeightsPreloaded = true;
    }

    
    Output _forward_fast(Input input)
    {
        /*Output dummy;
        return dummy;*/
        if (this.weights_address == 0)
        {
            Output dummy;
            dummy.init(-1);

            return dummy;
        }
        HalfFeature<32> out_feature = input;

        moveInputsToSharedMem<32, true>(input.vals);
        if (bWeightsPreloaded)
            __inline_wmma_128_32_32_fused_preloadedweights(this.weights_address, N);
        else
            __inline_wmma_128_32_32_fused(this.weights_address, N);

        // Move the output to local memory.
        Output output;
        const SharedMemRef outPtr = calcOffset<32 * 32>();
        moveOutputsToLocalArray<32, true>(outPtr, output.vals);
        return output;
    }

    [Differentiable]
    static Output forward(no_diff MLPHalf32X32<N, Act> mlp, Input input) {
        return mlp._forward(input); }

    static Output forward_fast(no_diff MLPHalf32X32<N, Act> mlp, Input input) {
            return mlp._forward_fast(input);
    }
}

#endif // !_SRENDERER_ADDON_HALF_TINYNN_LINEAR_HLSLI_HEADER_

#ifndef _SRENDERER_ADDON_HALF_TINYNN_LINEAR_HLSLI_HEADER_
#define _SRENDERER_ADDON_HALF_TINYNN_LINEAR_HLSLI_HEADER_

#include "HalfMatmulInclude.glsl"
#include "TinynnTensorview.slang"


struct ISharedMem<let C : int>
{
    typedef uint SharedMemRef;

    no_diff ThreadInfo threadInfo;

    uint calcOffset<let N : int>()
    { // thread_idx.x = from 0 to 32,  = threadInfo.thread_idx.y from 0 tp 4
        return uint(((threadInfo.thread_idx.x / 32) + threadInfo.thread_idx.y * (threadInfo.block_dim.x * 1.0 / 32)) * N);
    }

    // Load the output array from shared memory.
    void loadArray<let N : int, let colMajor : bool>(SharedMemRef memptr, out float16_t input[N])
    {
        const uint threadIdInWarp = threadInfo.thread_idx.x % 32;
        // Each thread in the warp will move N contiguous elements from their corresponding shared memory.
        if (!colMajor)
        {
            [ForceUnroll] for (int i = 0; i < N; i++) // rowMajor matrix loading
                input[i] = __inline_get_half_shared_buffer(memptr + threadIdInWarp * N + i);
        }
        else
        {
            [ForceUnroll] for (int i = 0; i < N; i++) // colMajor matrix loading
                input[i] = __inline_get_half_shared_buffer(memptr + i * 32 + threadIdInWarp);
        }
    }

    // Store the input array to the shared memory.
    void storeArray<let N : int, let colMajor : bool>(SharedMemRef memptr, float16_t input[N])
    {
        const uint threadIdInWarp = threadInfo.thread_idx.x % 32;
        // Each thread in the warp will move N contiguous elements to their corresponding shared memory.
        if (!colMajor)
        {
            [ForceUnroll] for (int i = 0; i < N; i++) // rowMajor matrix writing
                __inline_set_half_shared_buffer(memptr + (threadIdInWarp * N + i), float16_t(input[i]));
            // checking for the bank conflicts:
            // 0th thread = [0, 1, 2, 3, 4, 5, 6, 7, 8,....]
            // 1th thread = [32, 33, 34, 35, 36.....]
            // BANK CONFLICTS
        }
        else
        {
            [ForceUnroll] for (int i = 0; i < N; i++) // colMajor matrix writing
                __inline_set_half_shared_buffer(memptr + (i * 32 + threadIdInWarp), float16_t(input[i]));
            // checking for the bank conflicts:
            // 0th thread = [0, 32, 64, 96....]
            // 1th thread = [1, 33, 65, 97]
            // NO BANK CONFLICTS
        }
    }


    // Move the input array to the shared memory.
    void moveInputsToSharedMem<let N : int, let colMajor : bool>(float16_t input[N])
    {
        // Pack in row-major format.
        SharedMemRef inPtr = calcOffset<32 * C>();
        ;
        storeArray<N, colMajor>(inPtr, input);
    }

    // Load the output array from the shared memory.
    void moveOutputsToLocalArray<let N : int, let colMajor : bool>(SharedMemRef outPtr, out float16_t outputs[N])
    {
        loadArray<N, colMajor>(outPtr, outputs);
        [ForceUnroll] for (int i = 0; i < N; i++) outputs[i] = outputs[i];
    }

};

// TODO: FIX CODE DUPLICATION ISharedMem. It is caused by some unexepected compilations errors if we inheret ISharedMem
// Base of all half linear layers.
struct LinearHalf<let C : int> // : ISharedMem<C>
{
    typedef HalfFeature<C> Input;
    typedef HalfFeature<C> Output;
 
    no_diff TensorView weights_view;
    no_diff TensorView bias_view;

    typedef uint SharedMemRef;

    no_diff ThreadInfo threadInfo;

    uint calcOffset<let N : int>()
    { // thread_idx.x = from 0 to 32,  = threadInfo.thread_idx.y from 0 tp 4 
        return uint(((threadInfo.thread_idx.x / 32) + threadInfo.thread_idx.y * (threadInfo.block_dim.x * 1.0 / 32)) * N);
    }

    // Load the output array from shared memory.
    void loadArray<let N : int, let colMajor : bool>(SharedMemRef memptr, out float16_t input[N])
    {

        const uint threadIdInWarp = threadInfo.thread_idx.x % 32;
        // Each thread in the warp will move N contiguous elements from their corresponding shared memory.
        if (!colMajor)
        {
            [ForceUnroll] for (int i = 0; i < N; i++) // rowMajor matrix loading
                input[i] = __inline_get_half_shared_buffer(memptr + threadIdInWarp * N + i);
        }
        else
        {
            [ForceUnroll] for (int i = 0; i < N; i++) // colMajor matrix loading
                input[i] = __inline_get_half_shared_buffer(memptr + i * 32 + threadIdInWarp);
        }
    }

    // Store the input array to the shared memory.
    void storeArray<let N : int, let colMajor : bool>(SharedMemRef memptr, float16_t input[N])
    {
        const uint threadIdInWarp = threadInfo.thread_idx.x % 32;
        // Each thread in the warp will move N contiguous elements to their corresponding shared memory.
        if (!colMajor)
        {
            [ForceUnroll] for (int i = 0; i < N; i++) // rowMajor matrix writing
                __inline_set_half_shared_buffer(memptr + (threadIdInWarp * N + i), float16_t(input[i]));
            // checking for the bank conflicts:
            // 0th thread = [0, 1, 2, 3, 4, 5, 6, 7, 8,....]
            // 1th thread = [32, 33, 34, 35, 36.....]
        }
        else
        {
            [ForceUnroll] for (int i = 0; i < N; i++) // colMajor matrix writing
                __inline_set_half_shared_buffer(memptr + (i * 32 + threadIdInWarp), float16_t(input[i]));
        }
    }

    void preload_weights<let N : int, let NWarps : int>(SharedMemRef memptr)
    {
        const uint num_elements_per_warp = N / NWarps;
        const uint warp_offset = num_elements_per_warp * (threadInfo.thread_idx.y % NWarps);
        const uint threadIdInWarp = threadInfo.thread_idx.x % 32;
        [ForceUnroll] for (int i = 0; i < num_elements_per_warp; i++)
        { // colMajor matrix writing

            uint element_id = (i + warp_offset) * 32 + threadIdInWarp;
            __inline_set_half_shared_buffer(memptr + element_id, weights_view.load_prim(element_id));
        }
    }

    // Move the input array to the shared memory.
    void moveInputsToSharedMem<let N : int>(float16_t input[N])
    {
        // Pack in row-major format.
        SharedMemRef inPtr = calcOffset<32 * C>();
        ;
        storeArray<N, false>(inPtr, input);
    }

    // Load the output array from the shared memory.
    void moveOutputsToLocalArray<let N : int>(SharedMemRef outPtr, out float16_t outputs[N])
    {
        loadArray<N, false>(outPtr, outputs);
        [ForceUnroll] for (int i = 0; i < N; i++) outputs[i] = outputs[i];
    }

}

struct LinearHalf16X16 : LinearHalf<16> {
    __init(inout uint offset_prim, inout uint offset_grad, ThreadInfo threadInfo) {
        this.weights_view = TensorView(offset_prim, offset_grad, 16, 16);
        offset_prim += 256; offset_grad += 256;
        this.threadInfo = threadInfo;
    }

    // move the weights from global memory to shared memory.
    void moveWeightsToSharedMem<let colMajor : bool>() {
        const SharedMemRef wtPtr = 2048 + calcOffset<16 * 16>();
        const int2 threadIdx = threadInfo.thread_idx;
        // Copy weights to shared memory.
        const int i_base = threadIdx.x % 16;
        const int j_base = (threadIdx.x / 16) * 8;
        [ForceUnroll] for (uint j = 0; j < 8; j++) {
            const float16_t w = float16_t(weights_view.load_prim(i_base, j + j_base));
            if (colMajor) __inline_set_half_shared_buffer(wtPtr + i_base * 16 + j + j_base, w);
            else __inline_set_half_shared_buffer(wtPtr + (j + j_base) * 16 + i_base, w);
        }
    }

    Output _eval(Input in_feature) {
        // Move the input and weights to shared memory.
        moveInputsToSharedMem<16>(in_feature.vals);
        moveWeightsToSharedMem<false>();
        // Do the matmul.
        __inline_wmma_128_16_16();
        // Move the output to local memory.
        Output out_feature;
        const SharedMemRef outPtr = 3072 + calcOffset<32 * 16>();
        moveOutputsToLocalArray<16>(outPtr, out_feature.vals);
        // output the result.
        return out_feature;
    }

    void _eval_bwd(inout DifferentialPair<Input> in_feature_pair, Output.Differential d_output) {
        // Accumulate input derivatives. dodi
        // which is simply do*W^T
        GroupMemoryBarrierWithGroupSync();
        {   moveInputsToSharedMem<16>(d_output.vals);
            moveWeightsToSharedMem<true>();
            SharedMemRef dInPtr = 3072 + calcOffset<32 * 16>();
            __inline_wmma_128_16_16();
            Input.Differential d_input_feature;
            loadArray<16, false>(dInPtr, d_input_feature.vals);
            in_feature_pair = DifferentialPair<Input>(in_feature_pair.p, d_input_feature);
        }
        // Accumulate weight derivatives.
        // which involves some transpose and matrix multiplication again.
        GroupMemoryBarrierWithGroupSync();
        {   uint inputPtr = calcOffset<32 * 16>();
            storeArray<16, true>(inputPtr, d_output.vals);
            uint outPtr = 2048 + calcOffset<32 * 16>();
            [ForceUnroll] for (int i = 0; i < 16; i++)
            __inline_set_half_shared_buffer(
                outPtr + i * 32 + threadInfo.thread_idx.x,
                float16_t(in_feature_pair.p.vals[i]));
            __inline_wmma_16_128_16();

            uint wtPtr = 4096 + calcOffset<16 * 16>();
            // Copy weights to shared memory.
            const int i_base = threadInfo.thread_idx.x % 16;
            const int j_base = (threadInfo.thread_idx.x / 16) * 8;
            [ForceUnroll] for (uint j = 0; j < 8; j++) {
              float weight_grad = __inline_get_half_shared_buffer(wtPtr + i_base * 16 + j + j_base);
              weights_view.interlocked_add_grad(j + j_base, i_base, weight_grad); }
        }
        
    }

    [BackwardDerivative(eval_bwd)]
    static Output eval(LinearHalf16X16 layer, Input in_feature) {
        return layer._eval(in_feature); }
    static void eval_bwd(LinearHalf16X16 layer,
        inout DifferentialPair<Input> in_feature_pair,
        Output.Differential d_output) {
        return layer._eval_bwd(in_feature_pair, d_output); }
}

struct LinearHalf32X32 : LinearHalf<32> {

    __init(inout uint offset_prim, inout uint offset_grad, ThreadInfo threadInfo) {
        this.weights_view = TensorView(offset_prim, offset_grad, 32, 32);
        offset_prim += 1024; offset_grad += 1024;

        this.threadInfo = threadInfo;
    }

    
    // move the weights from global memory to shared memory.
    void moveWeightsToSharedMem<let colMajor : bool>() {
        SharedMemRef wtPtr = 4096 + calcOffset<32 * 32>();


        const int threadIdInWarp = threadInfo.thread_idx.x;
        const int warpId = threadInfo.thread_idx.y;

        /*
        each thread processes its own row\column.
        the weights in the memory are stored as row. each thread = its own thread.
        then we transpose it
        */
        // Copy weights to shared memory.
        [ForceUnroll] for (uint j = 0; j < 32; j++) {
            const float16_t w = weights_view.load_prim(threadIdInWarp, j); // weights[threadIdInWarp*32+j]
          if (colMajor) __inline_set_half_shared_buffer(wtPtr + threadIdInWarp * 32 + j, w);
          else __inline_set_half_shared_buffer(wtPtr + j * 32 + threadIdInWarp, w); // j*32+threadIdInWarp
        }
    }

    // move the weights from global memory to shared memory.
    void moveWeightsToSharedMemOur<let colMajor : bool>()
    {
        SharedMemRef wtPtr = 4096 + calcOffset<32 * 32>();

        
          int n = 4096 + threadInfo.thread_idx.y * 1024;
          if (n != 4096)
            return;
          wtPtr = n;
        
        const int threadIdInWarp = threadInfo.thread_idx.x;
        const int warpId = threadInfo.thread_idx.y;

        /*
        each thread processes its own row\column.
        the weights in the memory are stored as row. each thread = its own thread.
        then we transpose it
        */
        // Copy weights to shared memory.
        [ForceUnroll] for (uint j = 0; j < 32; j++)
        {
          const float16_t w = weights_view.load_prim(threadIdInWarp, j); // weights[threadIdInWarp*32+j]
          if (colMajor)
            __inline_set_half_shared_buffer(wtPtr + threadIdInWarp * 32 + j, w);
          else
            __inline_set_half_shared_buffer(wtPtr + j * 32 + threadIdInWarp, w); // j*32+threadIdInWarp
        }
    }

    uint64_t weights_address;

    [mutating] void setWeightsAddress(uint64_t address) {
        weights_address = address;
    }

    Output _eval(Input in_feature) {
        GroupMemoryBarrierWithGroupSync();
        // Move the input and weights to shared memory.
        moveInputsToSharedMem<32>(in_feature.vals);

      
        
        // Do the matmul.
        #if 0
        moveWeightsToSharedMem<false>();
        GroupMemoryBarrierWithGroupSync();
        __inline_wmma_128_32_32();
        #else
        GroupMemoryBarrierWithGroupSync();
        __inline_wmma_128_32_32_weightsvram(weights_address);
        #endif
      

        GroupMemoryBarrierWithGroupSync();
        // Move the output to local memory.
        Output out_feature;
        const SharedMemRef outPtr = calcOffset<32 * 32>();
        moveOutputsToLocalArray<32>(outPtr, out_feature.vals);
        // output the result.
        return out_feature;
    }

    void _eval_bwd(inout DifferentialPair<Input> in_feature_pair, Output.Differential d_output) {
        // Accumulate input derivatives. dodi
        // which is simply do*W^T
        // TO DO: ACCELERATE IT FUTHER!!!
        GroupMemoryBarrierWithGroupSync();
        { moveInputsToSharedMem<32>(d_output.vals);
          moveWeightsToSharedMem<true>();
          GroupMemoryBarrierWithGroupSync();
          // Do the matmul.
          __inline_wmma_128_32_32();
          Input.Differential d_input_feature;
          loadArray<32, false>(calcOffset<32 * 32>(), d_input_feature.vals);
          in_feature_pair = DifferentialPair<Input>(in_feature_pair.p, d_input_feature);
        }
        // Accumulate weight derivatives.
        // which involves some transpose and matrix multiplication again.
        {
          GroupMemoryBarrierWithGroupSync();
          uint inputPtr = calcOffset<32 * 32>();
          storeArray<32, true>(inputPtr, d_output.vals);
          uint outPtr = 4096 + calcOffset<32 * 32>();
          [ForceUnroll] for (int i = 0; i < 32; i++)
            __inline_set_half_shared_buffer(
                outPtr + i * 32 + threadInfo.thread_idx.x,
                float16_t(in_feature_pair.p.vals[i]));
          GroupMemoryBarrierWithGroupSync();
          __inline_wmma_32_128_32();
          GroupMemoryBarrierWithGroupSync();
          uint wtPtr = calcOffset<32 * 32>();
          [ForceUnroll] for (uint j = 0; j < 32; j++) {
            var threadIdInWarp = threadInfo.thread_idx.x % 32;
            float weight_grad = __inline_get_half_shared_buffer(wtPtr + threadIdInWarp * 32 + j);
            weights_view.interlocked_add_grad(j, threadIdInWarp, weight_grad); }
        }
        
    }

    [BackwardDerivative(eval_bwd)]
    static Output eval(LinearHalf32X32 layer, Input in_feature) {
        return layer._eval(in_feature);
    }

    static void eval_bwd(LinearHalf32X32 layer,
        inout DifferentialPair<Input> in_feature_pair,
        Output.Differential d_output) {
        return layer._eval_bwd(in_feature_pair, d_output); }
}

#endif // !_SRENDERER_ADDON_HALF_TINYNN_LINEAR_HLSLI_HEADER_

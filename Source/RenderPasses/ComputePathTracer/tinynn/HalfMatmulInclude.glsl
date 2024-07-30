#ifndef _SRENDERER_TINY_MATMUL_GLSLI_HEADER_
#define _SRENDERER_TINY_MATMUL_GLSLI_HEADER_

#extension GL_KHR_memory_scope_semantics: require
#extension GL_EXT_control_flow_attributes: require
#extension GL_NV_cooperative_matrix: require
#extension GL_EXT_shader_explicit_arithmetic_types: require
#extension GL_EXT_shader_explicit_arithmetic_types_float16: require
#extension GL_KHR_shader_subgroup_arithmetic: require
#extension GL_EXT_shader_subgroup_extended_types_float16: require

// fused shared memory for matrix multiplication
shared float16_t glsl_half_shared_buffer[8192];
// set/get the shared buffer
void glsl_set_half_shared_buffer(int i, float16_t value) { glsl_half_shared_buffer[i] = value; }
float16_t glsl_get_half_shared_buffer(int i) { return glsl_half_shared_buffer[i]; }

/**
 * Assume the input matrix is 32x16 per warp,
 * do a matrix multiplication with a 16x16 matrix.
 * The output matrix is 32x16 per warp.
 * All parameters are resident in shared memory, as follows:
 * --- offset = 0 --------------------------------
 * |   32*16  |   32*16  |   32*16   |   32*16   |
 * |  Input 0 |  Input 1 |  Input 2  |  Input 3  |
 * --- offset = 2048 -----------------------------
 * |   16*16  |   16*16  |   16*16   |   16*16   |
 * | Weight 0 | Weight 1 |  Weight 2 | Weight 3  |
 * --- offset = 3072 -----------------------------
 * |   32*16  |   32*16  |   32*16   |   32*16   |
 * | Output 0 | Output 1 | Output 2  | Output 3  |
 * --- offset = 5120 -----------------------------
 */
void glsl_wmma_128_16_16() {
    // load the data from shared memory to the fragment
    fcoopmatNV<16, gl_ScopeSubgroup, 16, 16> act_frag[2];
    fcoopmatNV<16, gl_ScopeSubgroup, 16, 16> weights_frag;
    fcoopmatNV<16, gl_ScopeSubgroup, 16, 16> result_frag[2];
    const uint li = gl_SubgroupInvocationID; // index in warp ("lane index")
    const uint wi = gl_SubgroupID;          // index in block ("warp index")
    // load the activations from shared memory
    coopMatLoadNV(act_frag[0], glsl_half_shared_buffer, wi*512 +   0, 16, false);
    coopMatLoadNV(act_frag[1], glsl_half_shared_buffer, wi*512 + 256, 16, false);
    // load the weights from shared memory
    coopMatLoadNV(weights_frag, glsl_half_shared_buffer, 2048 + wi*256, 16, true);
    // clear the output accumulation matrix
    result_frag[0] = fcoopmatNV<16, gl_ScopeSubgroup, 16, 16>(0.0f);
    result_frag[1] = fcoopmatNV<16, gl_ScopeSubgroup, 16, 16>(0.0f);
    // perform the matrix multiplication
    result_frag[0] = coopMatMulAddNV(act_frag[0], weights_frag, result_frag[0]);
    result_frag[1] = coopMatMulAddNV(act_frag[1], weights_frag, result_frag[1]);
    // Store the output matrix.
    coopMatStoreNV(result_frag[0], glsl_half_shared_buffer, 3072 + wi*512 +   0, 16, false);
    coopMatStoreNV(result_frag[1], glsl_half_shared_buffer, 3072 + wi*512 + 256, 16, false);
}

/**
 * Assume the input matrix is 16*32 per warp,
 * do a matrix multiplication with a 32*16 matrix.
 * The output matrix is 16*16 per warp.
 * All parameters are resident in shared memory, as follows:
 * --- offset = 0 --------------------------------
 * |   16*32  |   16*32  |   16*32   |   16*32   |
 * |  Input 0 |  Input 1 |  Input 2  |  Input 3  |
 * --- offset = 2048 -----------------------------
 * |   32*16  |   32*16  |   32*16   |   32*16   |
 * | Weight 0 | Weight 1 |  Weight 2 | Weight 3  |
 * --- offset = 4096 -----------------------------
 * |   16*16  |   16*16  |   16*16   |   16*16   |
 * | Output 0 | Output 1 | Output 2  | Output 3  |
 * --- offset = 5120 -----------------------------
 */
void glsl_wmma_16_128_16() {
    // load the data from shared memory to the fragment
    fcoopmatNV<16, gl_ScopeSubgroup, 16, 16> act_frag[2];
    fcoopmatNV<16, gl_ScopeSubgroup, 16, 16> weights_frag[2];
    fcoopmatNV<16, gl_ScopeSubgroup, 16, 16> result_frag;
    const uint li = gl_SubgroupInvocationID; // index in warp ("lane index")
    const uint wi = gl_SubgroupID;          // index in block ("warp index")
    // load the activations from shared memory
    coopMatLoadNV(act_frag[0], glsl_half_shared_buffer, wi*512 +  0, 32, false);
    coopMatLoadNV(act_frag[1], glsl_half_shared_buffer, wi*512 + 16, 32, false);
    // load the weights from shared memory
    coopMatLoadNV(weights_frag[0], glsl_half_shared_buffer, 2048 + wi*512 +  0, 32, true);
    coopMatLoadNV(weights_frag[1], glsl_half_shared_buffer, 2048 + wi*512 + 16, 32, true);
    // clear the output accumulation matrix
    result_frag = fcoopmatNV<16, gl_ScopeSubgroup, 16, 16>(0.0f);
    // perform the matrix multiplication
    result_frag = coopMatMulAddNV(act_frag[0], weights_frag[0], result_frag);
    result_frag = coopMatMulAddNV(act_frag[1], weights_frag[1], result_frag);
    // Store the output matrix.
    coopMatStoreNV(result_frag, glsl_half_shared_buffer, 4096 + wi*256, 16, false);
}

/**
 * Assume the input matrix is 32*32 per warp,
 * do a matrix multiplication with a 32*32 matrix.
 * The output matrix is 32*32 per warp.
 * All parameters are resident in shared memory, as follows:
 * --- offset = 0 --------------------------------
 * |   32*32  |   32*32  |   32*32   |   32*32   |
 * |  Input 0 |  Input 1 |  Input 2  |  Input 3  |
 * --- offset = 4096 -----------------------------
 * |   32*32  |   32*32  |   32*32   |   32*32   |
 * | Weight 0 |             padding              |
 * --- offset = 8192 -----------------------------
 * The output overrides the input matrix.
 * --- offset = 0 --------------------------------
 * |   32*32  |   32*32  |   32*32   |   32*32   |
 * | Output 0 | Output 1 |  Output 2 |  Output 3 |
 * --- offset = 4096 -----------------------------
 */





fcoopmatNV<16, gl_ScopeSubgroup, 16, 16> glsl_relu(fcoopmatNV<16, gl_ScopeSubgroup, 16, 16> m){
    for (int i = 0; i < m.length(); ++i) {
            m[i] = float16_t(m[i] > float16_t(0.0f))*m[i];
    }
    return m;
}

void glsl_wmma_128_32_32() {
    // load the data from shared memory to the fragment
    fcoopmatNV<16, gl_ScopeSubgroup, 16, 16> act_frag[4]; // we need 4 of them, because each representes 16x16. but the target tensor is 32x32
    fcoopmatNV<16, gl_ScopeSubgroup, 16, 16> weights_frag[2];
    fcoopmatNV<16, gl_ScopeSubgroup, 16, 16> result_frag[2];
    const uint li = gl_SubgroupInvocationID; // index in warp ("lane index")
    const uint wi = gl_SubgroupID*0;          // index in block ("warp index")

    // // load the activations from shared memory
    // coopMatLoadNV(act_frag[0], glsl_half_shared_buffer, 32*32*wi +   0, 32, false);
    // coopMatLoadNV(act_frag[1], glsl_half_shared_buffer, 32*32*wi +  16, 32, false);
    // coopMatLoadNV(act_frag[2], glsl_half_shared_buffer, 32*32*wi + 512, 32, false);
    // coopMatLoadNV(act_frag[3], glsl_half_shared_buffer, 32*32*wi + 528, 32, false);
    // // synchronize the threads in the group, so that the input matrix is all loaded
    // barrier();
    // // load the weights from shared memory
    // coopMatLoadNV(weights_frag[0], glsl_half_shared_buffer, 4096 + 32*32*wi +  0, 32, true);
    // coopMatLoadNV(weights_frag[1], glsl_half_shared_buffer, 4096 + 32*32*wi + 16, 32, true);
    // // clear the output accumulation matrix
    // result_frag[0] = fcoopmatNV<16, gl_ScopeSubgroup, 16, 16>(0.0f);
    // result_frag[1] = fcoopmatNV<16, gl_ScopeSubgroup, 16, 16>(0.0f);
    // // // Load 2 chunks of weights from shared memory into registers.
    // result_frag[0] = coopMatMulAddNV(act_frag[0], weights_frag[0], result_frag[0]);
    // result_frag[0] = coopMatMulAddNV(act_frag[1], weights_frag[1], result_frag[0]);
    // result_frag[1] = coopMatMulAddNV(act_frag[2], weights_frag[0], result_frag[1]);
    // result_frag[1] = coopMatMulAddNV(act_frag[3], weights_frag[1], result_frag[1]);
    // // Store the output matrix.
    // coopMatStoreNV(result_frag[0], glsl_half_shared_buffer, wi*32*32 +   0, 32, false);
    // coopMatStoreNV(result_frag[1], glsl_half_shared_buffer, wi*32*32 + 512, 32, false);
    // // load the weights from shared memory
    // coopMatLoadNV(weights_frag[0], glsl_half_shared_buffer, 4096 + 32*32*wi + 512, 32, true);
    // coopMatLoadNV(weights_frag[1], glsl_half_shared_buffer, 4096 + 32*32*wi + 528, 32, true);
    // // clear the output accumulation matrix
    // result_frag[0] = fcoopmatNV<16, gl_ScopeSubgroup, 16, 16>(0.0f);
    // result_frag[1] = fcoopmatNV<16, gl_ScopeSubgroup, 16, 16>(0.0f);
    // // // Load 2 chunks of weights from shared memory into registers.
    // result_frag[0] = coopMatMulAddNV(act_frag[0], weights_frag[0], result_frag[0]);
    // result_frag[0] = coopMatMulAddNV(act_frag[1], weights_frag[1], result_frag[0]);


    // result_frag[1] = coopMatMulAddNV(act_frag[2], weights_frag[0], result_frag[1]);
    // result_frag[1] = coopMatMulAddNV(act_frag[3], weights_frag[1], result_frag[1]);

    // // synchronize the threads in the warp, so that the output matrix is correct
    // coopMatStoreNV(result_frag[0], glsl_half_shared_buffer, wi*32*32 +  16, 32, false);
    // coopMatStoreNV(result_frag[1], glsl_half_shared_buffer, wi*32*32 + 528, 32, false);
}

// void infer(int layers){
//     barrier();
//     // Move the input and weights to shared memory.
//     moveInputsToSharedMem<32>(in_feature.vals); // I need to figure out how it works
//     // load the data from shared memory to the fragment

//     const uint li = gl_SubgroupInvocationID; // index in warp ("lane index")
//     const uint wi = gl_SubgroupID;          // index in block ("warp index")
//         // load the activations from shared memory
//     fcoopmatNV<16, gl_ScopeSubgroup, 16, 16> zeros = fcoopmatNV<16, gl_ScopeSubgroup, 16, 16>(0.0f);

//     for(int i=0; i < layers; i++){
//         moveWeightsToSharedMem<false>(); // I need to figure out how it works. We don't need to store the weights in the local shared memory at all
//         barrier();

//         fcoopmatNV<16, gl_ScopeSubgroup, 16, 16> weights_frag[4];
//         fcoopmatNV<16, gl_ScopeSubgroup, 16, 16> input_frag[4];
//         fcoopmatNV<16, gl_ScopeSubgroup, 16, 16> output_frag[2];
//         coopMatLoadNV(input_frag[0], glsl_half_shared_buffer, 32*32*wi +   0, 32, false);
//         coopMatLoadNV(input_frag[1], glsl_half_shared_buffer, 32*32*wi +  16, 32, false);

//         // synchronize the threads in the group, so that the input matrix is all loaded

//         /*
//             out goal is to compute a vector of 32 elements per each thread. # threads = 32.
//             so we should process 32x32 elements

//             output_flag[0] = the first 16 output neurons for the first 16 threads
//             output_flag[1] = the first 16 output neurons for the second 16 threads

//             output_flag[2] = the second 16 output neurons for the first 16 threads
//             output_flag[3] = the second 16 output neurons for the second 16 threads

//             firstly we compute first 16 elements of each vector. so we load only subset of the weight matrix
//         */

//         const int weights_offset = i*1024; // weights matrix = 32x32 = 1024. i - layer's idex

//         // we load only a 16x32 weight matrix, so we could estimate the first half of the vector
//         coopMatLoadNV(weights_frag[0], glsl_half_shared_buffer, 4096 + 32*32*wi +  0+weights_offset, 32, true);
//         coopMatLoadNV(weights_frag[1], glsl_half_shared_buffer, 4096 + 32*32*wi + 16+weights_offset, 32, true);
//         // then we load the 2nd 16x32 weight matrix, so we could estimate the second half of the vector
//         coopMatLoadNV(weights_frag[2], glsl_half_shared_buffer, 4096 + 32*32*wi + 512+weights_offset, 32, true);
//         coopMatLoadNV(weights_frag[3], glsl_half_shared_buffer, 4096 + 32*32*wi + 528+weights_offset, 32, true);


//         output_frag[0] = fcoopmatNV<16, gl_ScopeSubgroup, 16, 16>(0.0f);
//         output_frag[0] = coopMatMulAddNV(input_frag[0], weights_frag[0], output_frag[0]);
//         output_frag[0] = coopMatMulAddNV(input_frag[1], weights_frag[1], output_frag[0]);
//         output_frag[0] = glsl_relu(output_frag[0]); // apply activation. we can do it because of the uniform broadcasting property.


//         output_frag[1] = fcoopmatNV<16, gl_ScopeSubgroup, 16, 16>(0.0f);
//         output_frag[1] = coopMatMulAddNV(input_frag[2], weights_frag[0], output_frag[1]);
//         output_frag[1] = coopMatMulAddNV(input_frag[3], weights_frag[1], output_frag[1]);
//         output_frag[1] = glsl_relu(output_frag[1]);
//         coopMatStoreNV(output_frag[0], glsl_half_shared_buffer, wi*32*32 +   0, 32, false);
//         coopMatStoreNV(output_frag[1], glsl_half_shared_buffer, wi*32*32 + 512, 32, false);


//         coopMatLoadNV(input_frag[2], glsl_half_shared_buffer, 32*32*wi + 512, 32, false);
//         coopMatLoadNV(input_frag[3], glsl_half_shared_buffer, 32*32*wi + 528, 32, false);


//         // // Load 2 chunks of weights from shared memory into registers.
//         output_frag[0] = fcoopmatNV<16, gl_ScopeSubgroup, 16, 16>(0.0f);
//         output_frag[0] = coopMatMulAddNV(input_frag[0], weights_frag[2], output_frag[0]);
//         output_frag[0] = coopMatMulAddNV(input_frag[1], weights_frag[3], output_frag[0]);
//         output_frag[0] = glsl_relu(output_frag[0]);


//         output_frag[1] = fcoopmatNV<16, gl_ScopeSubgroup, 16, 16>(0.0f);
//         output_frag[1] = coopMatMulAddNV(input_frag[2], weights_frag[2], output_frag[1]);
//         output_frag[1] = coopMatMulAddNV(input_frag[3], weights_frag[3], output_frag[0]);
//         output_frag[1] = glsl_relu(output_frag[3]);
//         coopMatStoreNV(output_frag[0], glsl_half_shared_buffer, wi*32*32 +  16, 32, false);
//         coopMatStoreNV(output_frag[1], glsl_half_shared_buffer, wi*32*32 + 528, 32, false);

//         barrier();
//     }


//     GroupMemoryBarrierWithGroupSync();
//     Output out_feature;
//     const SharedMemRef outPtr = calcOffset<32 * 32>();
//     moveOutputsToLocalArray<32>(outPtr, out_feature.vals);
//     // output the result.
//     return out_feature;
// }


/**
 * Assume the input matrix is 32*32 per warp,
 * do a matrix multiplication with a 32*32 matrix.
 * The output matrix is 32*32 per warp.
 * All parameters are resident in shared memory, as follows:
 * --- offset = 0 --------------------------------
 * |   32*32  |   32*32  |   32*32   |   32*32   |
 * |  Input 0 |  Input 1 |  Input 2  |  Input 3  |
 * --- offset = 4096 -----------------------------
 * |   32*32  |   32*32  |   32*32   |   32*32   |
 * | Weight 0 | Weight 1 |  Weight 2 | Weight 3  |
 * --- offset = 8192 -----------------------------
 * The output overrides the input matrix.
 * --- offset = 0 --------------------------------
 * |   32*32  |   32*32  |   32*32   |   32*32   |
 * | Output 0 | Output 1 |  Output 2 |  Output 3 |
 * --- offset = 4096 -----------------------------
 */
void glsl_wmma_32_128_32() {
    // load the data from shared memory to the fragment
    fcoopmatNV<16, gl_ScopeSubgroup, 16, 16> act_frag[4];
    fcoopmatNV<16, gl_ScopeSubgroup, 16, 16> weights_frag[4];
    fcoopmatNV<16, gl_ScopeSubgroup, 16, 16> result_frag[4];
    const uint li = gl_SubgroupInvocationID; // index in warp ("lane index")
    const uint wi = gl_SubgroupID;          // index in block ("warp index")
    // load the activations from shared memory
    coopMatLoadNV(act_frag[0], glsl_half_shared_buffer, 32*32*wi +   0, 32, false);
    coopMatLoadNV(act_frag[1], glsl_half_shared_buffer, 32*32*wi +  16, 32, false);
    coopMatLoadNV(act_frag[2], glsl_half_shared_buffer, 32*32*wi + 512, 32, false);
    coopMatLoadNV(act_frag[3], glsl_half_shared_buffer, 32*32*wi + 528, 32, false);
    // load the weights from shared memory
    coopMatLoadNV(weights_frag[0], glsl_half_shared_buffer, 4096 + 32*32*wi +   0, 32, true);
    coopMatLoadNV(weights_frag[1], glsl_half_shared_buffer, 4096 + 32*32*wi +  16, 32, true);
    coopMatLoadNV(weights_frag[2], glsl_half_shared_buffer, 4096 + 32*32*wi + 512, 32, true);
    coopMatLoadNV(weights_frag[3], glsl_half_shared_buffer, 4096 + 32*32*wi + 528, 32, true);
    // clear the output accumulation matrix
    result_frag[0] = fcoopmatNV<16, gl_ScopeSubgroup, 16, 16>(0.0f);
    result_frag[1] = fcoopmatNV<16, gl_ScopeSubgroup, 16, 16>(0.0f);
    result_frag[2] = fcoopmatNV<16, gl_ScopeSubgroup, 16, 16>(0.0f);
    result_frag[3] = fcoopmatNV<16, gl_ScopeSubgroup, 16, 16>(0.0f);
    // Load 4 chunks of weights from shared memory into registers.
    result_frag[0] = coopMatMulAddNV(act_frag[0], weights_frag[0], result_frag[0]);
    result_frag[0] = coopMatMulAddNV(act_frag[1], weights_frag[1], result_frag[0]);
    result_frag[1] = coopMatMulAddNV(act_frag[0], weights_frag[2], result_frag[1]);
    result_frag[1] = coopMatMulAddNV(act_frag[1], weights_frag[3], result_frag[1]);
    result_frag[2] = coopMatMulAddNV(act_frag[2], weights_frag[0], result_frag[2]);
    result_frag[2] = coopMatMulAddNV(act_frag[3], weights_frag[1], result_frag[2]);
    result_frag[3] = coopMatMulAddNV(act_frag[2], weights_frag[2], result_frag[3]);
    result_frag[3] = coopMatMulAddNV(act_frag[3], weights_frag[3], result_frag[3]);
    // synchronize the threads in the warp, so that the output matrix is correct
    barrier();
    // Store the output matrix.
    coopMatStoreNV(result_frag[0], glsl_half_shared_buffer, 32*32*wi +   0, 32, false);
    coopMatStoreNV(result_frag[1], glsl_half_shared_buffer, 32*32*wi +  16, 32, false);
    coopMatStoreNV(result_frag[2], glsl_half_shared_buffer, 32*32*wi + 512, 32, false);
    coopMatStoreNV(result_frag[3], glsl_half_shared_buffer, 32*32*wi + 528, 32, false);
}

#endif // _SRENDERER_TINY_MATMUL_GLSLI_HEADER_


#ifndef _SRENDERER_TINY_MATMUL_GLSLI_HEADER_
#define _SRENDERER_TINY_MATMUL_GLSLI_HEADER_

#extension GL_KHR_memory_scope_semantics: require
#extension GL_EXT_control_flow_attributes: require
#extension GL_NV_cooperative_matrix: require
#extension GL_EXT_shader_explicit_arithmetic_types: require
#extension GL_EXT_shader_explicit_arithmetic_types_float16: require
#extension GL_KHR_shader_subgroup_arithmetic: require
#extension GL_EXT_shader_subgroup_extended_types_float16: require
#extension GL_EXT_buffer_reference : require

// #define GLSL_SHARED_MEMORY_SIZE

// fused shared memory for matrix multiplication

// #define GLSL_SHARED_MEMORY_SIZE must be defined. 8192 for training. 5120 for inference

#ifndef GLSL_SHARED_MEMORY_SIZE
#define GLSL_SHARED_MEMORY_SIZE 8192
#endif
//shared float16_t glsl_half_shared_buffer[GLSL_SHARED_MEMORY_SIZE];
shared float16_t glsl_half_shared_buffer[GLSL_SHARED_MEMORY_SIZE];
// set/get the shared buffer
void glsl_set_half_shared_buffer(int i, float16_t value) { glsl_half_shared_buffer[i] = value; }
float16_t glsl_get_half_shared_buffer(int i) { return glsl_half_shared_buffer[i]; }

// 2 = 2 bytes = half float 
layout(std430, buffer_reference, buffer_reference_align=2) buffer WeightPtr
{
    float16_t data[];
};

fcoopmatNV<16, gl_ScopeSubgroup, 16, 16> glsl_relu(fcoopmatNV<16, gl_ScopeSubgroup, 16, 16> m)
{
    for (int i = 0; i < m.length(); ++i)
    {
        m[i] = float16_t(float16_t(m[i] > float16_t(0.0f)) * m[i]);
    }
    return m;
}

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
 * |   32*32  |   32*32  |   32*32   |   32*32   | - WE READ LOCAL ARRAY TO PUT THE PREVIOUS OUTPUT (or the first input data) 
 * |  Input 0 |  Input 1 |  Input 2  |  Input 3  |
 * --- offset = 4096 -----------------------------
 * |   32*32  |   32*32  |   32*32   |   32*32   | - BOTTLENECK. VRAM -> SHARED MEM -> TENSORS CORES per each LAYER
 * | Weight 0 | Weight 1 | Weight 2  |  Weight 3  | = each warp can execute its 
 * --- offset = 8192 -----------------------------
 * The output overrides the input matrix.
 * --- offset = 0 --------------------------------
 * |   32*32  |   32*32  |   32*32   |   32*32   | - WE PUT OUTPUT TO THE LOCAL ARRAY -> REGISTERS. 32 registers at least per each thread
 * | Output 0 | Output 1 |  Output 2 |  Output 3 |
 * --- offset = 4096 -----------------------------
 */ // 32x4 = each thread 32 neurons 32x32x4 = 4096, 1024 = weights, 
void glsl_wmma_128_32_32() {
    // load the data from shared memory to the fragment
    fcoopmatNV<16, gl_ScopeSubgroup, 16, 16> act_frag[4];
    fcoopmatNV<16, gl_ScopeSubgroup, 16, 16> weights_frag[4];
    fcoopmatNV<16, gl_ScopeSubgroup, 16, 16 > result_frag[2];
    const uint li = gl_SubgroupInvocationID; // index in warp ("lane index")
    const uint wi = gl_SubgroupID;          // index in block ("warp index")
    // load the activations from shared memory
    coopMatLoadNV(act_frag[0], glsl_half_shared_buffer, 32*32*wi +   0, 32, false);
    coopMatLoadNV(act_frag[1], glsl_half_shared_buffer, 32*32*wi +  16, 32, false);
    coopMatLoadNV(act_frag[2], glsl_half_shared_buffer, 32*32*wi + 512, 32, false);
    coopMatLoadNV(act_frag[3], glsl_half_shared_buffer, 32*32*wi + 528, 32, false);
    // synchronize the threads in the group, so that the input matrix is all loaded
    barrier();
    // load the weights from shared memory
    coopMatLoadNV(weights_frag[0], glsl_half_shared_buffer, 4096 + 32 * 32 * wi, 32, true);
    coopMatLoadNV(weights_frag[1], glsl_half_shared_buffer, 4096 + 32 * 32 * wi + 16, 32, true);
    coopMatLoadNV(weights_frag[2], glsl_half_shared_buffer, 4096 + 32 * 32 * wi + 512, 32, true);
    coopMatLoadNV(weights_frag[3], glsl_half_shared_buffer, 4096 + 32 * 32 * wi + 528, 32, true);

    // clear the output accumulation matrix
    result_frag[0] = fcoopmatNV<16, gl_ScopeSubgroup, 16, 16>(0.0f);
    result_frag[1] = fcoopmatNV<16, gl_ScopeSubgroup, 16, 16>(0.0f);
    // // Load 2 chunks of weights from shared memory into registers.
    result_frag[0] = coopMatMulAddNV(act_frag[0], weights_frag[0], result_frag[0]);
    result_frag[0] = coopMatMulAddNV(act_frag[1], weights_frag[1], result_frag[0]);
    result_frag[1] = coopMatMulAddNV(act_frag[2], weights_frag[0], result_frag[1]);
    result_frag[1] = coopMatMulAddNV(act_frag[3], weights_frag[1], result_frag[1]);
    // Store the output matrix.
    coopMatStoreNV(result_frag[0], glsl_half_shared_buffer, wi*32*32 +   0, 32, false);
    coopMatStoreNV(result_frag[1], glsl_half_shared_buffer, wi*32*32 + 512, 32, false);
    // load the weights from shared memory
    // clear the output accumulation matrix
    result_frag[0] = fcoopmatNV<16, gl_ScopeSubgroup, 16, 16>(0.0f);
    result_frag[1] = fcoopmatNV<16, gl_ScopeSubgroup, 16, 16>(0.0f);
    // // Load 2 chunks of weights from shared memory into registers.
    result_frag[0] = coopMatMulAddNV(act_frag[0], weights_frag[2], result_frag[0]);
    result_frag[0] = coopMatMulAddNV(act_frag[1], weights_frag[3], result_frag[0]);
    result_frag[1] = coopMatMulAddNV(act_frag[2], weights_frag[2], result_frag[1]);
    result_frag[1] = coopMatMulAddNV(act_frag[3], weights_frag[3], result_frag[1]);

    // synchronize the threads in the warp, so that the output matrix is correct
    coopMatStoreNV(result_frag[0], glsl_half_shared_buffer, wi*32*32 +  16, 32, false);
    coopMatStoreNV(result_frag[1], glsl_half_shared_buffer, wi*32*32 + 528, 32, false);
}

void glsl_wmma_128_32_32_fused(uint64_t weights_pointer, uint32_t num_layers)
{

    // load the data from shared memory to the fragment
    fcoopmatNV<16, gl_ScopeSubgroup, 16, 16> act_frag[4];
    fcoopmatNV<16, gl_ScopeSubgroup, 16, 16> weights_frag[4];
    fcoopmatNV<16, gl_ScopeSubgroup, 16, 16> result_frag[2];

    const uint lane = gl_LocalInvocationID.x;
    const uint warp_id = gl_LocalInvocationID.y;
    const uint li = gl_SubgroupInvocationID; // index in warp ("lane index")
     uint wi = gl_SubgroupID;           // index in block ("warp index")

     wi = warp_id;

     for (uint32_t j = 0; j < num_layers; j++)
     {
         // pointer = virtual  address. 32x32 = # weights. 2 = half. 1024*2. 
        WeightPtr weights = WeightPtr(weights_pointer + j * 32 * 32 * 2);

        // we can load data from the VRAM directly
        coopMatLoadNV(weights_frag[0], weights.data, 0, 32, false);
        coopMatLoadNV(weights_frag[1], weights.data, 0 + 512, 32, false);
        coopMatLoadNV(weights_frag[2], weights.data, 0 + 16, 32, false);
        coopMatLoadNV(weights_frag[3], weights.data, 0 + 528, 32, false);
        barrier();

        coopMatLoadNV(act_frag[0], glsl_half_shared_buffer, 32 * 32 * wi + 0, 32, true);
        coopMatLoadNV(act_frag[1], glsl_half_shared_buffer, 32 * 32 * wi + 512, 32, true);
        coopMatLoadNV(act_frag[2], glsl_half_shared_buffer, 32 * 32 * wi + 16, 32, true);
        coopMatLoadNV(act_frag[3], glsl_half_shared_buffer, 32 * 32 * wi + 528, 32, true);
        

        // clear the output accumulation matrix
        result_frag[0] = fcoopmatNV<16, gl_ScopeSubgroup, 16, 16>(0.0f);
        result_frag[1] = fcoopmatNV<16, gl_ScopeSubgroup, 16, 16>(0.0f);
        // // Load 2 chunks of weights from shared memory into registers.
        result_frag[0] = coopMatMulAddNV(act_frag[0], weights_frag[0], result_frag[0]);
        result_frag[0] = coopMatMulAddNV(act_frag[1], weights_frag[1], result_frag[0]);
        result_frag[1] = coopMatMulAddNV(act_frag[2], weights_frag[0], result_frag[1]);
        result_frag[1] = coopMatMulAddNV(act_frag[3], weights_frag[1], result_frag[1]);

        
        coopMatStoreNV(glsl_relu(result_frag[0]), glsl_half_shared_buffer, wi * 32 * 32 + 0, 32, true);
        coopMatStoreNV(glsl_relu(result_frag[1]), glsl_half_shared_buffer, wi * 32 * 32 + 16, 32, true);
        
        // load the weights from shared memory
        // clear the output accumulation matrix
        result_frag[0] = fcoopmatNV<16, gl_ScopeSubgroup, 16, 16>(0.0f);
        result_frag[1] = fcoopmatNV<16, gl_ScopeSubgroup, 16, 16>(0.0f);
        // // Load 2 chunks of weights from shared memory into registers.
        result_frag[0] = coopMatMulAddNV(act_frag[0], weights_frag[2], result_frag[0]);
        result_frag[0] = coopMatMulAddNV(act_frag[1], weights_frag[3], result_frag[0]);
        result_frag[1] = coopMatMulAddNV(act_frag[2], weights_frag[2], result_frag[1]);
        result_frag[1] = coopMatMulAddNV(act_frag[3], weights_frag[3], result_frag[1]);

        // synchronize the threads in the warp, so that the output matrix is correct
        coopMatStoreNV(glsl_relu(result_frag[0]), glsl_half_shared_buffer, wi * 32 * 32 + 512, 32, true);
        coopMatStoreNV(glsl_relu(result_frag[1]), glsl_half_shared_buffer, wi * 32 * 32 + 528, 32, true);
     }
     barrier();
}

void glsl_wmma_128_32_32_weightsvram(uint64_t weights_pointer)
{
     // load the data from shared memory to the fragment
     fcoopmatNV<16, gl_ScopeSubgroup, 16, 16> act_frag[4];
     fcoopmatNV<16, gl_ScopeSubgroup, 16, 16> weights_frag[4];
     fcoopmatNV<16, gl_ScopeSubgroup, 16, 16> result_frag[2];
     const uint li = gl_SubgroupInvocationID; // index in warp ("lane index")
     const uint wi = gl_SubgroupID;           // index in block ("warp index")
     // load the activations from shared memory
     coopMatLoadNV(act_frag[0], glsl_half_shared_buffer, 32 * 32 * wi + 0, 32, false);
     coopMatLoadNV(act_frag[1], glsl_half_shared_buffer, 32 * 32 * wi + 16, 32, false);
     coopMatLoadNV(act_frag[2], glsl_half_shared_buffer, 32 * 32 * wi + 512, 32, false);
     coopMatLoadNV(act_frag[3], glsl_half_shared_buffer, 32 * 32 * wi + 528, 32, false);
     // synchronize the threads in the group, so that the input matrix is all loaded
     // pointer = virtual  address. 32x32 = # weights. 2 = half. 1024*2.
     WeightPtr weights = WeightPtr(weights_pointer);

     // we can load data from the VRAM directly
     coopMatLoadNV(weights_frag[0], weights.data, 0, 32, false);
     coopMatLoadNV(weights_frag[1], weights.data, 0 + 512, 32, false);
     coopMatLoadNV(weights_frag[2], weights.data, 0 + 16, 32, false);
     coopMatLoadNV(weights_frag[3], weights.data, 0 + 528, 32, false);
     barrier();

     // clear the output accumulation matrix
     result_frag[0] = fcoopmatNV<16, gl_ScopeSubgroup, 16, 16>(0.0f);
     result_frag[1] = fcoopmatNV<16, gl_ScopeSubgroup, 16, 16>(0.0f);
     // // Load 2 chunks of weights from shared memory into registers.
     result_frag[0] = coopMatMulAddNV(act_frag[0], weights_frag[0], result_frag[0]);
     result_frag[0] = coopMatMulAddNV(act_frag[1], weights_frag[1], result_frag[0]);
     result_frag[1] = coopMatMulAddNV(act_frag[2], weights_frag[0], result_frag[1]);
     result_frag[1] = coopMatMulAddNV(act_frag[3], weights_frag[1], result_frag[1]);
     // Store the output matrix.
     coopMatStoreNV(result_frag[0], glsl_half_shared_buffer, wi * 32 * 32 + 0, 32, false);
     coopMatStoreNV(result_frag[1], glsl_half_shared_buffer, wi * 32 * 32 + 512, 32, false);
     // load the weights from shared memory
     // clear the output accumulation matrix
     result_frag[0] = fcoopmatNV<16, gl_ScopeSubgroup, 16, 16>(0.0f);
     result_frag[1] = fcoopmatNV<16, gl_ScopeSubgroup, 16, 16>(0.0f);
     // // Load 2 chunks of weights from shared memory into registers.
     result_frag[0] = coopMatMulAddNV(act_frag[0], weights_frag[2], result_frag[0]);
     result_frag[0] = coopMatMulAddNV(act_frag[1], weights_frag[3], result_frag[0]);
     result_frag[1] = coopMatMulAddNV(act_frag[2], weights_frag[2], result_frag[1]);
     result_frag[1] = coopMatMulAddNV(act_frag[3], weights_frag[3], result_frag[1]);

     // synchronize the threads in the warp, so that the output matrix is correct
     coopMatStoreNV(result_frag[0], glsl_half_shared_buffer, wi * 32 * 32 + 16, 32, false);
     coopMatStoreNV(result_frag[1], glsl_half_shared_buffer, wi * 32 * 32 + 528, 32, false);
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


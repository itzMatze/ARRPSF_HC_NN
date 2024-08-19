#ifndef _SRENDERER_ADDON_TINYNN_MATMUL_HLSLI_HEADER_
#define _SRENDERER_ADDON_TINYNN_MATMUL_HLSLI_HEADER_

void __inline_set_half_shared_buffer(int i, float16_t value) {
    __requirePrelude(R"(#extension GL_GOOGLE_include_directive : enable)");
    __requirePrelude(R"(#include "HalfMatmulInclude.glsl")");
    __intrinsic_asm "glsl_set_half_shared_buffer($0, $1)";
}

float16_t __inline_get_half_shared_buffer(int i) {
    __requirePrelude(R"(#extension GL_GOOGLE_include_directive : enable)");
    __requirePrelude(R"(#include "HalfMatmulInclude.glsl")");
    __intrinsic_asm "glsl_get_half_shared_buffer($0)";
}

void __inline_wmma_128_16_16() {
    __requirePrelude(R"(#extension GL_GOOGLE_include_directive : enable)");
    __requirePrelude(R"(#include "HalfMatmulInclude.glsl")");
    __intrinsic_asm "glsl_wmma_128_16_16()";
}

void __inline_wmma_16_128_16() {
    __requirePrelude(R"(#extension GL_GOOGLE_include_directive : enable)");
    __requirePrelude(R"(#include "HalfMatmulInclude.glsl")");
    __intrinsic_asm "glsl_wmma_16_128_16()";
}

void __inline_wmma_128_32_32() {
    __requirePrelude(R"(#extension GL_GOOGLE_include_directive : enable)");
    __requirePrelude(R"(#include "HalfMatmulInclude.glsl")");
    __intrinsic_asm "glsl_wmma_128_32_32()";
}

void __inline_wmma_128_32_32_fused(uint64_t weights_address, uint32_t num_layers)
{
    __requirePrelude(R"(#extension GL_GOOGLE_include_directive : enable)");
    __requirePrelude(R"(#include "HalfMatmulInclude.glsl")");
    __intrinsic_asm "glsl_wmma_128_32_32_fused($0, $1)";
}

void __inline_wmma_128_32_32_fused_preloadedweights(uint64_t weights_address, uint32_t num_layers)
{
    __requirePrelude(R"(#extension GL_GOOGLE_include_directive : enable)");
    __requirePrelude(R"(#include "HalfMatmulInclude.glsl")");
    __intrinsic_asm "glsl_wmma_128_32_32_fused_preloadedweights($0, $1)";
}



void __inline_wmma_128_32_32_weightsvram(uint64_t weights_address)
{
    __requirePrelude(R"(#extension GL_GOOGLE_include_directive : enable)");
    __requirePrelude(R"(#include "HalfMatmulInclude.glsl")");
    __intrinsic_asm "glsl_wmma_128_32_32_weightsvram($0)";
}


void __inline_wmma_128_32_32_our2()
{
    __requirePrelude(R"(#extension GL_GOOGLE_include_directive : enable)");
    __requirePrelude(R"(#include "HalfMatmulInclude.glsl")");
    __intrinsic_asm "glsl_wmma_128_32_32_our2()";
}


void __inline_wmma_32_128_32() {
    __requirePrelude(R"(#extension GL_GOOGLE_include_directive : enable)");
    __requirePrelude(R"(#include "HalfMatmulInclude.glsl")");
    __intrinsic_asm "glsl_wmma_32_128_32()";
}

#endif // !_SRENDERER_ADDON_TINYNN_MATMUL_HLSLI_HEADER_

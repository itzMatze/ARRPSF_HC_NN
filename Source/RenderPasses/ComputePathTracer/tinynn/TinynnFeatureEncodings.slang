#ifndef _SRENDERER_ADDON_HALF_TINYNN_FEATUREGRID_HLSLI_HEADER_
#define _SRENDERER_ADDON_HALF_TINYNN_FEATUREGRID_HLSLI_HEADER_

import Utils.Debug.PixelDebug;

#include "TinynnHalfLinear.slang"
#if NN_USE_HASH_ENC_INTERPOLATION
#include "HashEncInterpolate.slang"
#endif
#if NN_USE_HASH_ENC
#include "HashEnc.slang"
#endif

#if NN_TRAIN
[Differentiable]
#endif
void shEnc<let KDegree : int, let N : int>(float3 dir, out float16_t[N] vals)
{
    // https://github.com/nvlabs/tiny-cuda-nn
	float xy=dir.x*dir.y, xz=dir.x*dir.z, yz=dir.y*dir.z, x2=dir.x*dir.x, y2=dir.y*dir.y, z2=dir.z*dir.z;
	float x4=x2*x2, y4=y2*y2, z4=z2*z2;
	float x6=x4*x2, y6=y4*y2, z6=z4*z2;

	// SH polynomials generated using scripts/gen_sh.py based on the recurrence relations in appendix A1 of https://www.ppsloan.org/publications/StupidSH36.pdf
	vals[0] = float16_t(0.28209479177387814f);                          // 1/(2*sqrt(pi))
	if (KDegree <= 1) { return; }
	vals[1] = float16_t(-0.48860251190291987f*dir.y);                               // -sqrt(3)*y/(2*sqrt(pi))
	vals[2] = float16_t(0.48860251190291987f*dir.z);                                // sqrt(3)*z/(2*sqrt(pi))
	vals[3] = float16_t(-0.48860251190291987f*dir.x);                               // -sqrt(3)*x/(2*sqrt(pi))
	if (KDegree <= 2) { return; }
	vals[4] = float16_t(1.0925484305920792f*xy);                                // sqrt(15)*xy/(2*sqrt(pi))
	vals[5] = float16_t(-1.0925484305920792f*yz);                               // -sqrt(15)*yz/(2*sqrt(pi))
	vals[6] = float16_t(0.94617469575755997f*z2 - 0.31539156525251999f);                         // sqrt(5)*(3*z2 - 1)/(4*sqrt(pi))
	vals[7] = float16_t(-1.0925484305920792f*xz);                               // -sqrt(15)*xz/(2*sqrt(pi))
	vals[8] = float16_t(0.54627421529603959f*x2 - 0.54627421529603959f*y2);                              // sqrt(15)*(x2 - y2)/(4*sqrt(pi))
	if (KDegree <= 3) { return; }
	vals[9] = float16_t(0.59004358992664352f*dir.y*(-3.0f*x2 + y2));                         // sqrt(70)*y*(-3*x2 + y2)/(8*sqrt(pi))
	vals[10] = float16_t(2.8906114426405538f*xy*dir.z);                             // sqrt(105)*xy*z/(2*sqrt(pi))
	vals[11] = float16_t(0.45704579946446572f*dir.y*(1.0f - 5.0f*z2));                                // sqrt(42)*y*(1 - 5*z2)/(8*sqrt(pi))
	vals[12] = float16_t(0.3731763325901154f*dir.z*(5.0f*z2 - 3.0f));                         // sqrt(7)*z*(5*z2 - 3)/(4*sqrt(pi))
	vals[13] = float16_t(0.45704579946446572f*dir.x*(1.0f - 5.0f*z2));                                // sqrt(42)*x*(1 - 5*z2)/(8*sqrt(pi))
	vals[14] = float16_t(1.4453057213202769f*dir.z*(x2 - y2));                              // sqrt(105)*z*(x2 - y2)/(4*sqrt(pi))
	vals[15] = float16_t(0.59004358992664352f*dir.x*(-x2 + 3.0f*y2));                                // sqrt(70)*x*(-x2 + 3*y2)/(8*sqrt(pi))
	if (KDegree <= 4) { return; }
	vals[16] = float16_t(2.5033429417967046f*xy*(x2 - y2));                             // 3*sqrt(35)*xy*(x2 - y2)/(4*sqrt(pi))
	vals[17] = float16_t(1.7701307697799304f*yz*(-3.0f*x2 + y2));                                // 3*sqrt(70)*yz*(-3*x2 + y2)/(8*sqrt(pi))
	vals[18] = float16_t(0.94617469575756008f*xy*(7.0f*z2 - 1.0f));                               // 3*sqrt(5)*xy*(7*z2 - 1)/(4*sqrt(pi))
	vals[19] = float16_t(0.66904654355728921f*yz*(3.0f - 7.0f*z2));                               // 3*sqrt(10)*yz*(3 - 7*z2)/(8*sqrt(pi))
	vals[20] = float16_t(-3.1735664074561294f*z2 + 3.7024941420321507f*z4 + 0.31735664074561293f);                                // 3*(-30*z2 + 35*z4 + 3)/(16*sqrt(pi))
	vals[21] = float16_t(0.66904654355728921f*xz*(3.0f - 7.0f*z2));                               // 3*sqrt(10)*xz*(3 - 7*z2)/(8*sqrt(pi))
	vals[22] = float16_t(0.47308734787878004f*(x2 - y2)*(7.0f*z2 - 1.0f));                                // 3*sqrt(5)*(x2 - y2)*(7*z2 - 1)/(8*sqrt(pi))
	vals[23] = float16_t(1.7701307697799304f*xz*(-x2 + 3.0f*y2));                                // 3*sqrt(70)*xz*(-x2 + 3*y2)/(8*sqrt(pi))
	vals[24] = float16_t(-3.7550144126950569f*x2*y2 + 0.62583573544917614f*x4 + 0.62583573544917614f*y4);                         // 3*sqrt(35)*(-6*x2*y2 + x4 + y4)/(16*sqrt(pi))
	if (KDegree <= 5) { return; }
	vals[25] = float16_t(0.65638205684017015f*dir.y*(10.0f*x2*y2 - 5.0f*x4 - y4));                            // 3*sqrt(154)*y*(10*x2*y2 - 5*x4 - y4)/(32*sqrt(pi))
	vals[26] = float16_t(8.3026492595241645f*xy*dir.z*(x2 - y2));                           // 3*sqrt(385)*xy*z*(x2 - y2)/(4*sqrt(pi))
	vals[27] = float16_t(-0.48923829943525038f*dir.y*(3.0f*x2 - y2)*(9.0f*z2 - 1.0f));                         // -sqrt(770)*y*(3*x2 - y2)*(9*z2 - 1)/(32*sqrt(pi))
	vals[28] = float16_t(4.7935367849733241f*xy*dir.z*(3.0f*z2 - 1.0f));                              // sqrt(1155)*xy*z*(3*z2 - 1)/(4*sqrt(pi))
	vals[29] = float16_t(0.45294665119569694f*dir.y*(14.0f*z2 - 21.0f*z4 - 1.0f));                             // sqrt(165)*y*(14*z2 - 21*z4 - 1)/(16*sqrt(pi))
	vals[30] = float16_t(0.1169503224534236f*dir.z*(-70.0f*z2 + 63.0f*z4 + 15.0f));                            // sqrt(11)*z*(-70*z2 + 63*z4 + 15)/(16*sqrt(pi))
	vals[31] = float16_t(0.45294665119569694f*dir.x*(14.0f*z2 - 21.0f*z4 - 1.0f));                             // sqrt(165)*x*(14*z2 - 21*z4 - 1)/(16*sqrt(pi))
	vals[32] = float16_t(2.3967683924866621f*dir.z*(x2 - y2)*(3.0f*z2 - 1.0f));                               // sqrt(1155)*z*(x2 - y2)*(3*z2 - 1)/(8*sqrt(pi))
	vals[33] = float16_t(-0.48923829943525038f*dir.x*(x2 - 3.0f*y2)*(9.0f*z2 - 1.0f));                         // -sqrt(770)*x*(x2 - 3*y2)*(9*z2 - 1)/(32*sqrt(pi))
	vals[34] = float16_t(2.0756623148810411f*dir.z*(-6.0f*x2*y2 + x4 + y4));                         // 3*sqrt(385)*z*(-6*x2*y2 + x4 + y4)/(16*sqrt(pi))
	vals[35] = float16_t(0.65638205684017015f*dir.x*(10.0f*x2*y2 - x4 - 5.0f*y4));                            // 3*sqrt(154)*x*(10*x2*y2 - x4 - 5*y4)/(32*sqrt(pi))
	if (KDegree <= 6) { return; }
	vals[36] = float16_t(1.3663682103838286f*xy*(-10.0f*x2*y2 + 3.0f*x4 + 3.0f*y4));                               // sqrt(6006)*xy*(-10*x2*y2 + 3*x4 + 3*y4)/(32*sqrt(pi))
	vals[37] = float16_t(2.3666191622317521f*yz*(10.0f*x2*y2 - 5.0f*x4 - y4));                            // 3*sqrt(2002)*yz*(10*x2*y2 - 5*x4 - y4)/(32*sqrt(pi))
	vals[38] = float16_t(2.0182596029148963f*xy*(x2 - y2)*(11.0f*z2 - 1.0f));                             // 3*sqrt(91)*xy*(x2 - y2)*(11*z2 - 1)/(8*sqrt(pi))
	vals[39] = float16_t(-0.92120525951492349f*yz*(3.0f*x2 - y2)*(11.0f*z2 - 3.0f));                               // -sqrt(2730)*yz*(3*x2 - y2)*(11*z2 - 3)/(32*sqrt(pi))
	vals[40] = float16_t(0.92120525951492349f*xy*(-18.0f*z2 + 33.0f*z4 + 1.0f));                           // sqrt(2730)*xy*(-18*z2 + 33*z4 + 1)/(32*sqrt(pi))
	vals[41] = float16_t(0.58262136251873131f*yz*(30.0f*z2 - 33.0f*z4 - 5.0f));                            // sqrt(273)*yz*(30*z2 - 33*z4 - 5)/(16*sqrt(pi))
	vals[42] = float16_t(6.6747662381009842f*z2 - 20.024298714302954f*z4 + 14.684485723822165f*z6 - 0.31784601133814211f);                         // sqrt(13)*(105*z2 - 315*z4 + 231*z6 - 5)/(32*sqrt(pi))
	vals[43] = float16_t(0.58262136251873131f*xz*(30.0f*z2 - 33.0f*z4 - 5.0f));                            // sqrt(273)*xz*(30*z2 - 33*z4 - 5)/(16*sqrt(pi))
	vals[44] = float16_t(0.46060262975746175f*(x2 - y2)*(11.0f*z2*(3.0f*z2 - 1.0f) - 7.0f*z2 + 1.0f));                               // sqrt(2730)*(x2 - y2)*(11*z2*(3*z2 - 1) - 7*z2 + 1)/(64*sqrt(pi))
	vals[45] = float16_t(-0.92120525951492349f*xz*(x2 - 3.0f*y2)*(11.0f*z2 - 3.0f));                               // -sqrt(2730)*xz*(x2 - 3*y2)*(11*z2 - 3)/(32*sqrt(pi))
	vals[46] = float16_t(0.50456490072872406f*(11.0f*z2 - 1.0f)*(-6.0f*x2*y2 + x4 + y4));                          // 3*sqrt(91)*(11*z2 - 1)*(-6*x2*y2 + x4 + y4)/(32*sqrt(pi))
	vals[47] = float16_t(2.3666191622317521f*xz*(10.0f*x2*y2 - x4 - 5.0f*y4));                            // 3*sqrt(2002)*xz*(10*x2*y2 - x4 - 5*y4)/(32*sqrt(pi))
	vals[48] = float16_t(10.247761577878714f*x2*y4 - 10.247761577878714f*x4*y2 + 0.6831841051919143f*x6 - 0.6831841051919143f*y6);                         // sqrt(6006)*(15*x2*y4 - 15*x4*y2 + x6 - y6)/(64*sqrt(pi))
	if (KDegree <= 7) { return; }
	vals[49] = float16_t(0.70716273252459627f*dir.y*(-21.0f*x2*y4 + 35.0f*x4*y2 - 7.0f*x6 + y6));                              // 3*sqrt(715)*y*(-21*x2*y4 + 35*x4*y2 - 7*x6 + y6)/(64*sqrt(pi))
	vals[50] = float16_t(5.2919213236038001f*xy*dir.z*(-10.0f*x2*y2 + 3.0f*x4 + 3.0f*y4));                             // 3*sqrt(10010)*xy*z*(-10*x2*y2 + 3*x4 + 3*y4)/(32*sqrt(pi))
	vals[51] = float16_t(-0.51891557872026028f*dir.y*(13.0f*z2 - 1.0f)*(-10.0f*x2*y2 + 5.0f*x4 + y4));                          // -3*sqrt(385)*y*(13*z2 - 1)*(-10*x2*y2 + 5*x4 + y4)/(64*sqrt(pi))
	vals[52] = float16_t(4.1513246297620823f*xy*dir.z*(x2 - y2)*(13.0f*z2 - 3.0f));                           // 3*sqrt(385)*xy*z*(x2 - y2)*(13*z2 - 3)/(8*sqrt(pi))
	vals[53] = float16_t(-0.15645893386229404f*dir.y*(3.0f*x2 - y2)*(13.0f*z2*(11.0f*z2 - 3.0f) - 27.0f*z2 + 3.0f));                              // -3*sqrt(35)*y*(3*x2 - y2)*(13*z2*(11*z2 - 3) - 27*z2 + 3)/(64*sqrt(pi))
	vals[54] = float16_t(0.44253269244498261f*xy*dir.z*(-110.0f*z2 + 143.0f*z4 + 15.0f));                              // 3*sqrt(70)*xy*z*(-110*z2 + 143*z4 + 15)/(32*sqrt(pi))
	vals[55] = float16_t(0.090331607582517306f*dir.y*(-135.0f*z2 + 495.0f*z4 - 429.0f*z6 + 5.0f));                              // sqrt(105)*y*(-135*z2 + 495*z4 - 429*z6 + 5)/(64*sqrt(pi))
	vals[56] = float16_t(0.068284276912004949f*dir.z*(315.0f*z2 - 693.0f*z4 + 429.0f*z6 - 35.0f));                              // sqrt(15)*z*(315*z2 - 693*z4 + 429*z6 - 35)/(32*sqrt(pi))
	vals[57] = float16_t(0.090331607582517306f*dir.x*(-135.0f*z2 + 495.0f*z4 - 429.0f*z6 + 5.0f));                              // sqrt(105)*x*(-135*z2 + 495*z4 - 429*z6 + 5)/(64*sqrt(pi))
	vals[58] = float16_t(0.07375544874083044f*dir.z*(x2 - y2)*(143.0f*z2*(3.0f*z2 - 1.0f) - 187.0f*z2 + 45.0f));                         // sqrt(70)*z*(x2 - y2)*(143*z2*(3*z2 - 1) - 187*z2 + 45)/(64*sqrt(pi))
	vals[59] = float16_t(-0.15645893386229404f*dir.x*(x2 - 3.0f*y2)*(13.0f*z2*(11.0f*z2 - 3.0f) - 27.0f*z2 + 3.0f));                              // -3*sqrt(35)*x*(x2 - 3*y2)*(13*z2*(11*z2 - 3) - 27*z2 + 3)/(64*sqrt(pi))
	vals[60] = float16_t(1.0378311574405206f*dir.z*(13.0f*z2 - 3.0f)*(-6.0f*x2*y2 + x4 + y4));                         // 3*sqrt(385)*z*(13*z2 - 3)*(-6*x2*y2 + x4 + y4)/(32*sqrt(pi))
	vals[61] = float16_t(-0.51891557872026028f*dir.x*(13.0f*z2 - 1.0f)*(-10.0f*x2*y2 + x4 + 5.0f*y4));                          // -3*sqrt(385)*x*(13*z2 - 1)*(-10*x2*y2 + x4 + 5*y4)/(64*sqrt(pi))
	vals[62] = float16_t(2.6459606618019f*dir.z*(15.0f*x2*y4 - 15.0f*x4*y2 + x6 - y6));                               // 3*sqrt(10010)*z*(15*x2*y4 - 15*x4*y2 + x6 - y6)/(64*sqrt(pi))
	vals[63] = float16_t(0.70716273252459627f*dir.x*(-35.0f*x2*y4 + 21.0f*x4*y2 - x6 + 7.0f*y6));                              // 3*sqrt(715)*x*(-35*x2*y4 + 21*x4*y2 - x6 + 7*y6)/(64*sqrt(pi))
}

#if NN_TRAIN
[Differentiable]
#endif
HalfFeature<32> computeFeature(
    no_diff float3 pos,
    no_diff float3 dir,
    no_diff float3 normal,
    no_diff FeatureHashGrid featureHashGrid,
) {
    HalfFeature<32> feature;
    uint offset = 0;
    [ForceUnroll]
    for (uint i = 0; i < 32; i++) feature.vals[i] = 1.0h;

#if NN_USE_HASH_ENC || NN_USE_HASH_ENC_INTERPOLATION
    feature.vals[offset++] = float16_t(pos.x);
    feature.vals[offset++] = float16_t(pos.y);
    feature.vals[offset++] = float16_t(pos.z);
    [ForceUnroll]
    for (uint i = 0; i < 1; i++)
    {
        feature.vals[offset++] = sin(float16_t(pos.x) * float16_t(3.1415926f * pow(2.0, (i))));
        feature.vals[offset++] = sin(float16_t(pos.y) * float16_t(3.1415926f * pow(2.0, (i))));
        feature.vals[offset++] = sin(float16_t(pos.z) * float16_t(3.1415926f * pow(2.0, (i))));
    }
    // shDegree^2 values
    float16_t shVals[9];
    shEnc<3, 9>(dir, shVals);
    [ForceUnroll]
    for (uint i = 0; i < 9; i++) feature.vals[offset++] = shVals[i];
#endif

#if NN_USE_HASH_ENC_INTERPOLATION
    [ForceUnroll]
    for (uint i = 0; i < 8; i++)
    {
        float value0 = 0.0;
        float value1 = 0.0;
#if USE_MULTI_LEVEL_DIR && USE_NIRC
        uint[8] indices0;
        float[8] weights0;
        uint[8] indices1;
        float[8] weights1;
#if NN_TRAIN
        no_diff featureHashGrid.InsertEntry(pos, dir, normal, i, true, indices0, weights0);
        no_diff featureHashGrid.InsertEntry(pos, dir, normal, i, false, indices1, weights1);
#else
        featureHashGrid.FindEntry(pos, dir, normal, i, true, indices0, weights0);
        featureHashGrid.FindEntry(pos, dir, normal, i, false, indices1, weights1);
#endif
        for (uint j = 0; j < 8; j++)
        {
            value0 += featureHashGrid.dataView.load_prim(indices0[j]) * weights0[j];
            value1 += featureHashGrid.dataView.load_prim(indices1[j]) * weights1[j];
        }
#else
        uint[8] indices;
        float[8] weights;
#if NN_TRAIN
        no_diff featureHashGrid.InsertEntry(pos, dir, normal, i, false, indices, weights);
#else
        featureHashGrid.FindEntry(pos, dir, normal, i, false, indices, weights);
#endif
        for (uint j = 0; j < 8; j++)
        {
            value0 += float16_t(featureHashGrid.dataView.load_prim(indices[j] * 2) * weights[j]);
            value1 += float16_t(featureHashGrid.dataView.load_prim(indices[j] * 2 + 1) * weights[j]);
        }
#endif
        feature.vals[offset++] = float16_t(value0);
        feature.vals[offset++] = float16_t(value1);
    }

#elif NN_USE_HASH_ENC
    [ForceUnroll]
    for (uint i = 0; i < 8; i++)
    {
#if USE_MULTI_LEVEL_DIR && USE_NIRC
#if NN_TRAIN
        uint idx0 = no_diff featureHashGrid.InsertEntry(pos, dir, normal, i, true);
        uint idx1 = no_diff featureHashGrid.InsertEntry(pos, dir, normal, i, false);
#else
        uint idx0 = featureHashGrid.FindEntry(pos, dir, normal, i, true);
        uint idx1 = featureHashGrid.FindEntry(pos, dir, normal, i, false);
#endif
        feature.vals[offset++] = float16_t(featureHashGrid.dataView.load_prim(idx0));
        feature.vals[offset++] = float16_t(featureHashGrid.dataView.load_prim(idx1));
#else
#if NN_TRAIN
        uint idx = no_diff featureHashGrid.InsertEntry(pos, dir, normal, i, false);
#else
        uint idx = featureHashGrid.FindEntry(pos, dir, normal, i, false);
#endif
        feature.vals[offset++] = float16_t(featureHashGrid.dataView.load_prim(idx));
        feature.vals[offset++] = float16_t(featureHashGrid.dataView.load_prim(idx + 1));
#endif
    }

#elif NN_USE_FREQ_ENC
    feature.vals[offset++] = float16_t(pos.x);
    feature.vals[offset++] = float16_t(pos.y);
    feature.vals[offset++] = float16_t(pos.z);
#if 0
    [ForceUnroll]
    for (uint i = 0; i < 8; i++)
    {
        feature.vals[offset++] = sin(float16_t(pos.x) * float16_t(3.1415926f * pow(2.0, (i))));
        feature.vals[offset++] = sin(float16_t(pos.y) * float16_t(3.1415926f * pow(2.0, (i))));
        feature.vals[offset++] = sin(float16_t(pos.z) * float16_t(3.1415926f * pow(2.0, (i))));
    }
#endif
#if 1
    [ForceUnroll]
    for (uint i = 0; i < 4; i++)
    {
        feature.vals[offset++] = sin(float16_t(pos.x) * float16_t(3.1415926f * pow(2.0, (i * 2.0))));
        feature.vals[offset++] = sin(float16_t(pos.y) * float16_t(3.1415926f * pow(2.0, (i * 2.0))));
        feature.vals[offset++] = sin(float16_t(pos.z) * float16_t(3.1415926f * pow(2.0, (i * 2.0))));
    }
#endif
#if 1
    // shDegree^2 values
    float16_t shVals[16];
    shEnc<4, 16>(dir, shVals);
    [ForceUnroll]
    for (uint i = 0; i < 16; i++) feature.vals[offset++] = shVals[i];
#else
    feature.vals[offset++] = float16_t(dir.x);
    feature.vals[offset++] = float16_t(dir.y);
    feature.vals[offset++] = float16_t(dir.z);
    [ForceUnroll]
    for (uint i = 0; i < 4; i++)
    {
        feature.vals[offset++] = sin(float16_t(dir.x) * float16_t(3.1415926f * pow(2.0, (i * 2.0))));
        feature.vals[offset++] = sin(float16_t(dir.y) * float16_t(3.1415926f * pow(2.0, (i * 2.0))));
        feature.vals[offset++] = sin(float16_t(dir.z) * float16_t(3.1415926f * pow(2.0, (i * 2.0))));
    }
#endif
#endif
    return feature;
}
#endif // !_SRENDERER_ADDON_HALF_TINYNN_FEATUREGRID_HLSLI_HEADER_


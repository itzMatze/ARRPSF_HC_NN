#include "Optimizer.hlsli"

import Utils.Debug.PixelDebug;

RWStructuredBuffer<float16_t> PrimalBuffer;
RWStructuredBuffer<float16_t> FilteredPrimalBuffer;
RWStructuredBuffer<float> GradientBuffer;
RWStructuredBuffer<float> GradientCountBuffer;
RWStructuredBuffer<float> GradientAuxBuffer;

static const uint kParamCount = NN_PARAM_COUNT;
static const int kGradOffset = NN_GRAD_OFFSET;
static const uint kOptimizerType = NN_OPTIMIZER_TYPE;
static const float kLearningRate = NN_LEARNING_RATE; // |    SGD    |   Adam   |
static const float kParam0 = NN_PARAM_0;             // |  momentum |  beta_1  |
static const float kParam1 = NN_PARAM_1;             // | dampening |  beta_2  |
static const float kFilterAlpha = NN_FILTER_ALPHA;

enum OptimizerType : uint32_t {
    SGD = 0,
    ADAM = 1,
};

cbuffer CB {
    int t;  // iteration index
};

[shader("compute")]
[numthreads(256, 1, 1)]
void main(int3 dtid: SV_DispatchThreadID) {
    const int tid = dtid.x;
    if (tid >= kParamCount) return;
    printSetPixel(uint2(10000, dtid.x));

    const float theta = float(PrimalBuffer[tid]);
    float count = GradientCountBuffer[kGradOffset + tid];
    float df_dtheta = count > 0.01 ? GradientBuffer[kGradOffset + tid] / float(count) : 0.0;
    if (isnan(df_dtheta) || isinf(df_dtheta)) df_dtheta = 0.0;
    float theta_new = theta;

    if (kOptimizerType == uint32_t(OptimizerType::SGD)) {
        float b_t = GradientAuxBuffer[kGradOffset + tid];
        if (t == 0) b_t = 0.f;
        theta_new = sgd_optimizer(theta, df_dtheta, b_t, t, kLearningRate, kParam0, kParam1);
        GradientAuxBuffer[kGradOffset + tid] = b_t;
    } else if (kOptimizerType == uint32_t(OptimizerType::ADAM)) {
        float m_t = GradientAuxBuffer[2 * (kGradOffset + tid) + 0];
        print("ADAM", m_t);
        float v_t = GradientAuxBuffer[2 * (kGradOffset + tid) + 1];
        if (t == 0) { m_t = 0; v_t = 0; }
        theta_new = adam_optimizer(theta, df_dtheta, m_t, v_t, t, kLearningRate, kParam0, kParam1);
        GradientAuxBuffer[2 * (kGradOffset + tid) + 0] = m_t;
        GradientAuxBuffer[2 * (kGradOffset + tid) + 1] = v_t;
    }
    theta_new = clamp(theta_new, -65000.0, 65000.0);
    PrimalBuffer[tid] = (float16_t)theta_new;
    FilteredPrimalBuffer[tid] = (float16_t)((1 - kFilterAlpha) * theta_new + kFilterAlpha * float(FilteredPrimalBuffer[tid]));
}


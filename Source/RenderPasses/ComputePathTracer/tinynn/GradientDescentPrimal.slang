#include "Optimizer.hlsli"

import Utils.Debug.PixelDebug;

RWStructuredBuffer<float> PrimalBuffer;
RWStructuredBuffer<float> FilteredPrimalBuffer;
RWStructuredBuffer<float> GradientBuffer;
RWStructuredBuffer<float> GradientAuxBuffer;

static const uint kParamCount = NN_PARAM_COUNT;
static const int kGradOffset = NN_GRAD_OFFSET;
static const uint kOptimizerType = NN_OPTIMIZER_TYPE;
static const float kLearningRate = NN_LEARNING_RATE; // |    SGD    |   Adam   |
static const float kParam0 = NN_PARAM_0;             // |  momentum |  beta_1  |
static const float kParam1 = NN_PARAM_1;             // | dampening |  beta_2  |
static const float kParam2 = NN_PARAM_2;             // |           |  epsilon  |
static const float kFilterAlpha = 0.99;

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

    const float theta = PrimalBuffer[tid];
    float df_dtheta = GradientBuffer[kGradOffset + tid];

    if (kOptimizerType == uint32_t(OptimizerType::SGD)) {
        float b_t = GradientAuxBuffer[kGradOffset + tid];
        if (t == 0) b_t = 0.f;
        const float theta_new = sgd_optimizer(theta, df_dtheta, b_t, t, kLearningRate, kParam0, kParam1);
        GradientAuxBuffer[kGradOffset + tid] = b_t;
        PrimalBuffer[tid] = theta_new;
    } else if (kOptimizerType == uint32_t(OptimizerType::ADAM)) {
        float m_t = GradientAuxBuffer[2 * (kGradOffset + tid) + 0];
        float v_t = GradientAuxBuffer[2 * (kGradOffset + tid) + 1];
        if (t == 0) { m_t = 0; v_t = 0; }
        const float theta_new = adam_optimizer(theta, df_dtheta, m_t, v_t, t, kLearningRate, kParam0, kParam1, kParam2);
        GradientAuxBuffer[2 * (kGradOffset + tid) + 0] = m_t;
        GradientAuxBuffer[2 * (kGradOffset + tid) + 1] = v_t;
        PrimalBuffer[tid] = theta_new;
        FilteredPrimalBuffer[tid] = (1 - kFilterAlpha) * theta_new + kFilterAlpha * FilteredPrimalBuffer[tid];
    }
}


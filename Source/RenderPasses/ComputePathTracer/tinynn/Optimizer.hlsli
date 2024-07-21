#ifndef _SRENDERER_ADDON_DIFFERENTIABLE_OPTIMIZER_HEADER_
#define _SRENDERER_ADDON_DIFFERENTIABLE_OPTIMIZER_HEADER_

float sgd_optimizer(
    float theta,
    float df_dtheta,
    inout float b_t,
    in int t,
    in float lr,
    in float param_0,
    in float param_1
) {
    const int t = t;
    const float momentum = param_0;
    const float dampening = param_1;
    float g_t = df_dtheta;
    b_t = (momentum != 0 && t > 1)
              ? momentum * b_t + (1 - dampening) * g_t
              : g_t;
    g_t = b_t;
    if (isnan(g_t) || isinf(g_t)) {
        g_t = 0.0f;
        b_t = 0.0f;
    }
    return theta - lr * g_t;
}

float adam_optimizer(
    float device_params,
    float device_grads,
    inout float device_exp_avgs,
    inout float device_exp_avg_sqs,
    in int t,
    in float lr,
    in float param_0,
    in float param_1,
) {
    const int t = t + 1;
    const float beta_1 = param_0;
    const float beta_2 = param_1;
    const float epsilon = 1e-15;

    device_exp_avgs = lerp(device_exp_avgs, device_grads, 1 - beta_1);
    device_exp_avg_sqs *= beta_2;
    device_exp_avg_sqs += (1 - beta_2) * device_grads * device_grads;

    const float bias_correction1 = 1 - pow(beta_1, t);
    const float bias_correction2 = 1 - pow(beta_2, t);

    const float step_size = (lr / bias_correction1) * -1.0f;
    const float bias_correction2_sqrt = sqrt(bias_correction2);
    float exp_avg_sq_sqrt = sqrt(device_exp_avg_sqs);
    exp_avg_sq_sqrt = exp_avg_sq_sqrt / bias_correction2_sqrt;
    exp_avg_sq_sqrt = exp_avg_sq_sqrt + epsilon;
    float theta_new = device_params + (device_exp_avgs / exp_avg_sq_sqrt) * step_size;

    return theta_new;
}

#endif // _SRENDERER_ADDON_DIFFERENTIABLE_OPTIMIZER_HEADER_


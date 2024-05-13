add_plugin(ComputePathTracer)

target_sources(ComputePathTracer PRIVATE
    ComputePathTracer.cpp
    ComputePathTracer.h
    ComputePathTracer.slang
    ComputePathTracerTrain.slang
    LightSampling.slang
    HashCacheResolve.slang
    HashCacheHashGridCommon.slang
    HashCacheCommon.slang
    tinynn/GradientClear.slang
    tinynn/GradientDescentPrimal.slang
    tinynn/HalfMatmulInclude.hlsli
    tinynn/Optimizer.hlsli
    tinynn/TinynnActivations.hlsli
    tinynn/TinynnFeaturegrid.hlsli
    tinynn/TinynnHalfLinear.hlsli
    tinynn/TinynnHalfMLP.hlsli
    tinynn/TinynnTensorview.hlsli
)

target_copy_shaders(ComputePathTracer RenderPasses/ComputePathTracer)

target_source_group(ComputePathTracer "RenderPasses")

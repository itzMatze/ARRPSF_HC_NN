add_plugin(ComputePathTracer)

target_sources(ComputePathTracer PRIVATE
    ComputePathTracer.cpp
    ComputePathTracer.h
    ComputePathTracer.slang
    ComputePathTracerTrain.slang
    LightSampling.slang
    RadianceHashCacheResolve.slang
    RadianceHashCacheHashGridCommon.slang
    RadianceHashCacheCommon.slang
    tinynn/GradientClear.slang
    tinynn/GradientDescentPrimal.slang
    tinynn/HalfMatmulInclude.hlsli
    tinynn/Optimizer.hlsli
    tinynn/TinynnActivations.hlsli
    tinynn/TinynnFeatureEncodings.hlsli
    tinynn/TinynnHalfLinear.slang
    tinynn/TinynnHalfMLP.hlsli
    tinynn/TinynnTensorview.slang
    IRDebugVis.slang
)

target_copy_shaders(ComputePathTracer RenderPasses/ComputePathTracer)

target_source_group(ComputePathTracer "RenderPasses")

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
    tinynn/HalfMatmulInclude.glsl
    tinynn/Optimizer.slang
    tinynn/TinynnActivations.slang
    tinynn/TinynnFeatureEncodings.slang
    tinynn/TinynnHalfLinear.slang
    tinynn/TinynnHalfMLP.slang
    tinynn/TinynnTensorview.slang
    IRDebugVis.slang
)

target_copy_shaders(ComputePathTracer RenderPasses/ComputePathTracer)

target_source_group(ComputePathTracer "RenderPasses")

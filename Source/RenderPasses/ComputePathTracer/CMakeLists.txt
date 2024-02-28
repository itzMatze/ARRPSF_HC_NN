add_plugin(ComputePathTracer)

target_sources(ComputePathTracer PRIVATE
    ComputePathTracer.cpp
    ComputePathTracer.h
    ComputePathTracer.slang
)

target_copy_shaders(ComputePathTracer RenderPasses/ComputePathTracer)

target_source_group(ComputePathTracer "RenderPasses")

from falcor import *

def render_graph_ComputePathTracer(options = {}):
    passes = {}
    passes['ImageLoader'] = createPass("ImageLoader", {'filename': '/mnt/X/KIT/Falcor/images/reference_converged.png'})
    passes['VBufferRT'] = createPass("VBufferRT", {'samplePattern': 'Stratified', 'sampleCount': 16})
    passes['ComputePathTracer'] = createPass("ComputePathTracer", options)
    passes['AccumulatePass'] = createPass("AccumulatePass", {'enabled': False, 'precisionMode': 'Single'})
    passes['ToneMapper'] = createPass("ToneMapper", {'autoExposure': False, 'exposureCompensation': 0.0})
    passes['ToneMapperIR'] = createPass("ToneMapper", {'autoExposure': False, 'exposureCompensation': 0.0, 'outputSize': 'Fixed', 'fixedOutputSize': (1000, 1000)})
    g = RenderGraph("ComputePathTracer")
    for key, value in passes.items():
        g.addPass(value, key)

    g.addEdge("ImageLoader.dst", "ComputePathTracer.refImage")
    g.addEdge("VBufferRT.vbuffer", "ComputePathTracer.vbuffer")
    g.addEdge("VBufferRT.viewW", "ComputePathTracer.viewW")
    g.addEdge("ComputePathTracer.color", "AccumulatePass.input")
    g.addEdge("AccumulatePass.output", "ToneMapper.src")
    g.addEdge("ComputePathTracer.ir_debug", "ToneMapperIR.src")
    g.markOutput("ToneMapper.dst")
    return g, passes

ComputePathTracer, _ = render_graph_ComputePathTracer()
try: m.addGraph(ComputePathTracer)
except NameError: None

from falcor import *

def render_graph_ComputePathTracer(options = {}):
    passes = {}
    passes['ImageLoader'] = createPass("ImageLoader")
    passes['VBufferRT'] = createPass("VBufferRT", {'samplePattern': 'Stratified', 'sampleCount': 16})
    passes['ComputePathTracer'] = createPass("ComputePathTracer", options)
    passes['AccumulatePass'] = createPass("AccumulatePass", {'enabled': False, 'precisionMode': 'Single'})
    passes['ToneMapper'] = createPass("ToneMapper", {'autoExposure': False, 'exposureCompensation': 0.0})
    g = RenderGraph("ComputePathTracer")
    for key, value in passes.items():
        g.addPass(value, key)

    g.addEdge("ImageLoader.dst", "ComputePathTracer.refImage")
    g.addEdge("VBufferRT.vbuffer", "ComputePathTracer.vbuffer")
    g.addEdge("VBufferRT.viewW", "ComputePathTracer.viewW")
    g.addEdge("ComputePathTracer.color", "AccumulatePass.input")
    g.addEdge("AccumulatePass.output", "ToneMapper.src")
    g.markOutput("ToneMapper.dst")
    return g, passes

ComputePathTracer, _ = render_graph_ComputePathTracer()
try: m.addGraph(ComputePathTracer)
except NameError: None

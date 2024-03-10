from falcor import *

def render_graph_ComputePathTracer():
    g = RenderGraph("ComputePathTracer")
    VBufferRT = createPass("VBufferRT", {'samplePattern': 'Stratified', 'sampleCount': 16})
    g.addPass(VBufferRT, "VBufferRT")
    ComputePathTracer = createPass("ComputePathTracer", {'maxBounces': 3})
    g.addPass(ComputePathTracer, "ComputePathTracer")
    AccumulatePass = createPass("AccumulatePass", {'enabled': True, 'precisionMode': 'Single'})
    g.addPass(AccumulatePass, "AccumulatePass")
    ToneMapper = createPass("ToneMapper", {'autoExposure': False, 'exposureCompensation': 0.0})
    g.addPass(ToneMapper, "ToneMapper")

    g.addEdge("VBufferRT.vbuffer", "ComputePathTracer.vbuffer")
    g.addEdge("VBufferRT.viewW", "ComputePathTracer.viewW")
    g.addEdge("ComputePathTracer.color", "AccumulatePass.input")
    g.addEdge("AccumulatePass.output", "ToneMapper.src")
    g.markOutput("ToneMapper.dst")
    return g

ComputePathTracer = render_graph_ComputePathTracer()
try: m.addGraph(ComputePathTracer)
except NameError: None

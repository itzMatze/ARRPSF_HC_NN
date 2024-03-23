from falcor import *
from scripts.ComputePathTracer import *
import os, shutil

settings_to_test = [
    {'useNEE': True},
    {'useNEE': False}
]

test_scenes = [
    'convergence_test',
    'cornell_box'
]

def print_statistics_to_file(capture, dir, filename_base):
    meanFrameTime = capture['events']['/onFrameRender/gpu_time']['stats']['mean']
    with open(f'{dir}{filename_base}.log', 'w') as file:
        file.write(f'{meanFrameTime}')

def eval_settings(options, dir, filename_base):
    # load render graph
    compute_path_tracer, passes = render_graph_ComputePathTracer(options)
    try: m.addGraph(compute_path_tracer)
    except NameError: None
    # setup frame capture
    m.frameCapture.outputDir = os.path.abspath(dir)
    m.frameCapture.baseFilename = f'{filename_base}'
    # warmup with frame captures
    FRAME_LIST = [1, 2, 3, 4, 256]
    m.frameCapture.addFrames(compute_path_tracer, FRAME_LIST)
    for _ in range(max(FRAME_LIST)):
        m.renderFrame()
    # reset for profiling
    m.clock.stop()
    m.clock.play()
    m.frameCapture.reset()
    passes['AccumulatePass'].reset()
    # start profiling
    m.profiler.enabled = True
    m.profiler.start_capture()
    # render
    for _ in range(512):
        m.renderFrame()
    # end capture and store results
    capture = m.profiler.end_capture()
    m.profiler.enabled = False
    print_statistics_to_file(capture, dir, filename_base)
    m.removeGraph(compute_path_tracer)

def main():
    for scene in test_scenes:
        m.loadScene(f'media/test_scenes/{scene}.pyscene')
        # reset evaluation dir for current scene
        dir = f'evaluation/ComputePathTracer/{scene}/'
        shutil.rmtree(dir, True)
        os.makedirs(os.path.dirname(dir), exist_ok=True)
        for i, options in enumerate(settings_to_test):
            filename_base = f'{i}'
            eval_settings(options, dir, filename_base)
        m.unloadScene()
    exit(0)

main()


import argparse
import time
import torch
import json
import os
from gaussian_renderer import render
from scene import Scene, GaussianModel

def measure_fps(model_path, dataset_path, n_runs=100):
    scene = Scene(dataset_path, shuffle=False)
    model = GaussianModel.load(model_path)

    cameras = scene.getTestCameras()

    torch.cuda.reset_peak_memory_stats()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    times = []

    with torch.no_grad():
        for cam in cameras:
            render(model, cam)

        start.record()
        for i in range(n_runs):
            cam = cameras[i % len(cameras)]
            render(model, cam)
        end.record()

        torch.cuda.synchronize()

    elapsed = start.elapsed_time(end) / 1000.0
    fps = n_runs / elapsed

    max_mem = torch.cuda.max_memory_allocated() / 1024**2

    return fps, max_mem


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--runs", type=int, default=100)

    args = parser.parse_args()

    fps, mem = measure_fps(args.model, args.data, args.runs)

    result = {
        "fps": fps,
        "max_memory_mb": mem
    }

    out = os.path.join(args.model, "fps_result.json")

    with open(out, "w") as f:
        json.dump(result, f, indent=2)

    print("FPS:", fps)
    print("Peak Mem (MB):", mem)

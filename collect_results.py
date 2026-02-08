import os
import json
import pandas as pd

rows = []

for d in os.listdir("output"):
    path = os.path.join("output", d)
    if not os.path.isdir(path):
        continue

    result_file = os.path.join(path, "test_results.json")
    fps_file = os.path.join(path, "fps_result.json")

    if not os.path.exists(result_file):
        continue

    with open(result_file) as f:
        r = json.load(f)

    fps = None
    mem = None

    if os.path.exists(fps_file):
        with open(fps_file) as f:
            fr = json.load(f)
            fps = fr.get("fps")
            mem = fr.get("max_memory_mb")

    parts = d.split("_")
    method = parts[0]
    scene = parts[1]
    iters = parts[2]

    rows.append({
        "exp_name": d,
        "method": method,
        "scene": scene,
        "iters": iters,
        "PSNR": r.get("psnr"),
        "SSIM": r.get("ssim"),
        "LPIPS": r.get("lpips"),
        "FPS": fps,
        "PeakMem(MB)": mem
    })

df = pd.DataFrame(rows)

df.to_csv("all_results.csv", index=False)

print("Saved to all_results_v1.csv")
print(df)

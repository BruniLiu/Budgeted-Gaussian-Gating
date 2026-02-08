import json, os
from pathlib import Path

EXPS = [
  "pruneonly_garden_i15k_tau30",
  "pruneonly_bicycle_i15k_tau30",
  "pruneonly_bonsai_i15k_tau30",
  "pruneonly_kitchen_i15k_tau30",
  "gated_garden_i15k_noprune",
  "random_garden_i15k_keep45"
]

def ply_size(exp):
    pc = Path("output") / exp / "point_cloud"
    if not pc.exists():
        return None

    iters = sorted(pc.glob("iteration_*"))
    if not iters:
        return None

    ply = iters[-1] / "point_cloud.ply"
    if not ply.exists():
        return None

    return round(ply.stat().st_size / (1024*1024), 2)

print("\n===== Supplementary Summary =====\n")

for exp in EXPS:
    rpath = Path("output") / exp / "results.json"
    if not rpath.exists():
        print(f"{exp}: results.json not found")
        continue

    with open(rpath) as f:
        data = json.load(f)

    key = list(data.keys())[0]
    m = data[key]

    size = ply_size(exp)

    print(f"{exp}")
    print(f"  PSNR : {m['PSNR']:.3f}")
    print(f"  SSIM : {m['SSIM']:.3f}")
    print(f"  LPIPS: {m['LPIPS']:.3f}")
    print(f"  Size : {size} MB\n")

import torch
import argparse
from scene import Scene, GaussianModel

def load_model(model_path):
    gaussians = GaussianModel(3, "adam")
    scene = Scene(None, gaussians)
    scene.load(model_path)
    return gaussians, scene

def save_model(gaussians, scene, out_path):
    scene.save_to_path(out_path)

def prune_by_opacity(gaussians, target_num):
    with torch.no_grad():
        op = gaussians.get_opacity.squeeze()
        idx = torch.argsort(op, descending=True)[:target_num]

        gaussians._xyz = torch.nn.Parameter(gaussians._xyz[idx])
        gaussians._features_dc = torch.nn.Parameter(gaussians._features_dc[idx])
        gaussians._features_rest = torch.nn.Parameter(gaussians._features_rest[idx])
        gaussians._opacity = torch.nn.Parameter(gaussians._opacity[idx])
        gaussians._scaling = torch.nn.Parameter(gaussians._scaling[idx])
        gaussians._rotation = torch.nn.Parameter(gaussians._rotation[idx])

def prune_random(gaussians, target_num, seed=0):
    torch.manual_seed(seed)
    N = gaussians.get_xyz.shape[0]
    perm = torch.randperm(N)[:target_num]

    gaussians._xyz = torch.nn.Parameter(gaussians._xyz[perm])
    gaussians._features_dc = torch.nn.Parameter(gaussians._features_dc[perm])
    gaussians._features_rest = torch.nn.Parameter(gaussians._features_rest[perm])
    gaussians._opacity = torch.nn.Parameter(gaussians._opacity[perm])
    gaussians._scaling = torch.nn.Parameter(gaussians._scaling[perm])
    gaussians._rotation = torch.nn.Parameter(gaussians._rotation[perm])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--target_num", type=int, required=True)
    parser.add_argument("--mode", choices=["opacity", "random"], required=True)
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()

    gaussians, scene = load_model(args.input)

    before = gaussians.get_xyz.shape[0]
    print(f"Before pruning: {before}")

    if args.mode == "opacity":
        prune_by_opacity(gaussians, args.target_num)
    else:
        prune_random(gaussians, args.target_num, seed=args.seed)

    after = gaussians.get_xyz.shape[0]
    print(f"After pruning: {after}")

    save_model(gaussians, scene, args.output)

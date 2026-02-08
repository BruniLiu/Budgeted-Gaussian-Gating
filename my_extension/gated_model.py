import torch
from torch import nn
from scene.gaussian_model import GaussianModel

from .gating_network import DensifyGate
from .losses import total_gating_loss


class GatedGaussianModel(GaussianModel):
    """
    Extended GaussianModel with learnable gating mechanism.

    Adds:
    - gating MLP (DensifyGate)
    - gating loss computation
    - gate score stats
    - hard pruning utility (actually removes Gaussians)
    - prune-only baselines (random / opacity)

    All original behaviors are preserved unless use_gating=True.
    """

    def __init__(self, sh_degree, optimizer_type, use_gating=False):
        super().__init__(sh_degree, optimizer_type)

        self.use_gating = use_gating
        self.gate = DensifyGate().to("cuda") if self.use_gating else None

        # cached values
        self._last_gate_scores = None
        self._last_opt = None

        # gating hyperparams
        self.gate_lr_default = 1e-3
        self.gate_w_min = 0.2

    def training_setup(self, opt):
        super().training_setup(opt)

        self._last_opt = opt

        if self.use_gating and self.gate is not None:
            print("[Gating] Using separate optimizer for gate.")
            self.gate_optimizer = torch.optim.Adam(
                self.gate.parameters(),
                lr=getattr(opt, "gate_lr", self.gate_lr_default),
            )


    def get_gate_input(self):
        xyz = self.get_xyz
        scaling = self.get_scaling
        opacity = self.get_opacity
        return torch.cat([xyz, scaling, opacity], dim=1)


    def compute_gate_scores(self):
        if not self.use_gating or self.gate is None:
            return None

        feat = self.get_gate_input()
        scores = self.gate(feat)

        self._last_gate_scores = scores.detach()
        return scores

    @torch.no_grad()
    def get_gate_stats(self):
        if not self.use_gating or self.gate is None:
            return None

        w = self._last_gate_scores
        if w is None:
            return None

        return {
            "mean": float(w.mean().item()),
            "min": float(w.min().item()),
            "max": float(w.max().item()),
            "lt_05": float((w < 0.5).float().mean().item()),
            "lt_03": float((w < 0.3).float().mean().item()),
        }


    def compute_gating_loss(self, opt):
        if not self.use_gating or self.gate is None:
            return torch.tensor(0.0, device=self.get_xyz.device)

        scores = self.compute_gate_scores()
        if scores is None:
            return torch.tensor(0.0, device=self.get_xyz.device)

        return total_gating_loss(
            scores,
            lambda_sparse=getattr(opt, "lambda_sparse", 0.1),
            gamma_reg=getattr(opt, "gamma_reg", 0.01),
            reg_type=getattr(opt, "gate_reg_type", "entropy"),
        )



    @torch.no_grad()
    def _apply_prune_mask(self, mask: torch.Tensor):
        """
        Core pruning function shared by all strategies.
        """
        mask = mask.to(self._xyz.device)
        after = int(mask.sum().item())

        self._xyz = nn.Parameter(self._xyz[mask])
        self._features_dc = nn.Parameter(self._features_dc[mask])
        self._features_rest = nn.Parameter(self._features_rest[mask])
        self._opacity = nn.Parameter(self._opacity[mask])
        self._scaling = nn.Parameter(self._scaling[mask])
        self._rotation = nn.Parameter(self._rotation[mask])

        dev = self._xyz.device

        if hasattr(self, "xyz_gradient_accum"):
            self.xyz_gradient_accum = torch.zeros((after, 1), device=dev)

        if hasattr(self, "denom"):
            self.denom = torch.zeros((after, 1), device=dev)

        if hasattr(self, "max_radii2D"):
            self.max_radii2D = torch.zeros((after,), device=dev)

        print(f"[Pruning] After: {after} Gaussians")


    @torch.no_grad()
    def hard_prune(self, threshold=0.3):
        if not self.use_gating or self.gate is None:
            print("[Pruning] Gating disabled.")
            return

        scores = self.compute_gate_scores()
        if scores is None:
            return

        mask = (scores.squeeze(-1) >= threshold)

        before = self.get_xyz.shape[0]
        after = mask.sum().item()

        print(f"[Pruning] Before: {before}")
        print(f"[Pruning] Keeping: {after}")

        self._apply_prune_mask(mask)

   
    # 2) Opacity-based prune-only
   
    @torch.no_grad()
    def hard_prune_by_opacity(self, keep_ratio: float):
        """
        Prune without gating: keep highest-opacity Gaussians
        """
        assert 0 < keep_ratio <= 1.0

        N = self.get_xyz.shape[0]
        K = max(1, int(N * keep_ratio))

        op = self.get_opacity.squeeze(-1)

        _, idx = torch.topk(op, k=K, largest=True)

        mask = torch.zeros(N, dtype=torch.bool, device=op.device)
        mask[idx] = True

        print(f"[Prune-Only:Opacity] keep_ratio={keep_ratio:.3f}")
        print(f"[Prune-Only] Before: {N}, Keeping: {K}")

        self._apply_prune_mask(mask)

  
    @torch.no_grad()
    def hard_prune_random(self, keep_ratio: float, seed: int = 0):
        """
        Random baseline pruning
        """
        assert 0 < keep_ratio <= 1.0

        N = self.get_xyz.shape[0]
        K = max(1, int(N * keep_ratio))

        g = torch.Generator(device=self.get_xyz.device)
        g.manual_seed(seed)

        perm = torch.randperm(N, generator=g, device=self.get_xyz.device)
        idx = perm[:K]

        mask = torch.zeros(N, dtype=torch.bool, device=self.get_xyz.device)
        mask[idx] = True

        print(f"[Prune-Only:Random] keep_ratio={keep_ratio:.3f}, seed={seed}")
        print(f"[Prune-Only] Before: {N}, Keeping: {K}")

        self._apply_prune_mask(mask)


    def modulated_opacity(self):
        if not self.use_gating or self.gate is None:
            return self.get_opacity

        scores = self.compute_gate_scores()
        if scores is None:
            return self.get_opacity

        w_min = self.gate_w_min
        scores = w_min + (1.0 - w_min) * scores

        return self.get_opacity * scores

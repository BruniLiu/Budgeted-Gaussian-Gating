# Budgeted Gaussian Gating

**Learnable sparsification for 3D Gaussian Splatting (3DGS)** via a lightweight gating network that controls densification/pruning under an explicit budget.

This repository explores a simple idea:  
> instead of densifying Gaussians blindly, learn an *importance score* for each Gaussian and use it to keep the representation compact while preserving rendering quality.

---

## Key Idea

3DGS can produce a large number of Gaussian primitives during training, which increases storage and runtime overhead.  
We attach a small **gating network** to predict an **importance score** per Gaussian, enabling:

- **Differentiable, learnable sparsification** during training
- **Budget-aware control** (e.g., keep ratio / target size)
- **Comparable baselines** (random / prune-only) for fair ablations

> This repo is intended as a research codebase for experimentation & reporting (see `paper_reports/`).

---

## Features

- Gating network integrated into the 3DGS training pipeline
- Multiple pruning strategies:
  - **Gate-based** pruning (learned)
  - **Random** pruning baseline
  - **Offline prune** utilities for post-training ablations
- End-to-end evaluation + result aggregation scripts
- Trade-off visualization utilities (quality ↔ size)

---

## Repository Structure

Core folders / entry points:

- `my_extension/` – gating modules & custom logic (core contribution)
- `scene/`, `gaussian_renderer/`, `utils/` – 3DGS components
- `train.py` – baseline training entry
- `train_gated.py` – training with gating enabled
- `render.py` – rendering / novel view synthesis
- `offline_prune.py` – post-training pruning tools
- `benchmark.py`, `full_eval.py`, `metrics.py` – evaluation & metrics
- `collect_results.py`, `collect_supplementary.py` – results aggregation
- `plot_tradeoff.py` – visualize size/quality trade-offs
- `run_all.sh`, `eval_all.sh`, `eval_supplementary.sh`, `exp_commands.sh` – automation scripts
- `paper_reports/` – experiment reports / writeups

# Budgeted Gaussian Gating

A learnable sparsification framework for 3D Gaussian Splatting (3DGS), enabling compact representations through adaptive gating-based densification control.

## Overview

This project introduces a lightweight gating mechanism to improve the efficiency of 3D Gaussian Splatting by learning to control the importance of each Gaussian primitive during training.

The key idea is to attach a small neural gating network that predicts importance scores for Gaussians, allowing:

- Differentiable pruning during training  
- Adaptive budget control  
- Significant reduction in model size  
- Minimal loss in rendering quality  

## Main Features

- Learnable gating network integrated into 3DGS training pipeline  
- Multiple pruning strategies:
  - gate-based pruning  
  - random pruning baseline  
  - keep-ratio alignment  
- Automated experiment scripts for large-scale comparisons  
- Comprehensive evaluation metrics collection  
- Reproducible benchmarking pipeline  

## Repository Structure

- `my_extension/` – Core implementation of gating modules  
- `train_gated.py` – Training entry point with gating  
- `offline_prune.py` – Post-training pruning tools  
- `run_*.sh` – Automated experiment scripts  
- `paper_reports/` – Experiment summaries and analysis  
- `collect_results.py` – Metrics aggregation  
- `plot_tradeoff.py` – Visualization tools  

## How to Run

### Training with Gating

```bash
python train_gated.py -s <dataset_path> --use_gating

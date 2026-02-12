# Budgeted Gaussian Gating

**Official Implementation of**
**“Budget-Constrained Learnable Gating for Compact 3D Gaussian Splatting”**

**Project Page:** [https://github.com/BruniLiu/Budgeted-Gaussian-Gating](https://github.com/BruniLiu/Budgeted-Gaussian-Gating)

---

## Overview

3D Gaussian Splatting (3DGS) achieves real-time high-quality novel view synthesis through differentiable rasterization and progressive densification. However, this densification strategy often produces a large number of redundant Gaussian primitives, leading to excessive model size and memory usage.

This repository implements **Budgeted Gaussian Gating (BGG)**—a learnable sparsification framework that enables:

* Differentiable importance estimation for each Gaussian
* Explicit control of final model budget
* Training-time sparsification without disrupting densification
* Simple one-shot pruning after convergence

Unlike traditional prune–refine pipelines, our method performs **end-to-end, budget-aware compression** fully compatible with the original 3DGS workflow.

---

## Key Contributions

* **Densification-Compatible Gating**
  Soft modulation of Gaussians instead of hard deletion during training

* **Explicit Budget Constraint**
  Direct control over target retention ratio

* **Lightweight Gating Network**
  Per-Gaussian importance estimation with negligible overhead

* **Simple Deployment**
  One-shot post-training pruning without fine-tuning

* **Extensive Evaluation Toolkit**
  Automated scripts for training, evaluation, ablation, and visualization

---

## Repository Structure

The project is organized as follows:

### Core folders / entry points

* **`my_extension/`**
  Gating modules and custom logic — **core contribution of this project**

* **`scene/`, `gaussian_renderer/`, `utils/`**
  Original 3DGS components and supporting utilities

* **`train.py`**
  Baseline 3DGS training entry point

* **`train_gated.py`**
  Training script with learnable gating enabled

* **`render.py`**
  Novel view synthesis and rendering

* **`offline_prune.py`**
  Post-training one-shot pruning tools

* **`benchmark.py`, `full_eval.py`, `metrics.py`**
  Quantitative evaluation and metric computation

* **`collect_results.py`, `collect_supplementary.py`**
  Aggregation of experiment outputs into structured tables

* **`plot_tradeoff.py`**
  Visualization of size–quality trade-offs

* **Automation scripts**

  * `run_all.sh`
  * `eval_all.sh`
  * `eval_supplementary.sh`
  * `exp_commands.sh`

* **`paper_reports/`**
  Experiment reports and technical writeups

---

## Installation

```bash
git clone https://github.com/BruniLiu/Budgeted-Gaussian-Gating.git
cd Budgeted-Gaussian-Gating
pip install -r requirements.txt
```

This project builds directly upon the official 3D Gaussian Splatting implementation.

---

## Usage

### 1. Train baseline 3DGS

```bash
python train.py \
  -s datasets/mipnerf360/garden \
  --iterations 15000 \
  --eval \
  -m output/baseline_garden
```

### 2. Train with Budgeted Gating

```bash
python train_gated.py \
  -s datasets/mipnerf360/garden \
  --iterations 15000 \
  --use_gating \
  --target_ratio 0.5 \
  -m output/gated_garden
```

### 3. Render results

```bash
python render.py -m output/gated_garden
```

### 4. Evaluate metrics

```bash
python metrics.py -m output/gated_garden
```

### 5. Post-training Pruning

```bash
python offline_prune.py \
  -m output/gated_garden \
  --threshold 0.3
```

---

## Experiments

The repository provides full automation for reproducing experiments:

```bash
bash run_all.sh
bash eval_all.sh
```

You can aggregate results and visualize trade-offs:

```bash
python collect_results.py
python plot_tradeoff.py
```

Supported experiment types:

* Baseline 3DGS
* Gate-only
* Prune-only
* Random pruning
* Threshold sweep
* Ablation without budget constraint

---

## Highlights

Our method consistently demonstrates:

* Significant model compression
* Better preservation of rendering quality
* Stable performance compared with naive pruning
* Smooth controllability over quality–size trade-offs

---

## Acknowledgements

This work is built upon:

* 3D Gaussian Splatting (Kerbl et al., 2023)
* Mip-NeRF360 dataset
* Prior research on neural rendering and model compression

---

## Contact

**Author:** Xiangyi Liu
**Email:** [20242802115@smail.lnu.edu.cn](mailto:20242802115@smail.lnu.edu.cn)

Feel free to open issues or pull requests for questions and contributions!


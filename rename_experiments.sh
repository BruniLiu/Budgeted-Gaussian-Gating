#!/bin/bash

cd output

echo "===== Renaming Experiments for Clarity ====="

# ---- Baselines ----
mv baseline_garden_i15k_r1   baseline_garden_i15k_noprune
mv baseline_bicycle_i15k_r1  baseline_bicycle_i15k_noprune
mv baseline_bonsai_i15k_r1   baseline_bonsai_i15k_noprune
mv baseline_kitchen_i15k_r1  baseline_kitchen_i15k_noprune

# ---- Gated Pruned (tau=0.3) ----
mv gated_garden_i15k_r1_pruned   gated_garden_i15k_tau30
mv gated_bicycle_i15k_r1_pruned  gated_bicycle_i15k_tau30
mv gated_bonsai_i15k_r1_pruned   gated_bonsai_i15k_tau30
mv gated_kitchen_i15k_r1_pruned  gated_kitchen_i15k_tau30

# ---- Sweeps ----
mv sweep_garden_tau02  gated_garden_i15k_tau20
mv sweep_garden_tau04  gated_garden_i15k_tau40
mv sweep_garden_tau05  gated_garden_i15k_tau50

# ---- Ablation ----
mv ablation_no_gateloss  gated_garden_i15k_nogateloss

echo "===== Rename Finished ====="

ls -l

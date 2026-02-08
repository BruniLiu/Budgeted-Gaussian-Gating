#!/bin/bash

# ===== Common Config =====

DATA_ROOT="datasets/mipnerf360"
SCRIPT="train_gated.py"

ITERS=15000
ITERS30=30000

SCENES=("garden" "bicycle" "bonsai" "kitchen")

TAUS=(20 30 40 50)
KEEP_RATIOS=(30 45 60)

# 1. Baselines

for scene in "${SCENES[@]}"; do
  python ${SCRIPT} \
    -s ${DATA_ROOT}/${scene} \
    --iterations ${ITERS} \
    -m output/baseline_${scene}_i15k_noprune \
    --eval
done


# 2. Gate-Only (no prune)

for scene in "${SCENES[@]}"; do
  python ${SCRIPT} \
    -s ${DATA_ROOT}/${scene} \
    --iterations ${ITERS} \
    -m output/gated_${scene}_i15k_noprune \
    --use_gating \
    --prune_mode none \
    --eval
done


# 3. Full τ sweep (Gated + PruneOnly) 

for scene in "${SCENES[@]}"; do
  for tau in "${TAUS[@]}"; do

    # prune-only sweep
    python ${SCRIPT} \
      -s ${DATA_ROOT}/${scene} \
      --iterations ${ITERS} \
      -m output/pruneonly_${scene}_i15k_tau${tau} \
      --prune_mode opacity \
      --prune_threshold 0.${tau} \
      --eval

    # gated sweep
    python ${SCRIPT} \
      -s ${DATA_ROOT}/${scene} \
      --iterations ${ITERS} \
      -m output/gated_${scene}_i15k_tau${tau} \
      --use_gating \
      --prune_mode gate \
      --prune_threshold 0.${tau} \
      --eval

  done
done


#  4. Keep-ratio aligned comparison

for scene in "${SCENES[@]}"; do
  for keep in "${KEEP_RATIOS[@]}"; do

    python ${SCRIPT} \
      -s ${DATA_ROOT}/${scene} \
      --iterations ${ITERS} \
      -m output/random_${scene}_i15k_keep${keep} \
      --prune_mode random \
      --keep_ratio 0.${keep} \
      --eval

    python ${SCRIPT} \
      -s ${DATA_ROOT}/${scene} \
      --iterations ${ITERS} \
      -m output/pruneonly_${scene}_i15k_keep${keep} \
      --prune_mode opacity \
      --keep_ratio 0.${keep} \
      --eval

    python ${SCRIPT} \
      -s ${DATA_ROOT}/${scene} \
      --iterations ${ITERS} \
      -m output/gated_${scene}_i15k_keep${keep} \
      --use_gating \
      --prune_mode gate \
      --keep_ratio 0.${keep} \
      --eval

  done
done


#  5. Ablation: no gateloss (2 scenes) 

for scene in garden bicycle; do
  python ${SCRIPT} \
    -s ${DATA_ROOT}/${scene} \
    --iterations ${ITERS} \
    -m output/gated_${scene}_i15k_nogateloss \
    --use_gating \
    --lambda_sparse 0.0 \
    --gamma_reg 0.0 \
    --prune_mode gate \
    --prune_threshold 0.30 \
    --eval
done


# 6. Convergence Check (30k) 

for scene in garden bicycle; do

  python ${SCRIPT} \
    -s ${DATA_ROOT}/${scene} \
    --iterations ${ITERS30} \
    -m output/baseline_${scene}_i30k_noprune \
    --eval

  python ${SCRIPT} \
    -s ${DATA_ROOT}/${scene} \
    --iterations ${ITERS30} \
    -m output/gated_${scene}_i30k_tau30 \
    --use_gating \
    --prune_mode gate \
    --prune_threshold 0.30 \
    --eval

  python ${SCRIPT} \
    -s ${DATA_ROOT}/${scene} \
    --iterations ${ITERS30} \
    -m output/pruneonly_${scene}_i30k_tau30 \
    --prune_mode opacity \
    --prune_threshold 0.30 \
    --eval

done


# 7. FPS + Memory Evaluation 

echo "Running FPS benchmark..."

for d in output/*; do
  if [ -d "$d" ]; then
    scene=$(echo $d | cut -d'_' -f2)

    python benchmark.py \
      --model $d \
      --data ${DATA_ROOT}/${scene}
  fi
done


# 8. Collect Results 

python collect_results.py

echo "All experiments completed."

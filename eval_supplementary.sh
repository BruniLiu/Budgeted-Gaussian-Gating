#!/bin/bash
set -e

echo "===== Supplementary Experiments Evaluation ====="

EXPS=(
  pruneonly_garden_i15k_tau30
  pruneonly_bicycle_i15k_tau30
  pruneonly_bonsai_i15k_tau30
  pruneonly_kitchen_i15k_tau30

  gated_garden_i15k_noprune

  random_garden_i15k_keep45
)

mkdir -p output/_eval_logs

for exp in "${EXPS[@]}"
do
  MODEL="output/${exp}"

  if [ ! -d "$MODEL" ]; then
    echo "[SKIP] ${exp} (directory not found)"
    continue
  fi

  echo "----- Rendering: ${exp} -----"
  python render.py -m "$MODEL" \
    | tee "output/_eval_logs/${exp}_render.log"

  echo "----- Metrics: ${exp} -----"
  python metrics.py -m "$MODEL" \
    | tee "output/_eval_logs/${exp}_metrics.log"

done

echo "===== All Supplementary Evaluations Done ====="

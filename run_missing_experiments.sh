set -e

echo "Running ONLY Necessary Missing Experiments"

SCENES=("garden" "bicycle" "bonsai" "kitchen")
DATA="datasets/mipnerf360"
ITERS=15000
SEED=0

TAU="0.3"
KEEP_RATIO="0.45"

has_result () {
  [[ -f "output/$1/results.json" ]]
}

run_if_missing () {
  local name="$1"
  shift
  if has_result "$name"; then
    echo "[SKIP] $name already exists"
  else
    echo "[RUN ] $name"
    "$@"
  fi
}


# 1. PRUNE-ONLY BASELINES 


for scene in "${SCENES[@]}"
do
  NAME="pruneonly_${scene}_i15k_tau30"

  run_if_missing "$NAME" \
    python train_gated.py \
      -s ${DATA}/${scene} \
      -m output/${NAME} \
      --iterations ${ITERS} \
      --eval \
      --prune_mode opacity \
      --prune_threshold ${TAU} \
      --seed ${SEED}
done



# 2. GATE ENABLED BUT NO PRUNING 


NAME="gated_garden_i15k_noprune"

run_if_missing "$NAME" \
  python train_gated.py \
    -s ${DATA}/garden \
    -m output/${NAME} \
    --iterations ${ITERS} \
    --eval \
    --use_gating \
    --prune_mode none \
    --seed ${SEED}


# 3. RANDOM PRUNING ABLATION


NAME="random_garden_i15k_keep45"

run_if_missing "$NAME" \
  python train_gated.py \
    -s ${DATA}/garden \
    -m output/${NAME} \
    --iterations ${ITERS} \
    --eval \
    --prune_mode random \
    --keep_ratio ${KEEP_RATIO} \
    --seed ${SEED}


echo "===== Essential Experiments Completed ====="

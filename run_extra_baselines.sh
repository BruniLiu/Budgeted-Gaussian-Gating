BASE_DIR="output"
SCENE="garden"
ITER=15000

TARGET_NUM=2031532

BASELINE_MODEL="$BASE_DIR/baseline_${SCENE}_i15k_r1"
OUT_PREFIX="$BASE_DIR/extra_${SCENE}"

#  1. Prune-only (opacity) 

python offline_prune.py \
  --input $BASELINE_MODEL \
  --output ${OUT_PREFIX}_pruneonly \
  --target_num $TARGET_NUM \
  --mode opacity

python render.py -m ${OUT_PREFIX}_pruneonly
python metrics.py -m ${OUT_PREFIX}_pruneonly


# 2. Random prune 

for SEED in 0 1 2
do
  python offline_prune.py \
    --input $BASELINE_MODEL \
    --output ${OUT_PREFIX}_random_seed${SEED} \
    --target_num $TARGET_NUM \
    --mode random \
    --seed $SEED

  python render.py -m ${OUT_PREFIX}_random_seed${SEED}
  python metrics.py -m ${OUT_PREFIX}_random_seed${SEED}
done

#!/bin/bash
set -e

# =========================
# CONFIG (you edit here)
# =========================
DATA_ROOT="datasets/mipnerf360"
SCENES=("garden" "bicycle" "bonsai" "kitchen")

ITER=15000
RES=1

# Gate prune hyperparams
PRUNE_MODE_GATE="gate"
PRUNE_TAU=0.30

# Prune-only hyperparams
KEEP_RATIO=0.45
RANDOM_SEEDS=(0 1 2)

# Root folder for outputs (WON'T overwrite if you keep names unique)
OUT="output"

# If you pass --run, it will execute commands. Otherwise only prints.
DO_RUN=0
if [ "$1" == "--run" ]; then
  DO_RUN=1
fi

# =========================
# helpers
# =========================
run_or_print () {
  local title="$1"
  local cmd="$2"

  echo ""
  echo "============================================================"
  echo "${title}"
  echo "CMD:"
  echo "${cmd}"
  echo "============================================================"

  if [ "${DO_RUN}" -eq 1 ]; then
    eval "${cmd}"
  fi
}

# =========================
# MAIN
# =========================
echo ""
echo "########## EXPERIMENT COMMAND LIST ##########"
echo "ITER=${ITER}  RES=${RES}  KEEP_RATIO=${KEEP_RATIO}  TAU=${PRUNE_TAU}"
echo "Run mode: $([ "${DO_RUN}" -eq 1 ] && echo EXECUTE || echo PRINT ONLY)"
echo "#############################################"

for scene in "${SCENES[@]}"; do

  echo ""
  echo "##############################"
  echo "SCENE: ${scene}"
  echo "##############################"

  # -------------------------
  # (A) Baseline
  # -------------------------
  EXP_A="${OUT}/${scene}__baseline__i${ITER}__r${RES}"
  CMD_A="python train_gated.py -s ${DATA_ROOT}/${scene} -m ${EXP_A} --iterations ${ITER} --eval --prune_mode none"
  run_or_print "[A] Baseline (no pruning) -> ${EXP_A}" "${CMD_A}"

  CMD_A_RENDER="python render.py -m ${EXP_A}"
  run_or_print "[A-render] Render baseline -> ${EXP_A}" "${CMD_A_RENDER}"

  CMD_A_METRICS="python metrics.py -m ${EXP_A} > ${EXP_A}/metrics.txt"
  run_or_print "[A-metrics] Metrics baseline -> ${EXP_A}" "${CMD_A_METRICS}"

  # -------------------------
  # (B) Ours: Gate + prune
  # -------------------------
  EXP_B="${OUT}/${scene}__gated__i${ITER}__tau${PRUNE_TAU}"
  CMD_B="python train_gated.py -s ${DATA_ROOT}/${scene} -m ${EXP_B} --iterations ${ITER} --eval --use_gating --prune_mode ${PRUNE_MODE_GATE} --prune_threshold ${PRUNE_TAU}"
  run_or_print "[B] Ours (gating + prune) -> ${EXP_B}" "${CMD_B}"

  CMD_B_RENDER="python render.py -m ${EXP_B}"
  run_or_print "[B-render] Render ours -> ${EXP_B}" "${CMD_B_RENDER}"

  CMD_B_METRICS="python metrics.py -m ${EXP_B} > ${EXP_B}/metrics.txt"
  run_or_print "[B-metrics] Metrics ours -> ${EXP_B}" "${CMD_B_METRICS}"

  # -------------------------
  # (C) Prune-only: opacity
  # -------------------------
  EXP_C="${OUT}/${scene}__prune_opacity__i${ITER}__keep${KEEP_RATIO}"
  CMD_C="python train_gated.py -s ${DATA_ROOT}/${scene} -m ${EXP_C} --iterations ${ITER} --eval --prune_mode opacity --keep_ratio ${KEEP_RATIO}"
  run_or_print "[C] Prune-only (opacity) -> ${EXP_C}" "${CMD_C}"

  CMD_C_RENDER="python render.py -m ${EXP_C}"
  run_or_print "[C-render] Render opacity-only -> ${EXP_C}" "${CMD_C_RENDER}"

  CMD_C_METRICS="python metrics.py -m ${EXP_C} > ${EXP_C}/metrics.txt"
  run_or_print "[C-metrics] Metrics opacity-only -> ${EXP_C}" "${CMD_C_METRICS}"

  # -------------------------
  # (D) Prune-only: random (3 seeds)
  # -------------------------
  for sd in "${RANDOM_SEEDS[@]}"; do
    EXP_D="${OUT}/${scene}__prune_random__i${ITER}__keep${KEEP_RATIO}__seed${sd}"
    CMD_D="python train_gated.py -s ${DATA_ROOT}/${scene} -m ${EXP_D} --iterations ${ITER} --eval --prune_mode random --keep_ratio ${KEEP_RATIO} --seed ${sd}"
    run_or_print "[D] Prune-only (random, seed=${sd}) -> ${EXP_D}" "${CMD_D}"

    CMD_D_RENDER="python render.py -m ${EXP_D}"
    run_or_print "[D-render] Render random-only seed=${sd} -> ${EXP_D}" "${CMD_D_RENDER}"

    CMD_D_METRICS="python metrics.py -m ${EXP_D} > ${EXP_D}/metrics.txt"
    run_or_print "[D-metrics] Metrics random-only seed=${sd} -> ${EXP_D}" "${CMD_D_METRICS}"
  done

done

echo ""
echo "########## DONE ##########"
echo "Tip: run './exp_commands.sh --run' to execute everything."

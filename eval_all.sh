#!/bin/bash

ROOT="output"

echo "===== Full Auto Evaluation Script ====="

for exp in ${ROOT}/*; do

    if [ ! -d "$exp" ]; then
        continue
    fi

    echo "==========================================="
    echo "[Processing] $exp"
    echo "==========================================="

    # 1. 先跑 render
    echo "[Step 1] Rendering..."
    python render.py -m "$exp"

    # 2. 再跑 metrics
    echo "[Step 2] Evaluating metrics..."
    python metrics.py -m "$exp" > "$exp/metrics.txt"

    echo "[Done] $exp"

done

echo "===== All Experiments Evaluated ====="

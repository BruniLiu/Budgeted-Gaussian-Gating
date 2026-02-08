set -euo pipefail

# Paper experiment suite:
#  - Collect metrics from existing experiment folders
#  - Validate pruning actually happened (size/#gaussians)
#  - Generate summary CSV + Markdown + LaTeX
#  - Optionally run missing baselines (random multi-scene, keep45 fairness)

DATA_ROOT="${DATA_ROOT:-datasets/mipnerf360}"
OUT_ROOT="${OUT_ROOT:-output}"         
REPORT_DIR="${REPORT_DIR:-paper_reports}"
ITER="${ITER:-15000}"                 
RUN_MISSING="${RUN_MISSING:-0}"     
SCENES=("garden" "bicycle" "bonsai" "kitchen")


TRAIN_PY="${TRAIN_PY:-train.py}"
RENDER_PY="${RENDER_PY:-render.py}"



mkdir -p "$REPORT_DIR"

need_cmd() {
  command -v "$1" >/dev/null 2>&1 || { echo "Missing command: $1"; exit 1; }
}

need_cmd python
need_cmd find
need_cmd awk
need_cmd sed


JQ_AVAILABLE=0
if command -v jq >/dev/null 2>&1; then
  JQ_AVAILABLE=1
fi


ply_path() {
  local model_dir="$1"
  echo "${model_dir}/point_cloud/iteration_${ITER}/point_cloud.ply"
}

ply_num_gaussians() {
  local ply="$1"
  if [[ ! -f "$ply" ]]; then
    echo ""
    return
  fi

  awk '
    /^element vertex/ {print $3; exit}
    /^end_header/ {exit}
  ' "$ply" 2>/dev/null || true
}


file_size_mb() {
  local f="$1"
  if [[ ! -f "$f" ]]; then
    echo ""
    return
  fi
  python - <<PY
import os
print(round(os.path.getsize("$f")/1024/1024, 2))
PY
}

read_metrics_json() {
  local model_dir="$1"
  local mj="${model_dir}/test/metrics.json"
  if [[ ! -f "$mj" ]]; then
    echo "|||"
    return
  fi

  if [[ "$JQ_AVAILABLE" -eq 1 ]]; then
    local psnr ssim lpips
    psnr=$(jq -r '.psnr // empty' "$mj" 2>/dev/null || true)
    ssim=$(jq -r '.ssim // empty' "$mj" 2>/dev/null || true)
    lpips=$(jq -r '.lpips // empty' "$mj" 2>/dev/null || true)
    echo "${psnr}|${ssim}|${lpips}"
  else
    python - <<PY
import json
p="$mj"
d=json.load(open(p))
psnr=d.get("psnr","")
ssim=d.get("ssim","")
lpips=d.get("lpips","")
print(f"{psnr}|{ssim}|{lpips}")
PY
  fi
}


infer_scene() {
  local name="$1"
  for s in "${SCENES[@]}"; do
    if echo "$name" | grep -qi "$s"; then
      echo "$s"; return
    fi
  done
  echo ""
}

infer_method() {
  local name="$1"

  if echo "$name" | grep -qi "pruneonly\|opacity"; then echo "prune-only"; return; fi
  if echo "$name" | grep -qi "random"; then echo "random"; return; fi
  if echo "$name" | grep -qi "gated" && echo "$name" | grep -qi "noprune"; then echo "gated(no-prune)"; return; fi
  if echo "$name" | grep -qi "gated\|gate" && echo "$name" | grep -qi "prune\|tau\|thr"; then echo "gated+pruned"; return; fi
  if echo "$name" | grep -qi "baseline\|vanilla\|none"; then echo "baseline"; return; fi
  echo "unknown"
}


run_train_render() {
  local scene="$1"
  local tag="$2"
  local extra_args="$3"
  local mdir="${OUT_ROOT}/${tag}"

  echo "[RUN] ${scene} -> ${mdir}"
  python "$TRAIN_PY" -s "${DATA_ROOT}/${scene}" --iterations "$ITER" -m "$mdir" $extra_args
  python "$RENDER_PY" -m "$mdir"
}


CSV="${REPORT_DIR}/summary_all.csv"
MD="${REPORT_DIR}/summary_all.md"
TEX="${REPORT_DIR}/summary_all.tex"
CHECK="${REPORT_DIR}/consistency_check.md"

echo "scene,exp,method,psnr,ssim,lpips,num_gaussians,ply_mb,metrics_json" > "$CSV"

mapfile -t MODEL_DIRS < <(find "$OUT_ROOT" -type f -path "*/point_cloud/iteration_${ITER}/point_cloud.ply" -print 2>/dev/null | sed "s#/point_cloud/iteration_${ITER}/point_cloud.ply##" | sort)

echo "[INFO] Found ${#MODEL_DIRS[@]} models with iteration_${ITER} ply."

for d in "${MODEL_DIRS[@]}"; do
  base="$(basename "$d")"
  scene="$(infer_scene "$base")"
  method="$(infer_method "$base")"

  ply="$(ply_path "$d")"
  ng="$(ply_num_gaussians "$ply")"
  mb="$(file_size_mb "$ply")"

  mtr="$(read_metrics_json "$d")"
  psnr="${mtr%%|*}"
  rest="${mtr#*|}"
  ssim="${rest%%|*}"
  lpips="${rest#*|}"
  mj="${d}/test/metrics.json"

  echo "${scene},${base},${method},${psnr},${ssim},${lpips},${ng},${mb},${mj}" >> "$CSV"
done


{
  echo "| Scene | Exp | Method | PSNR | SSIM | LPIPS | #Gaussians | PLY (MB) |"
  echo "|---|---|---|---:|---:|---:|---:|---:|"
  tail -n +2 "$CSV" | while IFS=, read -r scene exp method psnr ssim lpips ng mb mj; do
    echo "| ${scene:-NA} | ${exp} | ${method} | ${psnr:-NA} | ${ssim:-NA} | ${lpips:-NA} | ${ng:-NA} | ${mb:-NA} |"
  done
} > "$MD"



{
  echo "\\begin{tabular}{l l l c c c r r}"
  echo "\\hline"
  echo "Scene & Exp & Method & PSNR & SSIM & LPIPS & \\#G & MB \\\\"
  echo "\\hline"
  tail -n +2 "$CSV" | while IFS=, read -r scene exp method psnr ssim lpips ng mb mj; do
    s="${scene:-NA}"
    p="${psnr:-NA}"
    si="${ssim:-NA}"
    l="${lpips:-NA}"
    g="${ng:-NA}"
    m="${mb:-NA}"

    exp_tex=$(echo "$exp" | sed 's/_/\\_/g')
    meth_tex=$(echo "$method" | sed 's/_/\\_/g')
    echo "${s} & ${exp_tex} & ${meth_tex} & ${p} & ${si} & ${l} & ${g} & ${m} \\\\"
  done
  echo "\\hline"
  echo "\\end{tabular}"
} > "$TEX"



{
  echo "# Consistency Check"
  echo ""
  echo "This report flags potential *size/metric mismatches* (e.g., prune-only size unexpectedly close to unpruned)."
  echo ""
  echo "| Scene | Exp | Method | PSNR | SSIM | LPIPS | #G | MB | Flag |"
  echo "|---|---|---|---:|---:|---:|---:|---:|---|"
  tail -n +2 "$CSV" | while IFS=, read -r scene exp method psnr ssim lpips ng mb mj; do
    flag=""
    if [[ "$method" == "prune-only" && -n "$mb" ]]; then
    
      python - <<PY
mb=float("${mb}") if "${mb}" else 0.0
print("FLAG_TOO_LARGE" if mb>800 else "")
PY
    fi
 
    f=$(python - <<PY
mb=float("${mb}") if "${mb}" else 0.0
print("FLAG_TOO_LARGE" if mb>800 else "")
PY
)
    if [[ -n "$f" ]]; then flag="$f"; fi

    echo "| ${scene:-NA} | ${exp} | ${method} | ${psnr:-NA} | ${ssim:-NA} | ${lpips:-NA} | ${ng:-NA} | ${mb:-NA} | ${flag:-OK} |"
  done
} > "$CHECK"

echo ""
echo "[DONE]"
echo "CSV : $CSV"
echo "MD  : $MD"
echo "TEX : $TEX"
echo "CHECK: $CHECK"

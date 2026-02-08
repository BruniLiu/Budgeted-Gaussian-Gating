echo "===== 3DGS Gating Experiments Automation ====="

#  Baselines 
echo "[1/12] Running Baseline - Garden"
python train.py \
  -s datasets/mipnerf360/garden \
  -m output/baseline_garden_i15k_r1 \
  --iterations 15000 \
  --eval

echo "[2/12] Running Baseline - Bicycle"
python train.py \
  -s datasets/mipnerf360/bicycle \
  -m output/baseline_bicycle_i15k_r1 \
  --iterations 15000 \
  --eval

echo "[3/12] Running Baseline - Bonsai"
python train.py \
  -s datasets/mipnerf360/bonsai \
  -m output/baseline_bonsai_i15k_r1 \
  --iterations 15000 \
  --eval

echo "[4/12] Running Baseline - Kitchen"
python train.py \
  -s datasets/mipnerf360/kitchen \
  -m output/baseline_kitchen_i15k_r1 \
  --iterations 15000 \
  --eval


#  Gated Main Experiments 

echo "[5/12] Gated + Pruned - Garden"
python train_gated.py \
  -s datasets/mipnerf360/garden \
  -m output/gated_garden_i15k_r1_pruned \
  --iterations 15000 \
  --eval \
  --use_gating \
  --prune_threshold 0.3

echo "[6/12] Gated + Pruned - Bicycle"
python train_gated.py \
  -s datasets/mipnerf360/bicycle \
  -m output/gated_bicycle_i15k_r1_pruned \
  --iterations 15000 \
  --eval \
  --use_gating \
  --prune_threshold 0.3

echo "[7/12] Gated + Pruned - Bonsai"
python train_gated.py \
  -s datasets/mipnerf360/bonsai \
  -m output/gated_bonsai_i15k_r1_pruned \
  --iterations 15000 \
  --eval \
  --use_gating \
  --prune_threshold 0.3

echo "[8/12] Gated + Pruned - Kitchen"
python train_gated.py \
  -s datasets/mipnerf360/kitchen \
  -m output/gated_kitchen_i15k_r1_pruned \
  --iterations 15000 \
  --eval \
  --use_gating \
  --prune_threshold 0.3


#  Threshold Sweep (Trade-off) 

echo "[9/12] Sweep Tau=0.2"
python train_gated.py \
  -s datasets/mipnerf360/garden \
  -m output/sweep_garden_tau02 \
  --iterations 15000 \
  --eval \
  --use_gating \
  --prune_threshold 0.2

echo "[10/12] Sweep Tau=0.4"
python train_gated.py \
  -s datasets/mipnerf360/garden \
  -m output/sweep_garden_tau04 \
  --iterations 15000 \
  --eval \
  --use_gating \
  --prune_threshold 0.4

echo "[11/12] Sweep Tau=0.5"
python train_gated.py \
  -s datasets/mipnerf360/garden \
  -m output/sweep_garden_tau05 \
  --iterations 15000 \
  --eval \
  --use_gating \
  --prune_threshold 0.5


#  Ablation 

echo "[12/12] Ablation: No Gating Loss"
python train_gated.py \
  -s datasets/mipnerf360/garden \
  -m output/ablation_no_gateloss \
  --iterations 15000 \
  --eval \
  --use_gating \
  --lambda_sparse 0.0 \
  --gamma_reg 0.0 \
  --prune_threshold 0.3

echo "===== All Experiments Finished ====="
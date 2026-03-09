#!/bin/bash
###############################################################################
#  StructRouter — Full Experiment Pipeline
#
#  This script lists every experiment needed for the paper.
#  Run sections individually or uncomment/comment as needed.
#
#  Prerequisites:
#    1. conda activate structrouter
#    2. pip install -r requirements.txt
#    3. Data in ./data/ (run: python scripts/download.py)
#
#  Outputs are saved under results/ in corresponding subdirectories.
###############################################################################

set -e
cd "$(dirname "$0")/.."   # cd to project root

DEVICE="cuda"              # change to "cpu" if no GPU
SEEDS="42 2023 2024"       # 3 seeds for mean ± std
DATA_ROOT="./data"

echo "============================================================"
echo "  StructRouter Experiment Pipeline"
echo "  Device: $DEVICE"
echo "  Seeds:  $SEEDS"
echo "============================================================"

###############################################################################
# 0. SANITY CHECK — verify that the code runs
###############################################################################
echo ""
echo ">>> [0] Sanity check: convergence tests"
python scripts/test_convergence.py

echo ""
echo ">>> [0] Sanity check: efficiency analysis"
python scripts/eval_efficiency.py --device "$DEVICE"

###############################################################################
# 1. MAIN TABLE — Long-term Time Series Forecasting (Table 1 in paper)
#
#    7 datasets × 4 pred_lens × 3 seeds = 84 runs
#    This is the primary result table.
###############################################################################
echo ""
echo "============================================================"
echo "  [1] Main Experiments (7 datasets × 4 pred_lens × 3 seeds)"
echo "============================================================"

DATASETS="ETTh1 ETTh2 ETTm1 ETTm2 Weather ECL Traffic"
PRED_LENS="96 192 336 720"

for SEED in $SEEDS; do
for DS in $DATASETS; do
  # ETTh uses seq_len=96, others use 336
  SEQ_LEN=336
  if [[ "$DS" == ETTh* ]]; then
    SEQ_LEN=96
  fi

  for PL in $PRED_LENS; do
    echo ">>> Main: $DS, pred_len=$PL, seed=$SEED"
    python scripts/train_three_stage.py \
      --dataset "$DS" \
      --data_root "$DATA_ROOT" \
      --seq_len "$SEQ_LEN" \
      --pred_len "$PL" \
      --device "$DEVICE" \
      --seed "$SEED" \
      --output_dir "results/main/seed${SEED}" \
      2>&1 | tail -3
  done
done
done

###############################################################################
# 2. ABLATION STUDY — Component contribution (Table 2 in paper)
#
#    7 variants × 2 datasets × 1 pred_len × 3 seeds = 42 runs
###############################################################################
echo ""
echo "============================================================"
echo "  [2] Ablation Study"
echo "============================================================"

ABL_DATASETS="ETTh1 Weather"
ABL_PL=96

for SEED in $SEEDS; do
for DS in $ABL_DATASETS; do
  SEQ_LEN=336
  if [[ "$DS" == ETTh* ]]; then SEQ_LEN=96; fi

  echo ">>> Ablation: Full model on $DS"
  python scripts/train_three_stage.py \
    --dataset "$DS" --seq_len "$SEQ_LEN" --pred_len $ABL_PL \
    --device "$DEVICE" --seed "$SEED" \
    --output_dir "results/ablation/full/seed${SEED}" \
    2>&1 | tail -2

  echo ">>> Ablation: w/o Segment on $DS"
  python scripts/train_three_stage.py \
    --dataset "$DS" --seq_len "$SEQ_LEN" --pred_len $ABL_PL \
    --no_segment \
    --device "$DEVICE" --seed "$SEED" \
    --output_dir "results/ablation/no_segment/seed${SEED}" \
    2>&1 | tail -2

  echo ">>> Ablation: w/o Causal on $DS"
  python scripts/train_three_stage.py \
    --dataset "$DS" --seq_len "$SEQ_LEN" --pred_len $ABL_PL \
    --no_causal \
    --device "$DEVICE" --seed "$SEED" \
    --output_dir "results/ablation/no_causal/seed${SEED}" \
    2>&1 | tail -2

  echo ">>> Ablation: w/o Verification on $DS"
  python scripts/train_three_stage.py \
    --dataset "$DS" --seq_len "$SEQ_LEN" --pred_len $ABL_PL \
    --no_verification \
    --device "$DEVICE" --seed "$SEED" \
    --output_dir "results/ablation/no_verification/seed${SEED}" \
    2>&1 | tail -2

  echo ">>> Ablation: w/o RFT on $DS"
  python scripts/train_three_stage.py \
    --dataset "$DS" --seq_len "$SEQ_LEN" --pred_len $ABL_PL \
    --no_rft \
    --device "$DEVICE" --seed "$SEED" \
    --output_dir "results/ablation/no_rft/seed${SEED}" \
    2>&1 | tail -2

  echo ">>> Ablation: Hard Routing on $DS"
  python scripts/train_three_stage.py \
    --dataset "$DS" --seq_len "$SEQ_LEN" --pred_len $ABL_PL \
    --hard_routing \
    --device "$DEVICE" --seed "$SEED" \
    --output_dir "results/ablation/hard_routing/seed${SEED}" \
    2>&1 | tail -2

  echo ">>> Ablation: Single Expert on $DS"
  python scripts/train_three_stage.py \
    --dataset "$DS" --seq_len "$SEQ_LEN" --pred_len $ABL_PL \
    --num_experts 1 \
    --device "$DEVICE" --seed "$SEED" \
    --output_dir "results/ablation/single_expert/seed${SEED}" \
    2>&1 | tail -2
done
done

###############################################################################
# 3. HYPERPARAMETER SENSITIVITY (Figure / Table in paper)
#
#    Key params: alpha, gamma, hidden_dim, num_experts, dag_warmup_frac
#    On ETTh1, pred_len=96, quick mode for speed
###############################################################################
echo ""
echo "============================================================"
echo "  [3] Hyperparameter Sensitivity Analysis"
echo "============================================================"

python scripts/hyperparam_sensitivity.py \
  --dataset ETTh1 --pred_len 96 --quick \
  --params alpha gamma hidden_dim num_experts dag_warmup_frac

###############################################################################
# 4. EFFICIENCY COMPARISON (Table in paper)
#
#    Measure params, FLOPs, memory, inference time for each variant
###############################################################################
echo ""
echo "============================================================"
echo "  [4] Efficiency Analysis"
echo "============================================================"

# Different input dimensions to match different datasets
for DIM in 7 21 321 862; do
  echo ">>> Efficiency: input_dim=$DIM"
  python scripts/eval_efficiency.py \
    --input_dim "$DIM" \
    --device "$DEVICE" \
    2>&1 | tail -8
done

###############################################################################
# 5. CASE STUDY VISUALIZATION (Figures in paper)
#
#    Generate expert weight bars, causal graph heatmap, prediction plots,
#    prototype similarity matrix for 3 representative datasets.
###############################################################################
echo ""
echo "============================================================"
echo "  [5] Case Study Visualization"
echo "============================================================"

CASE_DATASETS="ETTh1 Weather Traffic"
for DS in $CASE_DATASETS; do
  # Use checkpoint if it exists, otherwise demo with random init
  CKPT="results/main/seed42/MATS-${DS}-96/model.pt"
  if [ -f "$CKPT" ]; then
    echo ">>> Case study: $DS (with trained model)"
    python scripts/visualize_case_study.py \
      --checkpoint "$CKPT" \
      --dataset "$DS" \
      --data_root "$DATA_ROOT" \
      --output_dir "results/case_study"
  else
    echo ">>> Case study: $DS (random init — run main experiments first)"
    python scripts/visualize_case_study.py \
      --dataset "$DS" \
      --data_root "$DATA_ROOT" \
      --output_dir "results/case_study"
  fi
done

###############################################################################
# 6. AGGREGATE RESULTS — Collect all results into summary tables
###############################################################################
echo ""
echo "============================================================"
echo "  [6] Aggregate Results"
echo "============================================================"

python -c "
import json, glob, os
from collections import defaultdict
import numpy as np

# Main results
print('\n=== MAIN RESULTS ===')
print(f'{\"Dataset\":<12} {\"PredLen\":<8} {\"MSE (mean±std)\":<20} {\"MAE (mean±std)\":<20}')
print('-' * 60)

results = defaultdict(list)
for f in sorted(glob.glob('results/main/seed*/MATS-*/results.json')):
    with open(f) as fp:
        r = json.load(fp)
    key = (r['dataset'], r['pred_len'])
    results[key].append(r)

for (ds, pl), runs in sorted(results.items()):
    mses = [r['mse'] for r in runs]
    maes = [r['mae'] for r in runs]
    print(f'{ds:<12} {pl:<8} {np.mean(mses):.4f}±{np.std(mses):.4f}     {np.mean(maes):.4f}±{np.std(maes):.4f}')

# Ablation results
print('\n=== ABLATION RESULTS ===')
abl_results = defaultdict(list)
for f in sorted(glob.glob('results/ablation/*/seed*/MATS-*/results.json')):
    parts = f.split('/')
    variant = parts[2]  # e.g. 'no_segment'
    with open(f) as fp:
        r = json.load(fp)
    key = (variant, r['dataset'])
    abl_results[key].append(r)

print(f'{\"Variant\":<20} {\"Dataset\":<12} {\"MSE\":<12} {\"MAE\":<12}')
print('-' * 56)
for (var, ds), runs in sorted(abl_results.items()):
    mse = np.mean([r['mse'] for r in runs])
    mae = np.mean([r['mae'] for r in runs])
    print(f'{var:<20} {ds:<12} {mse:.4f}       {mae:.4f}')
"

###############################################################################
echo ""
echo "============================================================"
echo "  ALL EXPERIMENTS COMPLETE"
echo "  Results saved in results/"
echo "============================================================"

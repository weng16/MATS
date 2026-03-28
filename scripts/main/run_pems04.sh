#!/bin/bash
###############################################################################
#  MATS — PEMS04 Dataset Experiments
#  Input: 96 steps, Output: 96/192/336/720 steps
#  GPU: 0
###############################################################################

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

export CUDA_VISIBLE_DEVICES=0

DATASET="pems04"
DATASET_UPPER="PEMS04"
INPUT_LEN=96
PRED_LENS=(96 192 336 720)

mkdir -p logs/${DATASET}

echo "============================================================"
echo "  MATS — ${DATASET_UPPER} Experiments (GPU 0)"
echo "  Time:      $(date '+%Y-%m-%d %H:%M:%S')"
echo "  Input len: ${INPUT_LEN}"
echo "  Pred lens: ${PRED_LENS[*]}"
echo "============================================================"

for pl in "${PRED_LENS[@]}"; do
    LOG_FILE="logs/${DATASET}/${DATASET}_${INPUT_LEN}_${pl}.log"
    
    echo ""
    echo "[$(date '+%H:%M:%S')] >>> ${DATASET_UPPER} input=${INPUT_LEN} pred=${pl}"
    
    /mnt/data/conda_env/mats/bin/python scripts/train.py \
        --config-name="main/${DATASET}_${pl}" \
        input_len=${INPUT_LEN} \
        2>&1 | tee "${LOG_FILE}"
    
    echo "[$(date '+%H:%M:%S')] <<< Done. Log: ${LOG_FILE}"
done

echo ""
echo "============================================================"
echo "  ${DATASET_UPPER} ALL DONE"
echo "============================================================"

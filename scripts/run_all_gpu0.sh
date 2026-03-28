#!/bin/bash
# GPU 0: ETT系列 + Traffic (串行执行各数据集，每个数据集内4个预测步长并行)
export CUDA_VISIBLE_DEVICES=0
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"
EXTRA_ARGS=("$@")

echo "========== GPU 0 START $(date) =========="

echo "[$(date '+%H:%M:%S')] Running ETTh1..."
bash "$SCRIPT_DIR/datasets/run_etth1.sh" "${EXTRA_ARGS[@]}"

echo "[$(date '+%H:%M:%S')] Running ETTh2..."
bash "$SCRIPT_DIR/datasets/run_etth2.sh" "${EXTRA_ARGS[@]}"

echo "[$(date '+%H:%M:%S')] Running ETTm1..."
bash "$SCRIPT_DIR/datasets/run_ettm1.sh" "${EXTRA_ARGS[@]}"

echo "[$(date '+%H:%M:%S')] Running ETTm2..."
bash "$SCRIPT_DIR/datasets/run_ettm2.sh" "${EXTRA_ARGS[@]}"

echo "[$(date '+%H:%M:%S')] Running Traffic..."
bash "$SCRIPT_DIR/datasets/run_traffic.sh" "${EXTRA_ARGS[@]}"

echo "========== GPU 0 ALL DONE $(date) =========="

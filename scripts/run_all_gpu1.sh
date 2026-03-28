#!/bin/bash
# GPU 1: Weather, ECL, Electricity, PEMS系列 (串行执行各数据集，每个数据集内4个预测步长并行)
export CUDA_VISIBLE_DEVICES=1
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"
EXTRA_ARGS=("$@")

echo "========== GPU 1 START $(date) =========="

echo "[$(date '+%H:%M:%S')] Running Weather..."
bash "$SCRIPT_DIR/datasets/run_weather.sh" "${EXTRA_ARGS[@]}"

echo "[$(date '+%H:%M:%S')] Running ECL..."
bash "$SCRIPT_DIR/datasets/run_ecl.sh" "${EXTRA_ARGS[@]}"

echo "[$(date '+%H:%M:%S')] Running Electricity..."
bash "$SCRIPT_DIR/datasets/run_electricity.sh" "${EXTRA_ARGS[@]}"

echo "[$(date '+%H:%M:%S')] Running PEMS03..."
bash "$SCRIPT_DIR/datasets/run_pems03.sh" "${EXTRA_ARGS[@]}"

echo "[$(date '+%H:%M:%S')] Running PEMS04..."
bash "$SCRIPT_DIR/datasets/run_pems04.sh" "${EXTRA_ARGS[@]}"

echo "[$(date '+%H:%M:%S')] Running PEMS07..."
bash "$SCRIPT_DIR/datasets/run_pems07.sh" "${EXTRA_ARGS[@]}"

echo "[$(date '+%H:%M:%S')] Running PEMS08..."
bash "$SCRIPT_DIR/datasets/run_pems08.sh" "${EXTRA_ARGS[@]}"

echo "========== GPU 1 ALL DONE $(date) =========="

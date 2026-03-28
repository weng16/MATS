#!/bin/bash
# ETTm1: 4个预测步长并行
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"
mkdir -p logs/ettm1
EXTRA_ARGS=("$@")

/mnt/data/conda_env/mats/bin/python scripts/train.py --config-name=main/ettm1_96 input_len=96 "${EXTRA_ARGS[@]}" > logs/ettm1/ettm1_96_96.log 2>&1 &
/mnt/data/conda_env/mats/bin/python scripts/train.py --config-name=main/ettm1_192 input_len=96 "${EXTRA_ARGS[@]}" > logs/ettm1/ettm1_96_192.log 2>&1 &
/mnt/data/conda_env/mats/bin/python scripts/train.py --config-name=main/ettm1_336 input_len=96 "${EXTRA_ARGS[@]}" > logs/ettm1/ettm1_96_336.log 2>&1 &
/mnt/data/conda_env/mats/bin/python scripts/train.py --config-name=main/ettm1_720 input_len=96 "${EXTRA_ARGS[@]}" > logs/ettm1/ettm1_96_720.log 2>&1 &

wait
echo "ETTm1 done"

#!/bin/bash
# PEMS03: 4个预测步长并行
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"
mkdir -p logs/pems03
EXTRA_ARGS=("$@")

/mnt/data/conda_env/mats/bin/python scripts/train.py --config-name=main/pems03_96 input_len=96 "${EXTRA_ARGS[@]}" > logs/pems03/pems03_96_96.log 2>&1 &
/mnt/data/conda_env/mats/bin/python scripts/train.py --config-name=main/pems03_192 input_len=96 "${EXTRA_ARGS[@]}" > logs/pems03/pems03_96_192.log 2>&1 &
/mnt/data/conda_env/mats/bin/python scripts/train.py --config-name=main/pems03_336 input_len=96 "${EXTRA_ARGS[@]}" > logs/pems03/pems03_96_336.log 2>&1 &
/mnt/data/conda_env/mats/bin/python scripts/train.py --config-name=main/pems03_720 input_len=96 "${EXTRA_ARGS[@]}" > logs/pems03/pems03_96_720.log 2>&1 &

wait
echo "PEMS03 done"

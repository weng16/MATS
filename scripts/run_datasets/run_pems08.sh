#!/bin/bash
# PEMS08: 4个预测步长并行
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"
mkdir -p logs/pems08
EXTRA_ARGS=("$@")

/mnt/data/conda_env/mats/bin/python scripts/train.py --config-name=main/pems08_96 input_len=96 "${EXTRA_ARGS[@]}" > logs/pems08/pems08_96_96.log 2>&1 &
/mnt/data/conda_env/mats/bin/python scripts/train.py --config-name=main/pems08_192 input_len=96 "${EXTRA_ARGS[@]}" > logs/pems08/pems08_96_192.log 2>&1 &
/mnt/data/conda_env/mats/bin/python scripts/train.py --config-name=main/pems08_336 input_len=96 "${EXTRA_ARGS[@]}" > logs/pems08/pems08_96_336.log 2>&1 &
/mnt/data/conda_env/mats/bin/python scripts/train.py --config-name=main/pems08_720 input_len=96 "${EXTRA_ARGS[@]}" > logs/pems08/pems08_96_720.log 2>&1 &

wait
echo "PEMS08 done"

#!/bin/bash
# ECL: 4个预测步长并行
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"
mkdir -p logs/ecl
EXTRA_ARGS=("$@")

/mnt/data/conda_env/mats/bin/python scripts/train.py --config-name=main/ecl_96 input_len=96 "${EXTRA_ARGS[@]}" > logs/ecl/ecl_96_96.log 2>&1 &
/mnt/data/conda_env/mats/bin/python scripts/train.py --config-name=main/ecl_192 input_len=96 "${EXTRA_ARGS[@]}" > logs/ecl/ecl_96_192.log 2>&1 &
/mnt/data/conda_env/mats/bin/python scripts/train.py --config-name=main/ecl_336 input_len=96 "${EXTRA_ARGS[@]}" > logs/ecl/ecl_96_336.log 2>&1 &
/mnt/data/conda_env/mats/bin/python scripts/train.py --config-name=main/ecl_720 input_len=96 "${EXTRA_ARGS[@]}" > logs/ecl/ecl_96_720.log 2>&1 &

wait
echo "ECL done"

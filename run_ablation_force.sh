#!/bin/bash
# 终极强力清理脚本

clean_cache() {
    echo "清理缓存中..."
    rm -rf ~/.basicts
    rm -rf ~/.easytorch
    find . -name "*.pkl" -delete
    find . -name "*.pt" -delete
    find data -name "*.pkl" -delete
    find data -name "*.pt" -delete
    echo "缓存清理完毕。"
}

# 移除了已经跑完的 hard_routing/etth1_96
configs=(
    "ablation/hard_routing/ettm1_96"
    "ablation/single_expert/etth1_96"
    "ablation/single_expert/ettm1_96"
)

for cfg in "${configs[@]}"; do
    echo "========================================="
    echo "开始运行: $cfg"
    echo "========================================="
    clean_cache
    python scripts/train.py --config-name="$cfg"
done

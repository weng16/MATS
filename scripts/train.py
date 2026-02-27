#!/usr/bin/env python3
"""
统一训练入口（Hydra 驱动）
用法:
    python scripts/train.py --config-name main/etth1_96
"""
import hydra
from omegaconf import DictConfig
import basicts
# 注册模型
from src.basicts_adapter.mats_arch import MATSArch

@hydra.main(version_base=None, config_path="../configs", config_name=None)
def main(cfg: DictConfig):
    # 根据 experiment.type 自动决定输出根目录
    if cfg.experiment.type == "main":
        root = "results/main"
    else:
        root = f"results/ablation/{cfg.experiment.ablation_type}"
    basicts.run(cfg, default_root_dir=root)

if __name__ == "__main__":
    main()
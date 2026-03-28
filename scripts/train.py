#!/usr/bin/env python3
import sys
import os
from pathlib import Path

os.environ["TQDM_DISABLE"] = "1"

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import hydra
from omegaconf import DictConfig, OmegaConf
import basicts
import importlib
from basicts.configs import BasicTSForecastingConfig
from src.basicts_adapter.mats_arch import MATSArch, GlobalZScoreScaler

def resolve_class(path: str):
    """把 'torch.optim.lr_scheduler.CosineAnnealingLR' 解析成 class"""
    module_path, class_name = path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)

@hydra.main(version_base=None, config_path="../configs", config_name="main/etth1_96")
def main(cfg: DictConfig):
    OmegaConf.set_struct(cfg, False)
    
    # 决定输出目录
    if cfg.experiment.type == "main":
        root = "results/main"
    else:
        root = f"results/ablation/{cfg.experiment.ablation_type}"

    # 把 OmegaConf 转成普通 dict
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    
    # 去掉 BasicTSForecastingConfig 不认识的字段
    cfg_dict.pop("experiment", None)
    cfg_dict.pop("input_dim", None)
    cfg_dict.pop("output_dim", None)
    data_path = cfg_dict.pop("data_path", None)

    # model 字段需要是 class 对象
    cfg_dict["model"] = MATSArch
    
    # dataset_params 补充预测长度和数据路径
    cfg_dict.setdefault("dataset_params", {})
    cfg_dict["gpus"] = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    input_len = cfg_dict.pop("input_len", 336)
    output_len = cfg_dict.pop("output_len", 96)
    cfg_dict["dataset_params"]["input_len"] = input_len
    cfg_dict["dataset_params"]["output_len"] = output_len

    # model_params -> model_config: sync seq_len/pred_len with actual data lengths
    from basicts.configs import BasicTSModelConfig
    model_params = cfg_dict.pop("model_params", {})
    model_params["seq_len"] = input_len
    model_params["pred_len"] = output_len
    cfg_dict["model_config"] = BasicTSModelConfig(model_params)
    
    # 设置数据路径 (使用绝对路径)
    project_root = Path(__file__).resolve().parent.parent
    dataset_name = cfg_dict.get("dataset_name", "ETTh1")
    if data_path:
        # data_path 可能是相对路径，比如 ./data/ETTm1/ETTm1.csv
        p = Path(data_path)
        if not p.is_absolute():
            p = project_root / p
        cfg_dict["data_file_path"] = str(p.parent)
    else:
        cfg_dict["data_file_path"] = str(project_root / "data" / dataset_name)

    # Resolve lr_scheduler and optimizer classes
    if isinstance(cfg_dict.get("lr_scheduler"), str):
        cfg_dict["lr_scheduler"] = resolve_class(cfg_dict["lr_scheduler"])
    if isinstance(cfg_dict.get("optimizer"), str):
        cfg_dict["optimizer"] = resolve_class(cfg_dict["optimizer"])

    # Fix the global scaler bug by injecting our custom scaler
    if not cfg_dict.get("norm_each_channel", True):
        cfg_dict["scaler"] = GlobalZScoreScaler

    ts_cfg = BasicTSForecastingConfig(**cfg_dict)
    
    project_root = Path(__file__).resolve().parent.parent
    ts_cfg.ckpt_save_dir = str(project_root / root / cfg.experiment.name)
    
    basicts.BasicTSLauncher.launch_training(ts_cfg)

if __name__ == "__main__":  
    main()
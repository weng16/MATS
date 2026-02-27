#!/usr/bin/env python3
"""
生成 BasicTS 配置文件模板
用法：python scripts/gen_basicts_cfg.py --dataset ETTh1 --pred_len 96
"""
import argparse
from pathlib import Path
import textwrap

TEMPLATE = textwrap.dedent("""
from easydict import EasyDict
from basicts.metrics import masked_mae, masked_mse
from src.basicts_adapter.mats_arch import MATSArch

CFG = EasyDict()

# ---- 通用 ----
CFG.DESCRIPTION = "MATS on {dataset}, pred_len={pred_len}"
CFG.RUNNER = "basicts.runners.SimpleTimeSeriesForecastingRunner"
CFG.DATASET_CLS = "basicts.data.TimeSeriesForecastingDataset"
CFG.DATASET_NAME = "{dataset}"
CFG.DATASET_TYPE = "long_term_forecast"
CFG.DATASET_INPUT_LEN  = 96
CFG.DATASET_OUTPUT_LEN = {pred_len}

# ---- 模型 ----
CFG.MODEL = EasyDict()
CFG.MODEL.NAME = "MATSArch"
CFG.MODEL.ARCH = MATSArch
CFG.MODEL.PARAM = dict(
    input_dim={num_var},
    output_dim={num_var},
    hidden_dim=256,
    seq_len=96,
    pred_len={pred_len},
    use_segment_processing=True,
    use_verification=True,
    use_rft=True,
    task_type='forecast',
)
CFG.MODEL.FORWARD_FEATURES = [0]   # 使用第0个特征通道
CFG.MODEL.TARGET_FEATURES  = [0]

# ---- 损失 & 优化 ----
CFG.TRAIN = EasyDict()
CFG.TRAIN.LOSS = masked_mae
CFG.TRAIN.OPTIM = EasyDict(TYPE="Adam", PARAM=dict(lr=5e-4, weight_decay=0.01))
CFG.TRAIN.LR_SCHEDULER = EasyDict(TYPE="CosineAnnealingLR", PARAM=dict(T_max=100))
CFG.TRAIN.NUM_EPOCHS = 100
CFG.TRAIN.CKPT_SAVE_DIR = "./checkpoints/MATS_{dataset}_{pred_len}"

# ---- 指标 ----
CFG.METRICS = dict(MSE=masked_mse, MAE=masked_mae)
CFG.METRICS_BEST = "MSE"

# ---- 数据加载 ----
CFG.TRAIN.DATA = EasyDict(BATCH_SIZE=32, SHUFFLE=True)
CFG.VAL.DATA   = EasyDict(BATCH_SIZE=32)
CFG.TEST.DATA  = EasyDict(BATCH_SIZE=32)
""")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="ETTh1")
    parser.add_argument("--pred_len", type=int, default=96)
    parser.add_argument("--num_var", type=int, default=7)
    args = parser.parse_args()
    cfg = TEMPLATE.format(dataset=args.dataset, pred_len=args.pred_len, num_var=args.num_var)
    out = Path(f"basicts_cfg/mats_{args.dataset}_{args.pred_len}.py")
    out.parent.mkdir(exist_ok=True)
    out.write_text(cfg)
    print(f"Config written to {out}")

if __name__ == "__main__":
    main()
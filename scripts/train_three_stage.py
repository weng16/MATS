#!/usr/bin/env python3
"""
Standalone three-stage training using StructRouter + ThreeStageTrainer.

This is the *primary* training path that exercises the full innovation pipeline:
  Stage 1: Self-supervised pretraining (contrastive + mask + structure pred)
  Stage 2: End-to-end fine-tuning with JointLoss (per-module LR, DAG warmup)
  Stage 3: RFT with PPO (freeze backbone, train router + tool adapter)

Usage:
    # Default: ETTh1, pred_len=96
    python scripts/train_three_stage.py

    # Specify dataset / pred_len
    python scripts/train_three_stage.py --dataset Weather --pred_len 336

    # Ablation: disable certain components
    python scripts/train_three_stage.py --dataset ETTh1 --pred_len 96 --no_segment
    python scripts/train_three_stage.py --dataset ETTh1 --pred_len 96 --no_verification
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import json
import logging
import time
import torch
import numpy as np

from src.models.struct_router import StructRouter
from src.losses.joint_loss import JointLoss
from src.trainers.three_stage_trainer import ThreeStageTrainer
from src.data.dataset import create_dataloaders, DATASET_CONFIG

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

DATASET_DIM = {
    "ETTh1": 7, "ETTh2": 7, "ETTm1": 7, "ETTm2": 7,
    "Weather": 21, "ECL": 321, "Traffic": 862,
    "Electricity": 321, "Exchange": 8, "ILI": 7,
}


def parse_args():
    p = argparse.ArgumentParser(description="StructRouter three-stage training")
    # Data
    p.add_argument("--dataset", type=str, default="ETTh1")
    p.add_argument("--data_root", type=str, default="./data")
    p.add_argument("--seq_len", type=int, default=336)
    p.add_argument("--pred_len", type=int, default=96)
    p.add_argument("--features", type=str, default="M")
    p.add_argument("--batch_size", type=int, default=32)
    # Model
    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--use_revin", action="store_true", default=True)
    p.add_argument("--no_revin", dest="use_revin", action="store_false")
    p.add_argument("--channel_independent", action="store_true", default=False)
    p.add_argument("--no_segment", action="store_true", default=False)
    p.add_argument("--no_verification", action="store_true", default=False)
    p.add_argument("--no_rft", action="store_true", default=False)
    p.add_argument("--no_causal", action="store_true", default=False)
    p.add_argument("--hard_routing", action="store_true", default=False)
    p.add_argument("--num_experts", type=int, default=5)
    # Training
    p.add_argument("--stage1_epochs", type=int, default=30)
    p.add_argument("--stage2_epochs", type=int, default=100)
    p.add_argument("--stage3_epochs", type=int, default=20)
    p.add_argument("--stage1_lr", type=float, default=1e-3)
    p.add_argument("--stage2_lr", type=float, default=5e-4)
    p.add_argument("--stage3_lr", type=float, default=1e-4)
    p.add_argument("--warmup_epochs", type=int, default=10)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--prototype_lr_scale", type=float, default=0.1)
    p.add_argument("--wcomm_lr_scale", type=float, default=3.0)
    # Loss
    p.add_argument("--alpha", type=float, default=0.1)
    p.add_argument("--beta", type=float, default=0.01)
    p.add_argument("--gamma", type=float, default=0.1)
    p.add_argument("--delta", type=float, default=0.01)
    p.add_argument("--epsilon", type=float, default=0.01)
    p.add_argument("--dag_warmup_frac", type=float, default=0.2)
    # Output
    p.add_argument("--output_dir", type=str, default="results")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    args = parse_args()
    set_seed(args.seed)

    dim = DATASET_DIM.get(args.dataset, 7)
    exp_name = f"MATS-{args.dataset}-{args.pred_len}"
    if args.no_segment:
        exp_name += "-noSeg"
    if args.no_verification:
        exp_name += "-noVerif"
    if args.no_rft:
        exp_name += "-noRFT"
    if args.no_causal:
        exp_name += "-noCausal"
    if args.hard_routing:
        exp_name += "-hardRoute"
    if args.num_experts == 1:
        exp_name += "-singleExpert"

    logger.info(f"Experiment: {exp_name}")
    logger.info(f"Dataset: {args.dataset}, seq_len={args.seq_len}, pred_len={args.pred_len}, dim={dim}")

    # --- Data ---
    train_loader, val_loader, test_loader, scaler, info = create_dataloaders(
        root_path=args.data_root,
        dataset_name=args.dataset,
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        batch_size=args.batch_size,
    )
    logger.info(f"Data: train={info['train_samples']}, val={info['val_samples']}, test={info['test_samples']}")

    # --- Model ---
    model = StructRouter(
        input_dim=dim,
        output_dim=dim,
        hidden_dim=args.hidden_dim,
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        num_experts=args.num_experts,
        num_agents=args.num_experts,
        use_segment_processing=not args.no_segment,
        use_verification=not args.no_verification,
        use_rft=not args.no_rft,
        use_revin=args.use_revin,
        channel_independent=args.channel_independent,
    ).to(args.device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model: {total_params:,} total params, {trainable_params:,} trainable")

    # --- Loss ---
    dag_warmup_steps = int(args.dag_warmup_frac * args.stage2_epochs * len(train_loader))
    joint_loss = JointLoss(
        alpha=args.alpha,
        beta=args.beta,
        gamma=args.gamma,
        delta=args.delta,
        epsilon=args.epsilon,
        dag_warmup_steps=dag_warmup_steps,
    ).to(args.device)

    # --- Trainer ---
    config = {
        "stage1_epochs": args.stage1_epochs,
        "stage2_epochs": args.stage2_epochs,
        "stage3_epochs": args.stage3_epochs,
        "stage1_lr": args.stage1_lr,
        "stage2_lr": args.stage2_lr,
        "stage3_lr": args.stage3_lr,
        "warmup_epochs": args.warmup_epochs,
        "max_grad_norm": args.max_grad_norm,
        "prototype_lr_scale": args.prototype_lr_scale,
        "wcomm_lr_scale": args.wcomm_lr_scale,
    }
    trainer = ThreeStageTrainer(model, joint_loss, config)

    # --- Train ---
    t0 = time.time()
    history = trainer.train(train_loader, val_loader)
    train_time = time.time() - t0
    logger.info(f"Training completed in {train_time:.1f}s")

    # --- Evaluate ---
    metrics = trainer.evaluate(test_loader)
    logger.info(f"Test results: MSE={metrics['mse']:.6f}, MAE={metrics['mae']:.6f}")

    # --- Save ---
    save_dir = Path(args.output_dir) / exp_name
    save_dir.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), save_dir / "model.pt")
    results = {
        "experiment": exp_name,
        "dataset": args.dataset,
        "seq_len": args.seq_len,
        "pred_len": args.pred_len,
        "mse": metrics["mse"],
        "mae": metrics["mae"],
        "rmse": metrics["rmse"],
        "total_params": total_params,
        "train_time_sec": train_time,
        "args": vars(args),
    }
    with open(save_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Saved to {save_dir}")
    return metrics


if __name__ == "__main__":
    main()

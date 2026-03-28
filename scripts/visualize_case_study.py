#!/usr/bin/env python3
"""
Case study visualization: expert weights, causal graphs, and predictions.

Loads a trained model checkpoint and generates analysis figures.

Usage:
    python scripts/visualize_case_study.py --checkpoint results/MATS-ETTh1-96/model.pt
    python scripts/visualize_case_study.py --checkpoint results/MATS-ETTh1-96/model.pt --dataset ETTh1
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import json
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from src.models.struct_router import StructRouter
from src.data.dataset import create_dataloaders, DATASET_CONFIG

EXPERT_NAMES = ["Periodic", "Trend", "Noise", "Abrupt", "General"]
EXPERT_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

DATASET_DIM = {
    "ETTh1": 7, "ETTh2": 7, "ETTm1": 7, "ETTm2": 7,
    "Weather": 21, "ECL": 321, "Traffic": 862,
}


def load_model(checkpoint_path, dataset, pred_len, seq_len, device):
    dim = DATASET_DIM.get(dataset, 7)

    # Try reading saved args from results.json next to the checkpoint
    results_path = Path(checkpoint_path).parent / "results.json"
    hidden_dim = 256
    if results_path.exists():
        with open(results_path) as f:
            saved = json.load(f)
        args_saved = saved.get("args", {})
        hidden_dim = args_saved.get("hidden_dim", 256)
        seq_len = args_saved.get("seq_len", seq_len)
        pred_len = args_saved.get("pred_len", pred_len)

    model = StructRouter(
        input_dim=dim, output_dim=dim, hidden_dim=hidden_dim,
        seq_len=seq_len, pred_len=pred_len,
        use_revin=True,
    ).to(device)
    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model, seq_len, pred_len


def plot_expert_weights(model, test_loader, save_dir, device, n_samples=3):
    """Visualize per-sample expert weights as stacked bars + time series."""
    fig, axes = plt.subplots(n_samples, 2, figsize=(14, 3 * n_samples),
                             gridspec_kw={"width_ratios": [3, 1]})
    if n_samples == 1:
        axes = axes.reshape(1, -1)

    with torch.no_grad():
        batch = next(iter(test_loader))
        x = batch["x"][:n_samples].to(device)
        output = model(x, return_details=True)
        weights = output["expert_weights"].cpu().numpy()

    for i in range(n_samples):
        # Left: time series
        ax = axes[i, 0]
        xi = x[i].cpu().numpy()
        for ch in range(min(xi.shape[1], 3)):
            ax.plot(xi[:, ch], alpha=0.7, linewidth=0.8)
        ax.set_title(f"Sample {i+1}", fontsize=10)
        ax.set_xlabel("Time step")

        # Right: stacked bar
        ax2 = axes[i, 1]
        bottom = 0
        for k, (name, color) in enumerate(zip(EXPERT_NAMES, EXPERT_COLORS)):
            ax2.barh(0, weights[i, k], left=bottom, color=color, label=name if i == 0 else "")
            bottom += weights[i, k]
        ax2.set_xlim(0, 1)
        ax2.set_yticks([])
        ax2.set_xlabel("Weight")

    if n_samples > 0:
        axes[0, 1].legend(loc="upper right", fontsize=7)

    plt.tight_layout()
    fig.savefig(save_dir / "expert_weights.pdf", dpi=150, bbox_inches="tight")
    fig.savefig(save_dir / "expert_weights.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved expert_weights.pdf/png")


def plot_causal_graph(model, save_dir):
    """Visualize the learned W_comm as a heatmap."""
    W = model.W_comm.detach().cpu().numpy()

    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        W, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
        xticklabels=EXPERT_NAMES, yticklabels=EXPERT_NAMES,
        ax=ax, square=True, linewidths=0.5,
    )
    ax.set_title("Learned Causal Communication Matrix $W_{comm}$")
    ax.set_xlabel("Receiver")
    ax.set_ylabel("Sender")

    plt.tight_layout()
    fig.savefig(save_dir / "causal_graph.pdf", dpi=150, bbox_inches="tight")
    fig.savefig(save_dir / "causal_graph.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved causal_graph.pdf/png")

    # DAG constraint value
    h = model.communication.dag_constraint().item()
    print(f"  DAG constraint h(W) = {h:.6f}")


def plot_prediction_comparison(model, test_loader, save_dir, device, n_vars=3):
    """Plot ground truth vs prediction for a few variables."""
    with torch.no_grad():
        batch = next(iter(test_loader))
        x = batch["x"][:1].to(device)
        y = batch["y"][:1].to(device)
        output = model(x, return_details=False)
        pred = output["prediction"][:1]

    x_np = x[0].cpu().numpy()
    y_np = y[0].cpu().numpy()
    pred_np = pred[0].cpu().numpy()

    fig, axes = plt.subplots(min(n_vars, y_np.shape[1]), 1,
                             figsize=(12, 3 * min(n_vars, y_np.shape[1])))
    if not hasattr(axes, "__len__"):
        axes = [axes]

    seq_len = x_np.shape[0]
    pred_len = y_np.shape[0]

    for i, ax in enumerate(axes):
        if i >= y_np.shape[1]:
            break
        ax.plot(range(seq_len), x_np[:, i], label="Input", color="gray", alpha=0.6)
        ax.plot(range(seq_len, seq_len + pred_len), y_np[:, i],
                label="Ground Truth", color="blue", linewidth=1.5)
        ax.plot(range(seq_len, seq_len + pred_len), pred_np[:, i],
                label="Prediction", color="red", linestyle="--", linewidth=1.5)
        ax.axvline(x=seq_len, color="black", linestyle=":", alpha=0.5)
        ax.set_ylabel(f"Var {i}")
        if i == 0:
            ax.legend(loc="upper right", fontsize=8)

    axes[-1].set_xlabel("Time step")
    fig.suptitle("Prediction vs Ground Truth", fontsize=12)
    plt.tight_layout()
    fig.savefig(save_dir / "prediction_comparison.pdf", dpi=150, bbox_inches="tight")
    fig.savefig(save_dir / "prediction_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved prediction_comparison.pdf/png")


def plot_prototype_similarity(model, save_dir):
    """Visualize prototype similarity matrix (should be near-identity if orthogonality reg works)."""
    P = model.weight_estimator.pattern_prototypes.detach().cpu()
    P_norm = F.normalize(P, dim=-1)
    gram = (P_norm @ P_norm.T).numpy()

    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        gram, annot=True, fmt=".2f", cmap="YlOrRd",
        xticklabels=EXPERT_NAMES, yticklabels=EXPERT_NAMES,
        ax=ax, square=True, vmin=0, vmax=1, linewidths=0.5,
    )
    ax.set_title("Prototype Gram Matrix $P_{norm}^T P_{norm}$")
    plt.tight_layout()
    fig.savefig(save_dir / "prototype_similarity.pdf", dpi=150, bbox_inches="tight")
    fig.savefig(save_dir / "prototype_similarity.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved prototype_similarity.pdf/png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=False,
                        default=None, help="Path to model checkpoint")
    parser.add_argument("--dataset", type=str, default="ETTh1")
    parser.add_argument("--seq_len", type=int, default=336)
    parser.add_argument("--pred_len", type=int, default=96)
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--output_dir", type=str, default="results/case_study")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    save_dir = Path(args.output_dir) / args.dataset
    save_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)

    if args.checkpoint and Path(args.checkpoint).exists():
        print(f"Loading checkpoint: {args.checkpoint}")
        model, args.seq_len, args.pred_len = load_model(
            args.checkpoint, args.dataset, args.pred_len, args.seq_len, device
        )
    else:
        print("No checkpoint provided or found. Using randomly initialized model for demo.")
        dim = DATASET_DIM.get(args.dataset, 7)
        model = StructRouter(
            input_dim=dim, output_dim=dim, hidden_dim=256,
            seq_len=args.seq_len, pred_len=args.pred_len,
            use_revin=True,
        ).to(device)
        model.eval()

    _, _, test_loader, _, _ = create_dataloaders(
        root_path=args.data_root,
        dataset_name=args.dataset,
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        batch_size=8,
    )

    print(f"\nGenerating visualizations for {args.dataset}...")
    print(f"Output: {save_dir}\n")

    plot_expert_weights(model, test_loader, save_dir, device)
    plot_causal_graph(model, save_dir)
    plot_prediction_comparison(model, test_loader, save_dir, device)
    plot_prototype_similarity(model, save_dir)

    print(f"\nAll visualizations saved to {save_dir}")


if __name__ == "__main__":
    main()

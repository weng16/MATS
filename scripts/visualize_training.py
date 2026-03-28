#!/usr/bin/env python3
"""Parse BasicTS training log and visualize training curves."""
import re
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

LOG_DIR = Path("results/main/MATS-Full-ETTh1-96")

def find_latest_log(log_dir: Path) -> Path:
    logs = sorted(log_dir.rglob("training_log_*.log"), key=lambda p: p.stat().st_mtime)
    return logs[-1] if logs else None

def parse_log(path: Path):
    records = {"train": [], "val": [], "test": []}
    pattern = re.compile(
        r"Result <(\w+)>:.*?"
        r"loss: ([\d.]+).*?"
        r"MAE: ([\d.]+).*?"
        r"MSE: ([\d.]+).*?"
        r"RMSE: ([\d.]+).*?"
        r"MAPE: ([\d.]+).*?"
        r"WAPE: ([\d.]+)"
    )
    lr_pattern = re.compile(r"lr: ([\d.e+-]+)")

    with open(path) as f:
        for line in f:
            m = pattern.search(line)
            if m:
                split = m.group(1)
                entry = {
                    "loss": float(m.group(2)),
                    "MAE":  float(m.group(3)),
                    "MSE":  float(m.group(4)),
                    "RMSE": float(m.group(5)),
                    "MAPE": float(m.group(6)),
                    "WAPE": float(m.group(7)),
                }
                lr_m = lr_pattern.search(line)
                if lr_m:
                    entry["lr"] = float(lr_m.group(1))
                records[split].append(entry)
    return records

def plot_curves(records, save_path: Path):
    n_epochs = len(records["train"])
    if n_epochs == 0:
        print("No training data found.")
        return
    epochs = np.arange(1, n_epochs + 1)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f"MATS Training Monitor — ETTh1 pred_len=96  ({n_epochs} epochs)", fontsize=14, fontweight="bold")

    metrics = ["MAE", "MSE", "RMSE"]
    for i, metric in enumerate(metrics):
        ax = axes[0, i]
        for split, color, marker in [("train", "#2196F3", "o"), ("val", "#FF9800", "s"), ("test", "#4CAF50", "^")]:
            if records[split]:
                vals = [r[metric] for r in records[split]]
                ax.plot(epochs[:len(vals)], vals, color=color, marker=marker, markersize=4, label=split, linewidth=1.5)
        ax.set_title(metric, fontsize=12)
        ax.set_xlabel("Epoch")
        ax.legend()
        ax.grid(True, alpha=0.3)

    # MAPE
    ax = axes[1, 0]
    for split, color, marker in [("train", "#2196F3", "o"), ("val", "#FF9800", "s"), ("test", "#4CAF50", "^")]:
        if records[split]:
            vals = [r["MAPE"] for r in records[split]]
            ax.plot(epochs[:len(vals)], vals, color=color, marker=marker, markersize=4, label=split, linewidth=1.5)
    ax.set_title("MAPE (%)", fontsize=12)
    ax.set_xlabel("Epoch")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Train-Val gap
    ax = axes[1, 1]
    if records["train"] and records["val"]:
        n = min(len(records["train"]), len(records["val"]))
        gap_mae = [records["val"][i]["MAE"] - records["train"][i]["MAE"] for i in range(n)]
        gap_mse = [records["val"][i]["MSE"] - records["train"][i]["MSE"] for i in range(n)]
        ax.plot(epochs[:n], gap_mae, color="#E91E63", marker="o", markersize=4, label="MAE gap (val-train)", linewidth=1.5)
        ax.plot(epochs[:n], gap_mse, color="#9C27B0", marker="s", markersize=4, label="MSE gap (val-train)", linewidth=1.5)
        ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.set_title("Generalization Gap", fontsize=12)
    ax.set_xlabel("Epoch")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # SOTA comparison
    ax = axes[1, 2]
    sota = {
        "PatchTST":     {"MSE": 0.370, "MAE": 0.400},
        "iTransformer": {"MSE": 0.386, "MAE": 0.405},
        "DLinear":      {"MSE": 0.375, "MAE": 0.399},
        "TimesNet":     {"MSE": 0.384, "MAE": 0.402},
        "FEDformer":    {"MSE": 0.376, "MAE": 0.419},
    }
    best_test = min(records["test"], key=lambda r: r["MSE"]) if records["test"] else None
    names = list(sota.keys()) + (["MATS (ours)"] if best_test else [])
    mse_vals = [v["MSE"] for v in sota.values()] + ([best_test["MSE"]] if best_test else [])
    mae_vals = [v["MAE"] for v in sota.values()] + ([best_test["MAE"]] if best_test else [])
    x = np.arange(len(names))
    w = 0.35
    colors_mse = ["#90CAF9"] * len(sota) + (["#F44336"] if best_test else [])
    colors_mae = ["#FFE0B2"] * len(sota) + (["#FF5722"] if best_test else [])
    ax.bar(x - w/2, mse_vals, w, color=colors_mse, edgecolor="gray", label="MSE")
    ax.bar(x + w/2, mae_vals, w, color=colors_mae, edgecolor="gray", label="MAE")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=9)
    ax.set_title("vs SOTA (ETTh1, pred=96)", fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved to {save_path}")

    # Print summary
    print("\n" + "=" * 70)
    print("  TRAINING SUMMARY")
    print("=" * 70)
    print(f"  Total epochs completed: {n_epochs}")
    print(f"  Total parameters: 11,340,625")
    print()
    if records["train"]:
        t = records["train"]
        print(f"  Train MAE:  {t[0]['MAE']:.4f} → {t[-1]['MAE']:.4f}  (Δ = {t[-1]['MAE'] - t[0]['MAE']:+.4f})")
        print(f"  Train MSE:  {t[0]['MSE']:.4f} → {t[-1]['MSE']:.4f}  (Δ = {t[-1]['MSE'] - t[0]['MSE']:+.4f})")
    if records["val"]:
        v = records["val"]
        print(f"  Val   MAE:  {v[0]['MAE']:.4f} → {v[-1]['MAE']:.4f}  (Δ = {v[-1]['MAE'] - v[0]['MAE']:+.4f})")
    if best_test:
        print(f"  Best test:  MSE={best_test['MSE']:.4f}, MAE={best_test['MAE']:.4f}")
    print()
    print("  ISSUES DETECTED:")
    if records["train"] and abs(records["train"][-1]["MAE"] - records["train"][0]["MAE"]) < 0.01:
        print("  [!] Train loss is FLAT — model is barely learning")
    if records["train"] and records["val"]:
        gap = records["val"][-1]["MAE"] - records["train"][-1]["MAE"]
        if gap > 0.15:
            print(f"  [!] Large generalization gap: val-train MAE = {gap:.4f}")
    if best_test and best_test["MSE"] > 0.40:
        print(f"  [!] Test MSE ({best_test['MSE']:.4f}) is far from SOTA (~0.370)")
    print("=" * 70)


if __name__ == "__main__":
    log_path = find_latest_log(LOG_DIR)
    if log_path is None:
        print(f"No log files found in {LOG_DIR}")
        sys.exit(1)
    print(f"Parsing: {log_path}")
    records = parse_log(log_path)
    save_path = Path("results/main/MATS-Full-ETTh1-96/training_curves.png")
    plot_curves(records, save_path)

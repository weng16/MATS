#!/usr/bin/env python3
"""
Hyperparameter sensitivity analysis.

Sweeps over key hyperparameters one at a time (with all others at default)
and records MSE/MAE on ETTh1 pred_len=96.

Usage:
    python scripts/hyperparam_sensitivity.py
    python scripts/hyperparam_sensitivity.py --dataset Weather --pred_len 96
    python scripts/hyperparam_sensitivity.py --quick   # fewer epochs for fast sweep
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import json
import subprocess
import itertools

SWEEP_CONFIGS = {
    "hidden_dim": [64, 128, 256, 512],
    "alpha": [0.01, 0.05, 0.1, 0.2, 0.5],
    "beta": [0.001, 0.005, 0.01, 0.05, 0.1],
    "gamma": [0.01, 0.05, 0.1, 0.2, 0.5],
    "delta": [0.001, 0.005, 0.01, 0.05, 0.1],
    "epsilon": [0.001, 0.005, 0.01, 0.05, 0.1],
    "stage2_lr": [1e-4, 3e-4, 5e-4, 1e-3, 2e-3],
    "prototype_lr_scale": [0.01, 0.05, 0.1, 0.3, 0.5],
    "wcomm_lr_scale": [1.0, 2.0, 3.0, 5.0, 10.0],
    "dag_warmup_frac": [0.0, 0.1, 0.2, 0.3, 0.5],
    "num_experts": [1, 3, 5, 7],
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="ETTh1")
    parser.add_argument("--pred_len", type=int, default=96)
    parser.add_argument("--quick", action="store_true",
                        help="Use fewer epochs for fast sweep")
    parser.add_argument("--params", nargs="+", default=None,
                        help="Which params to sweep (default: all)")
    args = parser.parse_args()

    epoch_args = ["--stage1_epochs", "5", "--stage2_epochs", "20", "--stage3_epochs", "5"] if args.quick else []

    params_to_sweep = args.params or list(SWEEP_CONFIGS.keys())
    all_results = {}

    for param_name in params_to_sweep:
        if param_name not in SWEEP_CONFIGS:
            print(f"Unknown param: {param_name}, skipping")
            continue

        values = SWEEP_CONFIGS[param_name]
        print(f"\n{'='*60}")
        print(f"Sweeping {param_name}: {values}")
        print(f"{'='*60}")

        param_results = []
        for val in values:
            exp_name = f"sweep-{param_name}-{val}"
            cmd = [
                "python", "scripts/train_three_stage.py",
                "--dataset", args.dataset,
                "--pred_len", str(args.pred_len),
                f"--{param_name}", str(val),
                "--output_dir", f"results/hyperparam/{param_name}",
            ] + epoch_args

            print(f"\n>>> {exp_name}: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True)

            # Try to read result from saved JSON
            result_path = Path(f"results/hyperparam/{param_name}") / f"MATS-{args.dataset}-{args.pred_len}" / "results.json"
            if result_path.exists():
                with open(result_path) as f:
                    r = json.load(f)
                param_results.append({
                    "value": val,
                    "mse": r.get("mse"),
                    "mae": r.get("mae"),
                })
                print(f"    MSE={r.get('mse', 'N/A'):.6f}, MAE={r.get('mae', 'N/A'):.6f}")
            else:
                print(f"    FAILED — no results found")
                if result.stderr:
                    print(f"    stderr: {result.stderr[-200:]}")
                param_results.append({"value": val, "mse": None, "mae": None})

        all_results[param_name] = param_results

    # Save summary
    out_path = Path("results/hyperparam/sensitivity_summary.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved summary to {out_path}")


if __name__ == "__main__":
    main()

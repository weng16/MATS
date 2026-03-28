#!/usr/bin/env python3
"""
Efficiency analysis: measure parameter count, FLOPs, memory, and inference time.

Usage:
    python scripts/eval_efficiency.py
    python scripts/eval_efficiency.py --hidden_dim 128 --device cpu
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import time
import torch
import json
from collections import OrderedDict

from src.models.struct_router import StructRouter


def count_parameters(model):
    """Count total and per-module parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    module_params = OrderedDict()
    for name, module in model.named_children():
        n = sum(p.numel() for p in module.parameters())
        module_params[name] = n

    return total, trainable, module_params


def measure_inference_time(model, x, warmup=10, repeats=50):
    """Measure average inference time in milliseconds."""
    model.eval()
    with torch.no_grad():
        for _ in range(warmup):
            model(x, return_details=False)

        if x.is_cuda:
            torch.cuda.synchronize()

        t0 = time.perf_counter()
        for _ in range(repeats):
            model(x, return_details=False)
        if x.is_cuda:
            torch.cuda.synchronize()
        elapsed = (time.perf_counter() - t0) / repeats * 1000  # ms

    return elapsed


def measure_memory(model, x):
    """Measure peak GPU memory during forward pass (MB)."""
    if not x.is_cuda:
        return 0.0

    torch.cuda.reset_peak_memory_stats()
    model.eval()
    with torch.no_grad():
        model(x, return_details=False)
    peak_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
    return peak_mb


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--seq_len", type=int, default=336)
    p.add_argument("--pred_len", type=int, default=96)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--input_dim", type=int, default=7)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    configs = [
        ("Full", dict(use_segment_processing=True, use_verification=True, use_rft=True)),
        ("w/o Segment", dict(use_segment_processing=False, use_verification=True, use_rft=True)),
        ("w/o Verification", dict(use_segment_processing=True, use_verification=False, use_rft=True)),
        ("w/o RFT", dict(use_segment_processing=True, use_verification=True, use_rft=False)),
        ("Minimal", dict(use_segment_processing=False, use_verification=False, use_rft=False)),
    ]

    results = []
    print(f"\n{'Variant':<20} {'Params':>10} {'Trainable':>10} {'Inf(ms)':>10} {'Mem(MB)':>10}")
    print("-" * 65)

    for name, kwargs in configs:
        model = StructRouter(
            input_dim=args.input_dim,
            output_dim=args.input_dim,
            hidden_dim=args.hidden_dim,
            seq_len=args.seq_len,
            pred_len=args.pred_len,
            use_revin=True,
            **kwargs,
        ).to(args.device)

        x = torch.randn(args.batch_size, args.seq_len, args.input_dim, device=args.device)

        total, trainable, module_params = count_parameters(model)
        inf_time = measure_inference_time(model, x)
        mem_mb = measure_memory(model, x)

        print(f"{name:<20} {total:>10,} {trainable:>10,} {inf_time:>10.2f} {mem_mb:>10.1f}")

        results.append({
            "variant": name,
            "total_params": total,
            "trainable_params": trainable,
            "inference_ms": round(inf_time, 2),
            "peak_memory_mb": round(mem_mb, 1),
            "module_params": {k: v for k, v in module_params.items()},
        })

    print()

    # Save
    out_path = Path("results/efficiency_analysis.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()

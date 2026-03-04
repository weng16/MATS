# StructRouter

**Structure-Aware Multi-Agent Framework for Mixed-Mode Time Series Analysis**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-red.svg)](https://pytorch.org/)

## Overview

StructRouter models time series as a **mixture of learnable structural patterns**
and routes them to specialized experts via **soft routing** with
**interpretable causal communication**.

**Key innovations:**

| ID | Innovation | Core formula |
|----|-----------|-------------|
| C1 | Multi-Pattern Weight Estimation | `w = Softmax(MLP(z) + softplus(λ)·Sim(z, P))` |
| C2 | Soft-Weighted Expert Fusion | `Output = Σ(wᵢ × Expertᵢ(x))` |
| C4 | SCM Causal Communication | `h(W) = tr(exp(W⊙W)) - N = 0` (NOTEARS DAG) |
| C6 | Verification Agent | 3D check + gradient-aware rerouting |
| C7 | Joint Loss with DAG warmup | `L = L_task + α·L_consist + β·L_sparse + γ(t)·L_dag + δ·L_balance + ε·L_ortho` |
| C8 | Three-Stage Training | Self-supervised → End-to-end → RFT (PPO) |

Additional convergence techniques: **RevIN**, per-module learning rates,
prototype orthogonality regularization.

---

## Installation

```bash
conda create -n structrouter python=3.10
conda activate structrouter
pip install -r requirements.txt
```

## Data

Download datasets and place CSVs in `data/{DatasetName}/`:

| Dataset | Variables | Frequency | Source |
|---------|-----------|-----------|--------|
| ETTh1/h2 | 7 | 1h | [ETDataset](https://github.com/zhouhaoyi/ETDataset) |
| ETTm1/m2 | 7 | 15min | [ETDataset](https://github.com/zhouhaoyi/ETDataset) |
| Weather | 21 | 10min | [Autoformer](https://github.com/thuml/Autoformer) |
| ECL | 321 | 1h | [Autoformer](https://github.com/thuml/Autoformer) |
| Traffic | 862 | 1h | [Autoformer](https://github.com/thuml/Autoformer) |

---

## Training

### Standalone Three-Stage Trainer (recommended for full pipeline)

```bash
# Test convergence first (no data needed)
python scripts/test_convergence.py

# TODO: add standalone training script entry point
```

### BasicTS Adapter (Hydra configs, for benchmark comparison)

```bash
# Single experiment
python scripts/train.py --config-name=main/etth1_96

# Specific dataset and prediction length
python scripts/train.py --config-name=main/weather_336

# All main experiments (7 datasets × 4 pred_lens = 28 runs)
python scripts/run_main.py

# Ablation study (ETTh1 + Weather, 6 variants)
python scripts/run_ablation.py
```

### Available Configs

**Main experiments** (`configs/main/`):

```
{etth1,etth2,ettm1,ettm2,weather,ecl,traffic}_{96,192,336,720}.yaml
```

**Ablation** (`configs/ablation/`):

```
{no_segment,no_causal,no_verification,no_rft,hard_routing,single_expert}/{etth1,weather}_96.yaml
```

---

## Project Structure

```
MATS/
├── configs/
│   ├── _base_/
│   │   ├── base.yaml              # Training hyperparameters
│   │   ├── datasets/              # Per-dataset configs (7 files)
│   │   └── model/mats_base.yaml   # Model architecture defaults
│   ├── main/                      # 28 main experiment configs
│   └── ablation/                  # 12 ablation configs
├── data/                          # Dataset CSVs
├── docs/
│   ├── convergence_diagnosis.md   # Root cause analysis + fix prescriptions
│   ├── concept_definitions.md     # C1-C8 formal definitions
│   ├── abstract.md                # Paper abstract
│   └── contributions.md           # 4 itemized contributions
├── paper/                         # ACM MM 2026 paper (acmart sigconf)
│   ├── main.tex
│   ├── contents/                  # Section files (00-05)
│   └── sample-base.bib
├── prompts/
│   ├── proposal.md                # Original task specification
│   ├── figures/                   # Figure generation prompts (5 files)
│   └── writing/                   # Writing quality prompts (3 files)
├── scripts/
│   ├── train.py                   # Hydra + BasicTS training entry
│   ├── run_main.py                # Batch main experiments
│   ├── run_ablation.py            # Batch ablation experiments
│   └── test_convergence.py        # Convergence fix validation (6 tests)
├── src/
│   ├── models/
│   │   ├── struct_router.py       # Main model (orchestrates everything)
│   │   ├── structure_encoder.py   # TCN + Self-Attention encoder
│   │   ├── weight_estimator.py    # C1: prototype-based weight estimation
│   │   ├── expert_fusion.py       # C2: soft-weighted expert fusion
│   │   ├── experts.py             # 5 specialized experts
│   │   ├── causal_communication.py # C4: SCM with DAG constraint
│   │   ├── verification_agent.py  # C6: 3D check + rerouting
│   │   ├── tool_adapter.py        # C3: PPO-based RFT
│   │   ├── segment_processor.py   # Segment detection + fusion
│   │   └── revin.py               # RevIN normalization
│   ├── losses/
│   │   └── joint_loss.py          # C7: joint loss with DAG warmup
│   ├── trainers/
│   │   └── three_stage_trainer.py # C8: three-stage progressive training
│   ├── data/
│   │   └── dataset.py             # Data loading + preprocessing
│   └── utils/
│       └── explainer.py           # Causal graph visualization
├── requirements.txt
└── README.md
```

---

## Convergence Fixes

The original training did not converge. Root causes and fixes are documented
in [`docs/convergence_diagnosis.md`](docs/convergence_diagnosis.md). Summary:

| Fix | What it does |
|-----|-------------|
| RevIN | Instance normalization handles distribution shift |
| VerificationAgent grad fix | Output structure retains gradients for L_consist |
| DAG warmup | γ ramps from 0 to target over first 20% of training |
| Per-module LR | Prototypes: 0.1× base, W_comm: 3× base |
| Expert balance | Variance-of-mean-usage loss prevents collapse |
| Prototype orthogonality | Gram matrix regularization maintains diversity |

Run `python scripts/test_convergence.py` to verify all fixes work.

---

## Citation

```
[TODO: fill after paper acceptance]
```

## License

MIT

# StructRouter Convergence Diagnosis

## Summary

Running `python scripts/train.py` does not converge. This document identifies root causes
and prescribes targeted fixes, each implemented as an isolated wrapper that does **not**
modify the core innovation logic.

---

## Root Cause 1: VerificationAgent Gradient Blockage

**Location**: `src/models/verification_agent.py`, lines 306-311

```python
# CURRENT (BROKEN):
with torch.no_grad():
    input_z = self.structure_encoder(input_seq)
    output_z = self.structure_encoder(output_seq)       # ← kills gradient from output
    input_structure = weight_estimator(input_z)
    output_structure = weight_estimator(output_z)       # ← kills gradient from output
```

**Impact**: The entire forward pass of VerificationAgent wraps structure extraction
in `torch.no_grad()`. This has two devastating effects:

1. `L_consist = ||S(input) - S(output)||²` receives **zero gradient** w.r.t. the
   prediction path, making the structure consistency loss a constant that cannot
   guide learning.
2. The Inverse Consistency Check in `struct_router.py` (lines 386-405) computes
   `prediction2` which depends on `verification_result['inverse_check']['suggested_weights']`,
   but since `output_structure` is computed without gradients, `suggested_weights`
   has no gradient either — the rerouting mechanism trains blind.

**Fix**: Allow gradients through `output_seq` → `output_z` → `output_structure`,
but keep `input_seq` detached (input structure is ground truth, not a training target).
Only detach `overall_confidence` when used as mixing coefficient (already done in
struct_router.py line 404).

---

## Root Cause 2: DAG Penalty Explosion

**Location**: `src/losses/joint_loss.py` (DAGConstraintLoss), `src/models/causal_communication.py`

**Formula**: `h(W) = tr(exp(W ⊙ W)) - N`

**Impact**: With `γ = 0.1` from epoch 0:
- At initialization, `W_comm ~ N(0, 0.01)`, so `W⊙W` entries are ~0.0001 and
  `h(W) ≈ 0`. This is fine.
- But as soon as any `W_comm` entry exceeds ~1.0 (which happens naturally during
  gradient updates), `exp(W⊙W)` grows exponentially, and the DAG loss term
  dominates the total loss, causing the optimizer to focus entirely on shrinking
  `W_comm` rather than improving predictions.
- Observed behavior: `L_causal` jumps from ~0 to >100 within a few epochs, then
  oscillates, preventing `L_task` from decreasing.

**Fix**: DAG penalty warmup — linearly increase `γ` from 0 to target over the
first 20% of epochs. This lets the model first learn useful representations, then
gradually enforce the DAG structure.

---

## Root Cause 3: Expert Weight Collapse

**Evidence**: After ~10 epochs, `expert_weights` converges to near-uniform
`[0.2, 0.2, 0.2, 0.2, 0.2]` or collapses to a single expert `[0.95, 0.01, ...]`.

**Cause**:
1. The `LoadBalanceLoss` in CV mode penalizes deviation from uniform, but the
   `SparsityLoss` in entropy mode simultaneously encourages concentration. These
   two objectives directly conflict at default weights (`δ=0.01` vs `β=0.01`).
2. The `weight_estimator` prototypes are initialized orthogonally (good), but
   there is no ongoing orthogonality regularization — prototypes drift and merge
   during training.
3. The softmax temperature is fixed at 1.0, giving no control over weight sharpness.

**Fix**:
- Add variance-of-mean-usage penalty (per-batch): `L_balance = Var(mean_usage)`
- Add prototype orthogonality regularization: `L_ortho = ||P^T P - I||²_F` with small weight
- Adjust sparsity/balance loss weight ratio

---

## Root Cause 4: No Input Normalization

**Impact**: Time series datasets have vastly different scales:
- ETTh1: OT column range ~[0, 50]
- Traffic: range [0, 1]
- Weather: mixed scales across variables

Without instance normalization, the model must waste capacity learning per-dataset
scale transformations. This is the single most impactful missing component — every
modern LTSF baseline (PatchTST, iTransformer, DLinear) uses RevIN.

**Fix**: Add Reversible Instance Normalization (RevIN) wrapping the entire
forward pass:
- Before encoder: normalize to zero-mean unit-variance per instance per channel
- After prediction head: denormalize using saved statistics

---

## Root Cause 5: Uniform Learning Rate

**Impact**: All parameters share the same learning rate (5e-4 in Stage 2).
But different modules have very different optimization landscapes:
- `weight_estimator.pattern_prototypes`: should learn slowly (they define the
  structure of the latent space)
- `communication.W_comm`: should learn faster (needs to find DAG structure quickly
  before the penalty ramps up)
- Expert networks: standard learning rate

**Fix**: Per-module parameter groups with different learning rates.

---

## Root Cause 6: No Learning Rate Warmup in Practice

The `Stage2Finetuner` has warmup logic, but `scripts/train.py` uses BasicTS
which bypasses the three-stage trainer entirely. The standalone trainer's warmup
works correctly but the primary training path lacks it.

**Fix**: Ensure both training paths have consistent warmup + cosine annealing.

---

## Prescription (Implementation Order)

| Priority | Fix | Files Modified | Risk |
|----------|-----|----------------|------|
| P0 | RevIN wrapper | New: `src/models/revin.py`, Modified: `struct_router.py` | Low |
| P0 | Fix VerificationAgent gradient bug | `verification_agent.py` | Low |
| P1 | DAG penalty warmup | `joint_loss.py`, `three_stage_trainer.py` | Low |
| P1 | Per-module LR groups | `three_stage_trainer.py` | Low |
| P1 | Expert balance loss improvement | `joint_loss.py` | Low |
| P2 | Prototype orthogonality reg | `joint_loss.py` | Low |
| P2 | LR warmup + cosine (both paths) | `three_stage_trainer.py`, `configs/` | Low |
| P2 | Channel independence option | `struct_router.py`, `experts.py` | Medium |

All fixes are designed as wrappers/additions that do not alter the core formulas:
- `w = Softmax(MLP(z) + softplus(λ)·Sim(z,P))` — unchanged
- `Output = Σ(wᵢ × Expertᵢ(x))` — unchanged
- `h(W) = tr(exp(W⊙W)) - N = 0` — unchanged
- VerificationAgent 3D check logic — unchanged (only gradient flow fixed)

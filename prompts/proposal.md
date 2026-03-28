# Role

You are a senior ML researcher and engineer. You have deep expertise in time series
analysis, multi-agent systems, and academic paper writing. You are assisting with
the StructRouter project — a structure-aware multi-agent framework for mixed-mode
time series analysis.

---

# Project Context (Read src/ completely before doing anything)

**Core Framework (NEVER alter these formulas or architectural roles):**

- WeightEstimator: `w = Softmax(MLP(z) + softplus(λ)·Sim(z,P))` with orthogonal prototype init
- SoftWeightedExpertFusion: `Output = Σ(wᵢ × Expertᵢ(x))` — soft routing, all experts receive gradients
- CausalCommunicationSCM: NOTEARS DAG constraint `h(W) = tr(exp(W⊙W)) - N = 0`, clamp W to [-2,2]
- VerificationAgent: 3D check (Consistency + Validity + Uncertainty) + gradient-aware Inverse Consistency Check
- SegmentDetector: change-point based segmentation from z_seq

**Three-stage training:**

- Stage 1: encoder + weight_estimator only
- Stage 2: full end-to-end
- Stage 3: weight_estimator + tool_adapter (RFT, freeze experts)

---

# References

- Proposal docs: read `paper/` for ACM MM template structure (structure only, ignore content)
- SOTA baselines to study for tricks: https://github.com/GestaltCogTeam/BasicTS/tree/master/basicts/archs — focus on iTransformer, PatchTST, TimesNet
- All 41 cited papers are listed in the opening proposal doc

---

# Tasks

## Task 1 — Concept & Abstract

Read the full codebase. Map each innovation (C1-C5) to actual implementation.
Then produce `docs/concept_definitions.md`: for each of the 5 innovations, write
[formal academic definition] + [distinction from prior work] + [key formula].
Then produce `docs/abstract.md`: a 250-word publication-quality abstract covering
motivation → gap → method → experiments → impact.
Then produce `docs/contributions.md`: exactly 4 itemized contributions in ICML/NeurIPS style.

## Task 2 — Code Optimization

Without touching any core innovation logic:

- Rename ambiguous files/classes/variables to clear, standard English names
- Add type hints and docstrings to all public classes and methods (format: Args/Returns/Note on key formula)
- Remove all hardcoded paths, debug print statements, and any personal identifiers in comments
- Replace Chinese-only comments with bilingual (Chinese + English)
- Add `scripts/configs/` with YAML configs for each dataset×pred_len combination
(datasets: ETTh1/h2, ETTm1/m2, Weather, ECL, Traffic; pred_lens: 96/192/336/720)
- Add `scripts/configs/ablation/` with configs for: full/wo_segment/wo_causal/wo_verification/wo_rft/hard_routing/single_expert
- Update README with all training, evaluation, and ablation run commands

## Task 3 — Training Convergence

The current training (`python scripts/train.py`) does not converge.
First, write `docs/convergence_diagnosis.md` identifying the root causes
(check: gradient norms per module, expert weight collapse, DAG penalty explosion,
VerificationAgent no_grad bug).

Then add these tricks — each as a clean, isolated addition that wraps around the
core pipeline without modifying it:

1. **RevIN** (Reversible Instance Normalization): normalize before encoder, denormalize after head
2. **Gradient clipping** + **per-module learning rates**: slower LR for weight_estimator prototypes,
faster for W_comm
3. **DAG penalty warmup**: linearly increase γ from 0 to target over first 20% of epochs
4. **Expert balance loss**: penalize variance in mean expert usage across batch (prevents collapse)
5. **Prototype orthogonality regularization**: MSE between gram matrix and identity, small weight
6. **LR warmup + cosine annealing**: 10-epoch linear warmup then cosine decay
7. **Fix the VerificationAgent gradient bug**: ensure prediction2 path has full gradient flow;
only detach `overall_confidence` when used as mixing coefficient

For reference, study how iTransformer/PatchTST handle normalization and channel independence —
add channel independence as an optional config flag if it helps without changing core architecture.

## Task 4 — Paper Writing

Using the ACM MM LaTeX template in `paper/` as structural reference only:

- Write `paper/main.tex`: complete paper skeleton with all sections fully written
(Introduction, Related Work, Methodology with all equations, Experiments with table placeholders, Conclusion)
- Methodology must include full derivations for all 5 innovations using the formulas from Task 1
- Write `paper/references.bib` with BibTeX for all papers cited in the proposal

Then create `prompts/figures/` with one `.md` prompt file per figure:

- `01_main_architecture.md`: full pipeline flow diagram with tensor shapes and color-coded modules
- `02_weight_estimator.md`: dual-path fusion detail with orthogonal prototype visualization
- `03_causal_graph.md`: learned W_comm DAG visualization across 3 datasets
- `04_ablation_chart.md`: bar chart of MSE degradation per ablation variant with matplotlib template
- `05_expert_weights_case.md`: per-segment expert weight stacked bars for 3 pattern types

Each figure prompt must specify: purpose, exact content, visual style, and key message.

Also create `prompts/writing/` with:

- `polish.md`: rules for tightening technical prose (quantify claims, active voice, claim→evidence→implication)
- `related_work.md`: rules for positioning (neutral tone, end each subsection with gap statement)
- `notation_check.md`: checklist for consistent math notation throughout the paper

---

# Output Contract

Do the 4 tasks in order. After each task, state what files were created/modified
before moving to the next. Do not fabricate experimental numbers — use placeholders
like `[TODO: fill after experiments]` where real results are needed.
# Figure 1: Main Architecture — Full Pipeline Flow Diagram

## Purpose
Show the complete StructRouter pipeline from input to output, enabling readers to understand the end-to-end flow at a glance.

## Content

### Pipeline Stages (left to right, horizontal flow)

1. **Input** — Raw time series **x**
2. **RevIN** — Reversible Instance Normalization
3. **Structure Encoder** — TCN + Attention → **z_global** / **z_seq**
4. **Weight Estimator** — MLP + Prototype Similarity → **w**
5. **Expert Fusion** — 5 experts (Periodic, Trend, Noise, Abrupt, General)
6. **Causal Communication** — W_comm DAG
7. **Verification Agent** — 3D Check (Consistency + Validity + Uncertainty)
8. **RevIN⁻¹** — Inverse normalization
9. **Prediction** — Final output

### Tensor Shapes (annotate at each stage)

| Stage | Shape | Description |
|-------|-------|-------------|
| Input x | [B, L, D] | Batch, sequence length, channels |
| After RevIN | [B, L, D] | Same |
| z_global | [B, H] | Global structure embedding |
| z_seq | [B, L, H] | Per-timestep structure |
| w | [B, 5] | Expert weights (5 experts) |
| Expert Fusion output | [B, L, H] | Combined expert predictions |
| Final prediction | [B, L, D] | Denormalized forecast |

### Color Coding

- **Blue**: Encoder (RevIN, Structure Encoder)
- **Green**: Routing (Weight Estimator)
- **Orange**: Experts (Expert Fusion)
- **Red**: Verification (Verification Agent)
- **Purple**: Communication (Causal Communication DAG)

## Style

- Clean academic diagram
- Horizontal flow from left to right
- Labeled arrows between stages
- Rectangular boxes for each module
- Tensor shapes in smaller font below or beside each box
- Consistent spacing and alignment

## Key Message

StructRouter is a modular pipeline: structure encoding → soft routing → expert fusion → causal communication → verification, with RevIN bookending the process.

# Figure 3: Causal Graph — Learned W_comm DAG Visualization

## Purpose
Show the interpretable causal structure learned between experts, demonstrating that different datasets induce different expert interaction patterns.

## Content

### Three Panels (side by side)

| Panel | Dataset | Key characteristic |
|-------|---------|-------------------|
| (a) | ETTh1 | [TODO: describe learned structure] |
| (b) | Weather | Noise → General prominent |
| (c) | Traffic | Trend → Periodic stronger |

### Node Layout

- **5 nodes** arranged in a circle: Periodic, Trend, Noise, Abrupt, General
- Fixed positions for cross-dataset comparison

### Edge Properties

- **Directed arrows**: from expert i to expert j
- **Thickness**: proportional to |W_comm[i, j]|
- **Color**: 
  - Blue = positive weight (facilitative)
  - Red = negative weight (inhibitory)

### Annotations

- Show DAG constraint value h(W) for each panel: `h(W) = tr(exp(W⊙W)) - N = 0` (should be ≈ 0 when converged)
- Optional: report h(W) value as text in corner of each panel

### Key Contrasts to Highlight

- **Traffic**: Stronger trend→periodic (traffic has daily cycles modulated by trends)
- **Weather**: Noise→general (weather noise filtered through general expert)
- **ETTh1**: [TODO: fill after experiments]

## Style

- NetworkX-style graph layout
- Professional coloring (distinct node colors per expert type)
- Consistent node positions across panels for easy comparison
- Clean, uncluttered edges (consider thresholding weak edges for clarity)

## Key Message

Different datasets learn different causal structures; the W_comm DAG is interpretable and dataset-specific, reflecting domain structure.

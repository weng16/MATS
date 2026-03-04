# Figure 2: Weight Estimator — Dual-Path Fusion Detail

## Purpose
Detail the Multi-Pattern Weight Estimator (C1) mechanism, showing how MLP and prototype similarity combine to produce expert weights.

## Content

### Dual-Path Architecture

**Input**: z (structure embedding, [B, H])

**Path 1 — MLP**:
- z → MLP → logits ([B, 5])

**Path 2 — Prototype Similarity**:
- Prototypes P = [p₁, p₂, p₃, p₄, p₅] as vectors in embedding space
- Orthogonal initialization (prototypes mutually orthogonal)
- Sim(z, P) → similarity scores → sim_logits ([B, 5])

**Fusion**:
- Combined: `logits + softplus(λ) × sim_logits`
- Softmax → **w** ([B, 5])

### Visualization Elements

1. **Prototype vectors**: Show P = [p₁...p₅] as arrows in 2D projection of embedding space
2. **Orthogonality**: Arrows should be visually orthogonal (90° between adjacent prototypes)
3. **Input mapping**: Show input z as a point, with dashed lines to nearest prototypes (highest similarity)
4. **Split diagram**: Left panel = Path 1 (MLP), Right panel = Path 2 (Prototype Sim), Bottom = fusion and Softmax

### Diagram Layout

```
        z [B, H]
           |
    +------+------+
    |             |
  Path 1       Path 2
  (MLP)    (Prototype Sim)
    |             |
  logits    sim_logits
    |             |
    +------+------+
           |
    logits + softplus(λ) × sim
           |
        Softmax
           |
         w [B, 5]
```

## Style

- Split diagram showing both paths merging
- 2D projection of prototype space as inset or secondary panel
- Arrows indicating flow direction
- Clear labels for λ (learnable scaling)

## Key Message

The weight estimator fuses data-driven MLP logits with interpretable prototype similarity; orthogonal prototypes ensure distinct pattern representations.

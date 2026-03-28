# Notation Checklist: Consistent Math Throughout the Paper

## Typography

| Element | Convention | Example |
|---------|------------|---------|
| Vectors | Bold lowercase | \(\mathbf{x}\), \(\mathbf{z}\), \(\mathbf{w}\) |
| Matrices | Bold uppercase | \(\mathbf{W}\), \(\mathbf{P}\) |
| Loss terms | Calligraphic | \(\mathcal{L}\), \(\mathcal{L}_{\text{rec}}\) |
| Sets / spaces | Calligraphic | \(\mathcal{X}\), \(\mathcal{Z}\) |
| Scalars | Italic | \(x\), \(\lambda\), \(\gamma\) |

## Subscripts and Superscripts

| Use | Convention | Example |
|-----|------------|---------|
| Component names | `\text{}` | \(z_{\text{global}}\), \(z_{\text{seq}}\) |
| Time index | Superscript | \(x^{(t)}\), \(h^{(t)}\) |
| Layer index | Superscript | \(h^{(\ell)}\) |
| Expert index | Subscript | \(w_i\), \(\text{Expert}_i\) |

## Dimension Symbols (use consistently)

| Symbol | Meaning | Typical use |
|--------|---------|-------------|
| \(B\) | Batch size | [B, L, D] |
| \(L\) | Sequence length | [B, L, D] |
| \(D\) | Input dimension / channels | [B, L, D] |
| \(H\) | Hidden dimension | [B, H], [B, L, H] |
| \(K\) | Number of experts | \(K=5\) |
| \(N\) | Number of agents / nodes | DAG: \(N\) nodes |

## Predefined Symbols (define once, use everywhere)

- **x**: input time series
- **z**: structure embedding (z_global, z_seq)
- **w**: expert weights
- **P**: prototype matrix
- **W_comm**: causal communication matrix
- **λ**: prototype similarity scaling (softplus)
- **γ**: DAG penalty weight

## Checks Before Final Draft

- [ ] **Every symbol defined before first use** — no undefined notation.
- [ ] **Same symbol never means two different things** — e.g., don't use \(N\) for both sequence length and number of experts.
- [ ] **Dimensions consistent across equations** — e.g., if \(w \in \mathbb{R}^{B \times K}\), then \(\sum_i w_i = 1\) per batch element.
- [ ] **Bold/calligraphic used correctly** — vectors bold, matrices bold, losses calligraphic.
- [ ] **Subscripts for components** — z_global, z_seq, not z_g or z_s in prose (abbreviations ok in equations if defined).

## LaTeX Snippets

```latex
% Vectors
\mathbf{x}, \mathbf{z}, \mathbf{w}

% Matrices  
\mathbf{W}, \mathbf{P}

% Loss
\mathcal{L}_{\text{total}}

% Subscripts
z_{\text{global}}, z_{\text{seq}}

% Dimensions
[B, L, D], [B, H], [B, K]
```

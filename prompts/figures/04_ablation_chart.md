# Figure 4: Ablation Chart — MSE Degradation per Component

## Purpose
Visually show the contribution of each component to overall performance; demonstrate that soft routing clearly beats hard routing.

## Content

### Grouped Bar Chart

**X-axis** (ablation variants, in order):
1. Full (baseline, best)
2. w/o Segment
3. w/o Causal
4. w/o Verification
5. w/o RFT
6. Hard Routing
7. Single Expert (worst)

**Y-axis**: MSE on ETTh1, pred_len=96

**Bar values**: [TODO] — use placeholders until experiments complete. Example structure:
```python
variants = ['Full', 'w/o Segment', 'w/o Causal', 'w/o Verification', 'w/o RFT', 'Hard Routing', 'Single Expert']
mse_values = [0.XXX, 0.XXX, 0.XXX, 0.XXX, 0.XXX, 0.XXX, 0.XXX]  # TODO: fill
std_values = [0.XXX, ...]  # TODO: fill for error bars
```

### Error Bars

- Include standard deviation or confidence intervals across runs
- Use same number of runs per variant for fair comparison

### Visual Emphasis

- Full model: distinct color (e.g., dark green) or hatched
- Hard Routing vs Full: annotate with % degradation
- Single Expert: clearly worst, annotate if desired

## Style

- Matplotlib with seaborn style
- Error bars on each bar
- Clean axis labels, readable font sizes
- Optional: horizontal line at Full model MSE for reference

## Key Message

Each component contributes; removing any degrades performance. Soft routing (Full) clearly outperforms Hard Routing and Single Expert.

---

## Matplotlib Template Code

```python
import matplotlib.pyplot as plt
import numpy as np

# Seaborn style (use 'seaborn-whitegrid' if seaborn < 0.12)
try:
    plt.style.use('seaborn-v0_8-whitegrid')
except OSError:
    plt.style.use('seaborn-whitegrid')

variants = ['Full', 'w/o Segment', 'w/o Causal', 'w/o Verification', 'w/o RFT', 'Hard Routing', 'Single Expert']
mse_values = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # TODO: fill after experiments
std_values = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # TODO: fill

x = np.arange(len(variants))
width = 0.6

fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.bar(x, mse_values, width, yerr=std_values, capsize=4, 
              color=['#2ecc71'] + ['#3498db']*4 + ['#e74c3c']*2,  # Full=green, ablations=blue, hard/single=red
              edgecolor='black', linewidth=0.5)

ax.set_ylabel('MSE (ETTh1, pred_len=96)', fontsize=12)
ax.set_xlabel('Ablation Variant', fontsize=12)
ax.set_title('Ablation Study: Component Contribution', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(variants, rotation=45, ha='right')
ax.set_ylim(bottom=0)

# Optional: horizontal reference line at Full model
full_mse = mse_values[0]
ax.axhline(y=full_mse, color='gray', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig('ablation_chart.pdf', dpi=300, bbox_inches='tight')
plt.show()
```

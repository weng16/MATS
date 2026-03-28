# Figure 5: Expert Weights — Per-Segment Case Visualization

## Purpose
Show that different segments of a time series receive different expert weights; the model correctly identifies and routes different structural patterns.

## Content

### Three Panels (vertical stack or 1×3 grid)

| Panel | Signal Type | Expected Weight Pattern | Key Message |
|-------|-------------|-------------------------|-------------|
| 1 | Periodic-dominant | High w₁ (Periodic expert) | Clear cycles → periodic expert |
| 2 | Trend-dominant | High w₂ (Trend expert) | Monotonic drift → trend expert |
| 3 | Mixed signal | Balanced weights across experts | Multiple patterns → distributed routing |

### Layout per Panel

**Top** (each subplot):
- The actual time series plotted (line plot)
- X-axis: time steps
- Y-axis: value
- Optional: segment boundaries if using SegmentDetector

**Bottom** (each subplot):
- Stacked bar chart showing w₁, w₂, w₃, w₄, w₅ per segment
- X-axis: segment index (or time bins)
- Y-axis: weight (0–1, stacked to 1)
- Color-coded by expert type

### Expert Color Mapping (consistent across panels)

- w₁ Periodic: e.g., blue
- w₂ Trend: e.g., green
- w₃ Noise: e.g., gray
- w₄ Abrupt: e.g., orange
- w₅ General: e.g., purple

### Data Requirements

- 3 example time series from validation set (or synthetic)
- Per-segment weights from model forward pass (or per-window if no segmentation)
- Segment indices from SegmentDetector if available

## Style

- Matplotlib with 3 subplots
- Color-coded by expert type (consistent palette)
- Clean layout: time series above, stacked bars below
- Legend for expert types (shared once)

## Key Message

The model correctly identifies and routes different structural patterns: periodic signals → periodic expert, trend signals → trend expert, mixed signals → balanced weights.

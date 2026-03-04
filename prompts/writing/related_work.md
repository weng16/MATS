# Writing Rules: Positioning Related Work

## Tone

- **Neutral and respectful** — never dismiss prior work.
- Acknowledge contributions fairly. Avoid: "merely", "only", "simply", "just".
- Frame limitations as gaps, not failures. "Prior work does not address X" not "Prior work fails at X."

## Structure per Subsection

For each related-work subsection, use this flow:

1. **Describe the method**: What does it do? What is the core idea?
2. **State what it achieves**: Cite reported results or strengths.
3. **Identify specific limitation**: One concrete gap (not a laundry list).
4. **Transition**: Lead into why StructRouter addresses this gap.

## Gap Statement Format

End each subsection with a clear gap statement. Use this template:

> "However, [specific limitation]. In contrast, StructRouter [specific advantage]."

**Examples**:
- "However, these methods treat all channels independently and do not model cross-channel structure. In contrast, StructRouter uses a structure encoder to capture global patterns before routing."
- "However, existing MoE approaches use hard routing, which prevents gradient flow to unused experts. In contrast, StructRouter employs soft weighted fusion so all experts receive gradients."

## Citation Density

- Cite **at least 3–5 works** per subsection.
- Prefer recent (last 3–5 years) and highly cited works.
- Include foundational work (e.g., original Transformer, TCN) where relevant.

## Organization

- **Group by approach**, not chronologically.
- Suggested groupings: (1) Deep learning for time series, (2) Mixture-of-Experts / routing, (3) Causal / structural modeling, (4) Verification / uncertainty.
- Within each group, order by conceptual proximity to StructRouter.

## Checklist

- [ ] No dismissive language.
- [ ] Each subsection ends with a gap statement.
- [ ] 3–5 citations per subsection.
- [ ] Grouped by approach.
- [ ] Clear transition from limitation to StructRouter's contribution.

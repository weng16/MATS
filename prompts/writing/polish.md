# Writing Rules: Tightening Technical Prose

## Quantify All Claims

- **Never** say "significantly" without a number. Use: "reduces MSE by 12.3%" or "improves by 0.015 on ETTh1."
- Replace vague intensifiers with exact metrics: "substantially" → "by 15%", "considerably" → "2× faster".
- Every comparative claim must have a concrete value or range.

## Active Voice Throughout

- Prefer: "We propose" over "It is proposed."
- Prefer: "The model learns" over "The model is learned."
- Prefer: "We evaluate" over "Evaluation is performed."
- Avoid passive constructions unless the agent is genuinely unknown or irrelevant.

## Claim → Evidence → Implication Pattern

For every non-trivial claim, follow this structure:

1. **Claim**: State the finding or result.
2. **Evidence**: Cite the experiment, table, or figure that supports it.
3. **Implication**: Explain why it matters for the reader or the method.

Example: "StructRouter outperforms PatchTST by 8.2% MSE on ETTh1 (Table 2). This indicates that structure-aware routing captures patterns missed by channel-independent baselines."

## Avoid Weasel Words

- Remove: "quite", "fairly", "somewhat", "rather", "relatively", "generally", "typically" (unless statistically precise).
- Replace with specific qualifiers or delete if redundant.

## Tighten Wordy Phrases

| Wordy | Tight |
|-------|-------|
| in order to | to |
| due to the fact that | because |
| in the case of | for |
| at this point in time | now |
| a large number of | many |
| in spite of the fact that | although |
| has the ability to | can |
| is able to | can |

## Sentence Structure

- **One idea per sentence.** Do not chain multiple findings with semicolons or "and" unless tightly related.
- **Max 25 words** for key claims (contributions, main results). Longer sentences for background or technical detail only.
- Prefer short subject–verb–object order. Avoid long introductory clauses.

## Tense Usage

- **Present tense** for method description: "The encoder produces z_global."
- **Past tense** for experiments: "We trained on ETTh1 for 50 epochs."
- **Present tense** for general truths and paper narrative: "Time series often exhibit mixed patterns."

## Checklist Before Submission

- [ ] Every "significantly" / "substantially" has a number.
- [ ] No passive voice where active is possible.
- [ ] Every claim has evidence (table/figure/section).
- [ ] No weasel words.
- [ ] Key sentences under 25 words.
- [ ] Method in present, experiments in past.

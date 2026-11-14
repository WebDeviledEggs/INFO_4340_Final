# Claude vs. human labels on the eval set (E001–E100)

Prompt v2 (frozen 2026-11-13). Model `claude-opus-5-5`, effort low. Output: `data/llm/eval_v2.csv`, 100/100 parsed, $0.43.

Agreement is weighted κ for tier, role and tone (quadratic) and Cohen's κ for the flags. Bootstrap 95% CIs use 1,000 resamples.

| Field | Claude vs. partner 1 | Claude vs. partner 2 | Governance rule (κ ≥ 0.6) |
|---|---|---|---|
| `projected_tier` | 0.735 (0.62–0.81) | 0.795 (0.71–0.85) | Kept |
| `role` | 0.758 (0.65–0.83) | 0.834 (0.75–0.89) | Kept |
| `tone` | 0.711 (0.62–0.78) | 0.711 (0.62–0.78) | Kept |
| `injury_flag` | 0.878 (0.65–1.00) | 1.000 (1.00–1.00) | Kept |
| `character_flag` | 0.000 | 0.000 | **Dropped** |

## Decision: drop `character_flag` from Claude's results (option A)

- Claude gave 0 positives on the 100 eval items. Partner 1 gave 6 and partner 2 gave 7.
- The 2023 pilot gave 2 positives out of 52, so the flag rarely fires in either run.
- The prompt is frozen, and the test set can't be used to tune it. So the flag stays as a documented failure and is excluded from the Claude results under the κ ≥ 0.6 rule.
- This should be stated in the write-up as a limitation of the prompt, not as a finding about the text.
- The batch file (`data/llm/batch_v2_analysis.csv`, 5,667 reports) still contains the column: 315 positives, concentrated in the train split (6.2%) and 2014 (19%); test split 0.8% (11 of 1,302). The 100 eval items match the batch on this field, so the zero is not a batch problem. The test-year rate is too low for 100 items to be expected to contain many positives.
- Model E features are restricted to `CLAUDE_FEATURE_FIELDS` in `src/llm.py` (projected_tier, role, injury_flag, tone). `character_flag` is not a model feature.

## Sensitivity check: masking leaks removed (n = 94)

Items removed: E011, E031, E042, E069, E085, E094 (see `docs/masking_limitations.md`).

| Field | P1 vs P2, all 100 → 94 | Claude vs P1, all 100 → 94 | Claude vs P2, all 100 → 94 |
|---|---|---|---|
| `projected_tier` | 0.907 → 0.902 | 0.735 → 0.713 | 0.795 → 0.783 |
| `role` | 0.883 → 0.878 | 0.758 → 0.748 | 0.834 → 0.827 |
| `tone` | 0.819 → 0.812 | 0.711 → 0.719 | 0.711 → 0.720 |
| `injury_flag` | 0.878 → 0.864 | 0.878 → 0.864 | 1.000 → 1.000 |

Reading: removing the leaking items moves every number by 0.02 or less, which is inside the bootstrap intervals. For tier and role, Claude's agreement with the partners drops slightly when the leaking items are removed. That is the direction you'd expect if the leaks helped Claude a little, but the effect is small. Claude's disagreement with the partners on E085 (R2-3 vs R1) is one example. The main findings don't change.

## Notes

- Claude flagged `recognized` on 5 items. Check these against the partners' notes when the write-up is drafted.
- The masking leaks in `docs/masking_limitations.md` affect Claude and the partners the same way.

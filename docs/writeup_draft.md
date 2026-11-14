# Does Analyst Text Help Predict the NFL Draft? A Reproducible AI-Assisted Workflow

**Draft, 2026-11-14.** Numbers come from `nfl_draft_text_analysis.ipynb` (sections 7–12) and the `docs/` result files. Target length: 3–5 pages.

## 1. Introduction

NFL.com publishes a written scouting report and a numeric grade for each prospect before the draft. Teams and draft media use both. Reading a report carefully takes time, and a large class can have more than 500 of them. Two questions matter for anyone deciding whether to use the text at scale:

1. Does the written text predict where a prospect is drafted, beyond the objective data and the analyst's own grade?
2. Can a language model turn the text into structured labels that help, and can those labels be trusted?

We also ask whether the NFL's own machine-learning score (`draftGrade`) does better than the analysts' grade. The four research questions are:

1. Does analyst scouting text help predict draft outcome beyond objective data and the analyst's grade? (Model D vs. B)
2. Do the scouting reports tell you anything about draft position that the numbers don't? (Text-only and length checks)
3. Does the NFL's machine-learning model (`draftGrade`) outperform the analysts (`grade`)? (Compare on the same test set)
4. Does Claude's reading of the text add predictive power? (Model E vs. B)

The workflow is: data → cleaning → LLM labeling → models → evaluation → governance rules. The AI step is an LLM that reads each report and returns structured labels. A person checks those labels on a sample before they are used.

**Why AI is a risky tool here.** The model may know how players were drafted (contamination), and the text may only repeat the analyst's grade. Both risks are tested below.

## 2. Data

- **Source:** `nfl_prospects_2014_2026.csv`, 6,152 prospect profiles from 13 draft classes (2014–2026), scraped from the NFL.com draft tracker by `scrape_nfl_prospects.py`.
- **Unit of observation:** one prospect in one draft class.
- **Analysis set:** 5,667 prospects with report text, an analyst grade, a known position, and not a specialist (K, P, LS). Text is present for 5,826 prospects; the analysis set is the subset the models use.
- **Outcome:** draft tier, a four-level ordinal label: Round 1, Rounds 2–3, Rounds 4–7, Undrafted. Undrafted players are about 46% of the rows and are kept, which tier labels allow and a pick-number label does not.
- **Splits, by year and never random:** train 3,453 (2015–2022), validation 473 (2023), test 1,302 (2024–26), and a 2014 holdout of 439 from a different analyst, used only as a check.
- **Known data issues:**
  - Analyst grade has a scale shift around 2021–22. 802 grades are below 5.50, the lowest described tier, and 95% of those are from 2014–2021. We kept all rows and use `grade_pct`, the percentile within the class.
  - `draftGrade` is missing for 1,596 of the 5,667 analysis rows (coverage 65–83% by class year). The draftGrade comparison uses only the rows that have it.
  - Author names, school names, and team names are masked in the text sent to the LLM. Six of the 100 eval reports still contain leaks (`docs/masking_limitations.md`).

## 3. Methods

**Models.** Every tier model is a class-weighted multinomial logistic regression, so the models differ only in the information they receive:

| Model | Information |
|---|---|
| A | Objective: measurables, production and athleticism scores, position, Power-5 flag |
| B | A + analyst grade |
| C | Text only (TF-IDF, words and bigrams) |
| L | Report length only |
| D | B + out-of-fold text probabilities |
| E | B + Claude fields (projected tier, role, injury flag, tone) |

Regularization strength (C) is chosen on the 2023 validation split. Gradient boosting is a robustness check.

**AI tool and settings.** Claude Opus 5.5 (`claude-opus-5-5`), effort "low", JSON output with a fixed schema, prompt v2. The prompt is a codebook: it defines each field and its decision rules. The prompt was piloted on 52 reports from 2023 with a three-round limit and then frozen. The self-reported confidence field was removed on 2026-11-12 by partner decision (codebook v1.1). The full run used the Message Batches API on 5,667 analysis reports ($11.62 at batch prices; batch IDs and outputs are saved in `data/llm/`).

**Labeling and agreement.** Both partners labeled the same 100 eval reports (E001–E100, 2024–26) independently, following the codebook. Agreement is quadratic-weighted κ for ordinal fields (tier, role, tone) and Cohen's κ for flags. Bootstrap 95% CIs use 1,000 resamples. Governance rule: a field is kept only if its κ is at least 0.6 against both partners.

**Evaluation.** Macro-F1 and quadratic-weighted κ on the test set. Differences between models are paired bootstrap deltas (the same test prospects for both models, 1,000 resamples). A difference counts as detectable if its CI excludes zero and it is larger than the minimum detectable difference (MDD). The tier MDD is 0.029 macro-F1, measured from training-data noise.

**Assumptions.** The 2014 analyst is a different writer, so the 2014 holdout is a check, not a test of the main claim. Test years 2024–26 are used for final scores only, and the 2014 holdout is not used in any comparison that drives a conclusion. The masking is imperfect.

## 4. Results

**Model comparison (test set, 2024–26, 1,302 prospects).**

| Model | Macro-F1 (95% CI) | QWK |
|---|---|---|
| L: length only | 0.395 (0.359–0.426) | 0.482 |
| C: text only (TF-IDF) | 0.448 (0.417–0.479) | 0.544 |
| A: objective | 0.578 (0.546–0.610) | 0.700 |
| D0: objective + text | 0.603 (0.573–0.631) | 0.730 |
| B: objective + grade | 0.662 (0.633–0.691) | 0.788 |
| D: objective + grade + text | 0.662 (0.634–0.691) | 0.785 |

Paired differences (macro-F1):
- **Grade beyond objective (B − A):** +0.084 (CI +0.053 to +0.113). Detectable.
- **Text beyond objective and grade (D − B):** −0.000 (CI −0.011 to +0.012). Not detectable. **Q1 answer: no.**
- **Text beyond objective only (D0 − A):** +0.025 (CI −0.002 to +0.051). Not detectable.

**Q2 (does text say more than the numbers?).** Text alone reaches 0.448 macro-F1, which is well above chance but well below the objective data (0.578). The length-only model reaches 0.395, so most of the text's signal is in what the report says, not how long it is. Adding text to the objective data (D0) adds nothing detectable once grade is included.

**Q3 (draftGrade vs. grade).** On the 950 test prospects that have both scores:
- Grade alone 0.582 macro-F1; draftGrade alone 0.559. Difference −0.023 (CI −0.063 to +0.015). Not detectable.
- Adding draftGrade to objective + grade: −0.017 (CI −0.045 to +0.009).
- Objective + draftGrade without grade is worse than objective + grade: −0.071 (CI −0.105 to −0.038). **Answer: draftGrade does not outperform the analysts' grade.**

**Q4 (Claude fields).** On the full test set:
- E − B (add Claude fields to objective + grade): +0.001 (CI −0.014 to +0.017). Not detectable.
- D − B (add text model): −0.001 (CI −0.013 to +0.012). Not detectable.
- **Answer: no detectable added value.** The CI rules out gains above about 0.02, so smaller effects remain possible.

**Human agreement and the LLM.** Human–human κ on the eval items is 0.907 for tier, 0.883 for role, and 0.819 for tone. Claude against each partner:

| Field | Claude vs. partner 1 | Claude vs. partner 2 | Kept? |
|---|---|---|---|
| Projected tier | 0.735 (0.62–0.81) | 0.795 (0.71–0.85) | Yes |
| Role | 0.758 (0.65–0.83) | 0.834 (0.75–0.89) | Yes |
| Tone | 0.711 (0.62–0.78) | 0.711 (0.62–0.78) | Yes |
| Injury flag | 0.878 (0.65–1.00) | 1.000 (1.00–1.00) | Yes |
| Character flag | 0.000 | 0.000 | **No** (Claude gave 0 positives; partners gave 6 and 7) |

The character flag fails the κ rule and is excluded from Model E. The 2023 pilot and the test-year base rate (0.8% positives) suggest the flag is rare in the test years, so the failure is partly a base-rate problem as well as a prompt problem.

**Contamination.** Claude marked 322 of 5,667 reports (5.7%) as recognized. Accuracy against the real tier is 0.543 for recognized players and 0.527 for the rest. On the test split the gap is 0.60 vs. 0.54, but only 40 players are recognized, so the difference cannot be separated from noise.

**Subgroups (Model D, test errors).** Overall error rate is 0.347 (452 of 1,302). Power-5 schools have an error rate of 0.366 vs. 0.265 for others (disparate-impact ratio 1.38, chi-square p = 0.004). This is the one subgroup gap that is clearly outside chance. Position group (p = 0.675) and test year (p = 0.171) show no significant difference. Low-grade prospects have lower error (0.226) than mid (0.422) and high (0.395) grade tertiles, but that is largely a class-mix effect.

**Cost.** The project spent $12.49 on LLM calls: $0.22 for the pilot v1, $0.22 for pilot v2, $0.43 for the 100 eval reports, and $11.62 for the batch. The batch set would have cost $23.24 at standard prices, so the Batch API saved about half. Per report at batch prices: $0.002. A human labeling at roughly 30 reports per hour at $25/hour would cost about $0.83 per report, roughly 400 times as much.

## 5. Governance

**Risk level: medium.** The text does not improve predictions beyond the analyst's grade, and the LLM labels do not add detectable value either. The main risks are that a reader may take LLM labels as analysis, that subgroup errors may go unnoticed, and that the character flag may be trusted when it is not reliable.

**Key rules, each with a number:**
1. **Field kept only if κ ≥ 0.6** against both human labelers. The character flag fails (κ 0.000) and is excluded.
2. **Human review trigger:** any prediction where the LLM tier and the model tier differ by 2 or more tiers, or where the LLM marks the player as recognized, goes to a person. In back-tests this flags 11 test reports (0.8%) and catches 5 of 452 errors (precision 0.455).
3. **Broader review trigger:** any disagreement between the LLM tier and the model tier goes to a person. This flags 43% of reports and catches 50% of errors, with precision 0.40 against a base error rate of 0.35. It is too broad to use alone; a person would need to review about 564 of 1,302 test reports.
4. **Subgroup watch:** any subgroup whose error rate is more than 1.25 times the lowest group's (DI > 1.25) is reported. Power-5 schools are currently flagged at 1.38.
5. **Pause condition:** the workflow is paused if test macro-F1 falls below validation macro-F1 minus the MDD (0.029).

**Escalation:** the owner of the workflow reviews flagged items weekly. A breach of rule 1 or 5 pauses the workflow until the owner signs off. The full memo is a separate document.

## 6. Reflection

[Partner 1 — to write: what you learned about the workflow, what surprised you, and what you would change.]

[Partner 2 — to write: the same, from your side of the project.]

Both partners' AI use logs are submitted separately.

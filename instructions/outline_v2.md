# Final Project Outline (v2, draft)

## 1. Research question

**Main question:** Do NFL.com scouting reports tell you anything about where a prospect gets drafted that the numbers don't already tell you?

**Sub-questions:**
1. How well can objective data alone predict draft outcome? (measurables, production and athleticism scores, position, school)
2. How much does the analyst's numeric `grade` add on top of that?
3. How much does the analyst's **text** add, beyond (a) the objective data and (b) the grade?
4. Can an LLM turn that text into reliable structured signals (for example projected round, starter vs. depth, injury or character flags)? Do those signals do better than simple TF-IDF features?
5. Where does the best model miss (by position, year, school tier, or phrases in the report), and what does that say about when to trust the text?

**Hypothesis (from the earlier notebook):** most of what the text says is already captured by `grade`. Once grade is in the model, text adds about 0 (+0.001 AUC, about 1 pick of MAE). Without grade, text adds more (+0.011 AUC). The open question is whether either gap is bigger than measurement noise.

## 2. Use case (Understand)

- **Who acts on it:** a team scouting department or draft media team deciding whether written scouting reports are worth reading and processing at scale, or whether a numbers model is enough.
- **What the AI does:** reads around 500 reports per class and pulls out structured signals and a projected round.
- **What humans still do:** write the reports, check the LLM's labels on a sample, and make the final call on borderline prospects.
- **Why AI might be the wrong tool:** the LLM may already *know* how famous players were drafted (data contamination), and the text may only repeat the analyst's grade.

## 3. Data

- `nfl_prospects_2014_2026.csv`: 6,152 prospect profiles across 13 classes, scraped by `scrape_nfl_prospects.py`.
- **Unit of observation:** one prospect in one draft class.
- **Feature tiers** (these tiers are the core of the design):

| Tier | Columns | Coverage |
|---|---|---|
| Objective | position/positionGroup, height, weight, arm, wingspan, hand, productionScore, athleticismScore, college (Power-5 flag) | measurables ~69%, scores ~67% |
| Analyst numeric | `grade` (and `draftGrade`, which is a derived score; check what it is) | ~96% |
| Analyst text | overview, strengths, weaknesses, (bio, nflComparison) | ~95% |
| Target | draftOverallPick / draftRound (missing = undrafted, ~46%) | 100% |

- **Known data issues:**
  - Author variants ("Lance Zierlein.", "Lance Zierlien", "Round 7/PFA", and others) need cleaning.
  - `bio` is identical to `overview` in about 13% of rows.
  - 2018 has 644 rows but only 479 grades. Check that year for duplicates.
  - Some reports may have been edited after the draft. Scan the text for words like "drafted", "selected", or team names.

## 4. Target and split

- **Primary target:** draft tier, as an ordinal 4-class label: Round 1 / Rounds 2–3 / Rounds 4–7 / Undrafted.
  - This lets us report P/R/F1 and weighted kappa, which the rubric asks for.
- **Secondary target:** log(overall pick) for drafted players only, scored with MAE in picks.
- **Split by year, never randomly.** Train on 2015–2022 (Zierlein era), validate on 2023, test on 2024–2026.
  - 2014 (Nawrocki) is kept out as a separate check, since it is a different analyst.

> **Why tiers instead of the exact pick?** (Use this in the write-up and Q&A.)
> 1. **They follow how the draft actually works.** Day 1 is Round 1, Day 2 is Rounds 2–3, Day 3 is Rounds 4–7. Analysts write the same way ("Day 2 starter", "late-round flier").
> 2. **Undrafted players stay in.** About 46% of prospects go undrafted, and they have no pick number. A pick-number model has to drop them, which biases the sample and loses half the data, or give them a made-up number like "pick 263". Tiers include them naturally.
> 3. **Exact late picks are mostly noise for this question.** Whether a player goes 150th or 180th depends on team needs and trades, not on the scouting report. Tiers keep the signal and drop that noise.
> 4. **The rubric asks for classification metrics.** Tiers give us precision, recall, F1, and weighted kappa. Weighted kappa counts being one tier off as better than being three tiers off.
> 5. **It is a fairer task for the LLM.** "Which day of the draft?" is something a reader can answer from a report. "Which exact pick?" isn't.
> 6. **Trade-off:** we lose detail within a tier, and the classes are uneven (R1 is about 7% of rows). To cover this, we use macro-F1 and keep log-pick MAE on drafted players as a secondary check.

## 5. Notebook structure

0. **Setup.** Imports, seed, paths, and a flag that reads saved LLM outputs instead of calling the API.
1. **Load and clean.** Document each step, with assertions: no duplicate prospect-years, pick between 1 and 262, valid round, text present where expected.
2. **Summary statistics and EDA.** Counts by year and position, draft rate, how grade relates to pick, text length.
3. **Model A: objective only.** Ordinal logit plus gradient boosting.
4. **Model B: objective + grade.**
5. **Model C: text only.** TF-IDF plus logistic regression.
6. **Model D: objective + grade + TF-IDF text.** Also a version without grade.
7. **AI step: LLM on the scouting text.**
   - 7a. **Prompt and settings.** `claude-opus-5-5`, `effort: "low"`, JSON output through structured outputs, run with the Message Batches API. Names, schools, and teams are masked in the text. There is no temperature setting because the current API rejects it. Results are reproducible because the outputs are saved once and re-read, never regenerated.
   - 7b. **Extraction.** For each report: projected tier, starter/depth/special-teams role, injury flag, character flag, overall sentiment (1–5), an evidence quote, and whether it recognized the player. (Self-reported confidence was removed on 2026-11-12 by partner decision.)
   - 7c. **Human validation.** Both partners hand-label the same ~120 reports. Measure human–human kappa (the noise ceiling) and LLM–human kappa per field.
   - 7d. **Contamination check.** Run each report with and without names. Also compare accuracy on older classes vs. 2026, and for famous vs. obscure players.
   - 7e. **Save outputs** to `data/llm_outputs.csv`. Everything later in the notebook reads from this file.
8. **Model E: objective + grade + LLM features.** Also report LLM direct prediction as its own model.
9. **Comparison.** Run every model on the same test set, with bootstrap 95% CIs (1,000 resamples) and the minimum detectable difference. This section answers the research question.
10. **Residual analysis** (replaces "residuals by analyst"; see the note below).
    - Errors of the best model by position, year, school tier, and LLM flag.
    - Which phrases go with large over- or under-predictions?
11. **Cost.** Tokens, dollars, and minutes per class, compared with what each model gains.
12. **Governance thresholds**, computed from the numbers above. These feed the memo.

> **Why not residuals by analyst:** one analyst (Zierlein) wrote about 88% of the reports, covering 2015–2026. Nawrocki only wrote 2014. Analyst and year are fully confounded, so any comparison between analysts would really be a comparison between years.

## 6. Evaluation plan (Evaluate)

| What | Metric | CI method |
|---|---|---|
| Tier models | macro-F1, per-class P/R, quadratic-weighted kappa vs. actual | bootstrap |
| Pick models | MAE (picks), Spearman rho | bootstrap |
| "Does text add?" | Δ metric (D − B, E − B) vs. the minimum detectable difference | paired bootstrap |
| LLM extraction | LLM–human kappa per field; human–human kappa as the ceiling | bootstrap |
| Cost | $ and seconds per 100 reports | n/a |

## 6b. Methods carried over from the weekly coursework

We reuse the course's own methods so graders see the same definitions they taught. Paths are relative to `INFO 4340/`.

| Final-project piece | Course source | What we reuse |
|---|---|---|
| 7c Human validation | `Week 7/Class 12/class11-12-python-textasdata.ipynb` | Codebook with **decision rules** written after reviewing disagreements. Both partners label independently. κ with a 2,000-resample bootstrap CI. Disagreement review. **Partner as the human ceiling.** ~~Accuracy by self-reported confidence~~ (confidence removed 2026-11-12). Weakest-class analysis ("which side did the partner take?"). |
| 7e Caching | same | Run once, cache to CSV, reload on re-run. Store tokens and latency per call. |
| 7f **LLM self-consistency (new)** | `Week 5/Class 8/class8-ask-three-times.ipynb`, Class 14 Part 4 | Send the same ~100 reports 3 times. Measure run-to-run agreement. This is the AI's own noise floor. |
| 9 Comparison | `Week 8/Class 14/class13-14-python-evaluation.ipynb` | **MDD = mean + 1 SD of pairwise rater disagreement** (the course definition, now in `src/evaluation.mdd`). Linear or quadratic weighted κ for ordinal scales. Model-vs-model gaps reported with paired bootstrap CIs. The eval harness pattern: cases → run → code/model/human grader → aggregate → decision. |
| 10 Subgroups | `Week 9/Class 1/class16-python-subgroup.ipynb` | Error rate by position group, Power-5 vs. not, and year. **Disparate impact ratio > 1.25** gets flagged. Chi-square (check expected counts ≥ 5). **Equalized-odds recall spread > 0.20** = unequal. Harm projected at scale. |
| 11 Cost | Class 12 §2.5 | Compare LLM cost vs. human cost ($25/h; about 30 reports/h for 1,300-character reports) vs. a **hybrid with a 10% human audit**. Cost per *correct* label. |
| 12 Governance | `Week 10/class18-python-threshold-backtest.ipynb` | Back-test every rule: **flag rate** and **share of errors caught**, with a minimum n per subgroup, and try 2–3 threshold values. Every rule in the memo gets one line of back-test evidence. |
| Governance memo | `instructions/Week 9_ GenAI Governance Memo.pdf` (assignment was excused, so we write from scratch) | Use its 7 numbered sections as headings: workflow; risk matrix (likelihood 1–5 × impact 1–5 per domain); top-3 rules with measurable triggers, each with one line of back-test evidence; data rules; human review with a quantitative trigger; prohibited uses and escalation (owner → breach → action → timeline); connection to Weeks 6–8 metrics. **AI drafts one domain's rules first, then we rewrite them, and the before/after goes in the AI use log.** |
| AI use log | `Week 8/Project/ai_use_log_instructions.md` | Table with Tool, Task, Prompt, Output, Decision, and Verification columns. One log per partner. |
| Adversarial review (optional) | `Week 5/Analytic Design Critique.../design/` | A fresh-context "adversarial reviewer" critiques the plan before the final run. Logged as an AI interaction. |

## 7. Governance ideas (Govern)

These need numbers once we have results. Baselines (2026-11-11): text adds nothing beyond the grade (D − B = −0.000 macro-F1). MDD from training noise = 0.029 macro-F1 and 0.016 QWK.

- **Contamination:** if accuracy with names exceeds accuracy without names by more than X, only use outputs from the masked version.
- **Extraction reliability:** if LLM–human kappa on a field is below 0.6, drop that field. If weekly spot-check kappa falls below the threshold, pause the workflow.
- **Review triggers** (replacing the removed confidence field): any prediction where the LLM tier and the grade-based tier differ by 2 or more tiers, or where `recognized` = true, goes to a human.
- **Subgroup fairness:** if MAE for any position group or for small schools is more than Z% worse than overall, flag it.
- **Drift:** retrain or recheck each class. Pause if test F1 drops below the validation F1 minus the minimum detectable difference.

## 8. Deliverables and timeline (due Wed Nov 18, 10 pm)

| Date | Work |
|---|---|
| Nov 10–11 | Finalize outline. Clean data. Set up repo layout (`data/`, `src/`, `requirements.txt`, `.env.example`). |
| Nov 12–13 | Models A–D. Write the LLM prompt. Hand-label about 120 reports each. |
| Nov 13–14 | Run the LLM once and save outputs. Compute kappa. Model E. Bootstrap CIs. |
| Nov 15 | Residuals, cost, governance numbers. Notebook runs top to bottom; export to HTML. |
| Nov 16–17 | Summary (3–5 pages), governance memo, slides, README with findings. |
| Nov 18 | Final check from a fresh clone, then submit. AI use logs (one each). |

## 9. LLM budget

Estimates assume about 1,100 input tokens per report (prompt plus about 300 tokens of report text) and about 450 output tokens (JSON plus light thinking). Batch pricing for Opus 5.5 is $2 per million input tokens and $10 per million output tokens, which is half the standard price.

| Run | Calls | Est. cost |
|---|---|---|
| Pilot and prompt iteration (standard API, 52 reports from 2023, **max 3 rounds**) | up to 156 | **$5 cap** (raised from ~$2 on 2026-11-11) |
| Full extraction on 2015–2026, masked (Model E needs training-year features too) | ~5,800 | ~$39 |
| Contamination check, names left in (subset) | 300 | ~$2 |
| Self-consistency: 100 reports × 2 extra runs | 200 | ~$1.50 |
| **Total** | | **~$50, hard cap $60** |

- **Pilot rules (2026-11-11):** at most 3 rounds (round 1 + 2 revisions), and stop early if round 1 is good. Revise only for systematic problems, logging the evidence for each change; bug fixes don't count toward the cap. Why: on 52 items macro-F1 moves about ±0.07 from noise, and each revision fits the prompt to 2023, so a cap written down in advance keeps the test score honest.
- Optional: run Haiku 5.5 on the test set for about $0.50, to compare a cheap model against Opus in the cost analysis.
- Each partner's API key goes in a local `.env` file and is never committed.

## 10. Contract analysis status (kept for now)

The earlier `nfl_draft_contract_analysis.ipynb` stays untouched. **If we drop it, here is why:**
1. **Different question.** It looks at what happens after the draft (second contracts), not whether analyst text predicts the draft.
2. **It breaks reproducibility.** It imports `prep.py` and reads `combined_data_2000-2023.csv`, and neither is in the repo. A fresh clone can't run it, and that hurts both the notebook and the repo rubric rows (about 20 points).
3. **Limited data.** Only the 2014–2019 classes have second contracts (1,147 players). Players drafted later are still on rookie deals.
4. **The text added nothing there.** Adding text gave −0.013 R², so it would not change the story.
5. **Time.** About 8 days remain. That time is better spent on evaluation and governance, which are 4 of the 7 rubric rows.

If we keep it, we need to recover `prep.py` and the contracts data (or document how to get them), and present it clearly as a side analysis.

## 11. Decisions

- [x] Primary target: 4 tiers (Day 1 / Day 2 / Day 3 / UDFA). Log-pick MAE is the secondary check.
- [x] `grade` counts as analyst input. Every model is reported with and without it.
- [x] LLM: Claude Opus 5.5 through the Batch API. Estimated $50 (pilot budget $5), capped at $60.
- [x] Contract analysis: kept for now. The case for dropping it is in section 10.
- [ ] Who does which part? Each partner should own pieces covering at least two competencies, for the presentation Q&A. (Deferred.)
- [ ] Do we run the optional Haiku comparison?
- [x] Week 9 memos: the assignment was excused, so we write the governance memo from scratch using the Week 9 structure (§6b).

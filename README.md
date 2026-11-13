# Does Analyst Text Help Predict the NFL Draft?

INFO 4340 final project. A reproducible, AI-assisted workflow that tests whether NFL.com scouting-report **text** adds predictive information about a prospect's draft outcome beyond **objective data** and the analyst's own numeric **grade**.

## Question
Each NFL.com Draft Tracker profile has three parts: measurables and scores, a numeric grade, and a written report from the analyst. If you already know the objective data and the grade, does the written report improve a prediction of the prospect's **draft tier**?

The tiers are Day 1 (Round 1), Day 2 (Rounds 2–3), Day 3 (Rounds 4–7), and Undrafted. See notebook section 1.5 for why we use tiers instead of the exact pick.

## Workflow
data → cleaning → baselines (numbers / grade / TF-IDF text) → **AI step: Claude extracts structured signals from masked reports** → combined model → evaluation (macro-F1, weighted kappa, bootstrap CIs, minimum detectable difference, cost) → governance thresholds

## Data
| | |
|---|---|
| Source | NFL.com Draft Tracker, scraped with `scrape_nfl_prospects.py` (one request per class) |
| File | `nfl_prospects_2014_2026.csv` (6,152 prospects × 24 columns) |
| Unit of observation | one prospect in one draft class |
| Analysis set | 5,667 prospects with text and grade, excluding specialists |
| Split by year | train 2015–22 · val 2023 · test 2024–26 · 2014 kept out (different analyst) |

## Setup
```bash
git clone https://github.com/WebDeviledEggs/INFO_4340_Final.git
cd INFO_4340_Final
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```
Tested with Python 3.14. You do **not** need an API key to run the notebook, because the LLM outputs are saved in `data/`. You only need one to re-run the LLM step: copy `.env.example` to `.env` and set `ANTHROPIC_API_KEY`.

## How to run
```bash
jupyter nbconvert --to notebook --execute --inplace nfl_draft_text_analysis.ipynb
jupyter nbconvert --to html nfl_draft_text_analysis.ipynb
```
To re-scrape the raw data (optional): `python scrape_nfl_prospects.py`

## Repo layout
| Path | Purpose |
|---|---|
| `nfl_draft_text_analysis.ipynb` / `.html` | Main notebook (the deliverable) |
| `src/cleaning.py` | Cleaning, targets, split by year, text masking |
| `src/evaluation.py` | Bootstrap CIs, paired bootstrap deltas, minimum detectable difference, subgroup and back-test helpers |
| `src/modeling.py` | Feature sets and model pipelines (models A–D, length baseline, stacked text score) |
| `src/labeling.py` | Human labeling sample and Excel workbooks |
| `docs/codebook.md` | Labeling codebook (shared by both partners and the LLM) |
| `data/prospects_clean.parquet` | Cleaned data, written by the notebook |
| `data/labels/` | Labeling workbooks (`labels_partner1.xlsx`, `labels_partner2.xlsx`) and the item key |
| `scrape_nfl_prospects.py` | Data collection |
| `nfl_draft_contract_analysis.ipynb` | Earlier exploratory work on draft and second contracts. **Not part of the main workflow.** It depends on files not in this repo |
| `instructions/` | Assignment instructions and the project outline |

## Findings
**Baselines (notebook section 9, test classes 2024–26, n = 1,302):**

| Model | Macro-F1 (95% CI) | QWK |
|---|---|---|
| A: objective data | 0.578 (0.546–0.610) | 0.700 |
| B: objective + analyst grade | 0.662 (0.633–0.691) | 0.788 |
| D: objective + grade + text (TF-IDF) | 0.662 (0.634–0.691) | 0.784 |

Once the grade is in the model, the text adds no detectable improvement: D − B = −0.000 macro-F1, with a 95% CI of −0.011 to +0.012. The minimum detectable difference is 0.029. The LLM results (Model E) are still to come.

## AI use and cost
_TBD._ Model: `claude-opus-5-5` through the Message Batches API, `effort: "low"`, JSON output. Names, schools, and teams are masked before sending. Estimated about $45 for all runs, with a $60 cap. Actual cost will be reported here.

## Governance
_TBD. Summary of thresholds. The full governance memo is a separate PDF._

## Authors
INFO 4340 project pair, University of Denver.

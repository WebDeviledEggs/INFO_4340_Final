# Codebook: labeling NFL.com scouting reports

Version 1.1 (2026-11-12). Both partners and Claude use this same codebook. **Any change after the practice round goes in the change log at the bottom**, with the date and the disagreement that caused it. This follows the Week 7 Class 12 method: decision rules are written *after* reviewing real disagreements.

## What you see and what you don't

Each item shows the prospect's **position** and the analyst's report (overview, strengths, weaknesses). Names, schools, and NFL teams are replaced with `[PLAYER]`, `[SCHOOL]`, and `[TEAM]`. Claude gets exactly the same input.

- **Label from the text only.** Don't search for the player. If you recognize who it is anyway, label from the text and write `recognized` in `notes`. We use this to measure how much outside knowledge affects labels.
- Don't open `data/labels/label_items_key.csv`. It has the real names.
- Label every item on your own. Don't talk about specific items with your partner until both sheets are done. Independent labels are what make the kappa meaningful.

## Fields

### 1. `projected_tier`: where does the **text** expect this player to be drafted?

| Value | Meaning |
|---|---|
| `R1` | Round 1 (Day 1) |
| `R2-3` | Rounds 2–3 (Day 2) |
| `R4-7` | Rounds 4–7 (Day 3) |
| `UDFA` | Not drafted (priority free agent, camp invite) |

**Decision rules, in order:**
1. **Explicit draft language wins.** "Day 2 pick", "first-round talent", "late-round flier", "priority free agent", "camp body". Fewer than 12% of reports have this, so most items need rule 2.
2. **Otherwise, use the projected NFL role.**
   - "Immediate / Day 1 starter", "high-end starter", "Pro Bowl potential" → `R1`
   - "Should become a starter", "starter in year 2", "early contributor" → `R2-3`
   - "Backup with upside", "rotational", "special teams value", "developmental", "could compete for a roster spot" → `R4-7`
   - "Will need to fight for a roster spot", "camp competition", "practice squad", "tough to project as more than…" → `UDFA`
3. **Ties: look at the weaknesses.** If the weaknesses include a physical-tools concern for the position ("below-average speed for a corner", "lacks size"), go one tier lower.
4. **Position matters only through the text.** Don't move a player up or down because "QBs go early". Label what the analyst says about *this* player.
5. If the report is too short to judge (for example, overview only, under ~3 sentences), label your best guess.

### 2. `role`: the NFL role the text projects

| Value | Use when the text says… |
|---|---|
| `starter` | will start early, Day 1 starter, plug-and-play, immediate impact |
| `eventual_starter` | can become a starter, starter with development, starter in the right scheme |
| `backup` | backup, rotational, sub-package, situational (e.g. "third-down back"), depth |
| `roster_bubble` | special teams only, practice squad, camp competition, fight for a spot |

If the text gives a range ("backup who could become a starter"), take the **lower** end.

### 3. `injury_flag` (0 or 1)

`1` if the text mentions **this player's** injury history, surgery, missed games because of injury, or a durability concern. Examples: "torn ACL in 2019", "durability is a question", "missed four games with a shoulder injury".

`0` if there's no mention, or only wording like "plays through contact" or "tough" with no actual injury.

### 4. `character_flag` (0 or 1)

`1` if the text raises an **off-field or work-habit concern**: arrest, suspension, dismissal, failed test, maturity questions, coachability, "teams will need to vet him".

`0` for praise ("scouts rave about his character", "team captain") and for on-field effort or motor comments ("motor runs hot and cold"). Effort on the field counts as play, not character.

### 5. `tone` (1–5): overall, how positive is the evaluation?

| Value | Meaning |
|---|---|
| 1 | Mostly negative. Weaknesses dominate; the analyst doubts he sticks. |
| 2 | More negative than positive. |
| 3 | Balanced. |
| 4 | More positive than negative. |
| 5 | Glowing. Weaknesses are minor or framed as easy fixes. |

Judge the **overview first**. The strengths and weaknesses lists always contain both sides, so count them only to break a tie.

### 6. `notes` (optional)

Free text. Write `recognized` if you knew who the player was.

## Practice round, then the real round

1. **Practice (items P01–P20, 2023 class).** Both partners label all 20 independently. Then compare, discuss every disagreement, and add decision rules to the change log below. Practice labels are **not** used in kappa.
2. **Evaluation (items E001–E100, 2024–26 classes).** Label independently. **Don't change the codebook during this round.** These 100 items give human–human kappa (the human ceiling) and LLM–human kappa.

Expected time: about 2 minutes per item, so roughly 4 hours each for all 120.

## How the labels are used

| Comparison | Metric | Why |
|---|---|---|
| Partner 1 vs. Partner 2 | weighted κ (tier, role, tone), Cohen's κ (flags), bootstrap CI | Human ceiling: the LLM can't be expected to agree with a human more than humans agree with each other |
| Claude vs. each partner | same | Is the extraction reliable enough to use? Fields below κ = 0.6 are dropped (governance rule) |
| Human disagreement → MDD | mean + 1 SD of \|partner 1 − partner 2\| on tier code and tone | Course minimum detectable difference for comparing LLM outputs |
| Human `projected_tier` vs. the actual draft | macro-F1, QWK | How well a human reader does from the text alone, compared with model C (TF-IDF) and Claude |

## Change log

| Date | Change | Triggered by |
|---|---|---|
| 2026-11-11 | v1.0 drafted from report phrasing in the training years (2015–22) | — |
| 2026-11-12 | v1.1: removed the `confidence` field for both humans and Claude. The column stays in the workbooks but is ignored; leave it blank. | Partner decision after the practice round (one partner had left it blank) |

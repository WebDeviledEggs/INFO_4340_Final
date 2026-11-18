# AI use log: Olivia Case

INFO 4340 final project. Partner: Adam Hoerger. This log covers every AI use in my part of the project (notebook, write-up, slides). The governance memo is my partner's own log.

## Tools

| Tool | Model | Used for |
|---|---|---|
| Claude Code (CLI assistant) | Claude (Haiku 5.5 in this session) | Coding and analysis help, notebook and slide edits, write-up review, memory notes |
| Anthropic API, Message Batches | Claude Opus 5.5 (`claude-opus-5-5`) | Structured labeling of scouting reports (tier, role, injury, tone, character, recognized) |

## Log

| Date | Tool | What I asked | What the AI produced | What I did with it |
|---|---|---|---|---|
| 2026-11-13 | Claude Code | Compare partner labels in `data/labels/` on the practice round; reconcile disagreements | Agreement statistics and a reconciliation sheet; flagged data issues (P04, P16, P20) | I resolved the disagreements myself. I set the agreed values and fixed P04 and P20 in the sheets. The AI did not choose any label. |
| 2026-11-13 | Claude Code | Pull funny lines from the report column for the slides | A list of candidate quotes | I picked the quotes I wanted and sent my own favorites. Both partners' quote lists are in `docs/slide_quotes.md`. |
| 2026-11-13 | Claude Code | Scope and cost of the full labeling batch; whether we need it | Cost estimate and the reasoning for running every analysis report | I decided to run the batch. The main question was reframed around the grade and the four research questions. |
| 2026-11-13 | Claude Code | Clarify the grade columns (grade vs. draftGrade) | Explanation of the two columns and where they come from | I checked the NFL.com definitions and pasted the grade scale into the conversation. |
| 2026-11-14 | Claude Code | Prompt pilot and sensitivity check on 52 validation reports, then the 100-report eval run | Pilot results (tier and role agreement v1 vs. v2 0.962) and the frozen prompt v2 | I chose which prompt version to freeze. I froze v2 after the practice round. |
| 2026-11-14 | Claude Code | Check where the data is stored; keep `data/llm` out of `.gitignore` | Storage check and a note on what is in the repo | I decided to keep the LLM outputs in the repo so they can be reused. |
| 2026-11-14 | Claude Code | Run the full 5,667-report labeling batch through the Batch API | Batch run (batch ID `msgbatch_01EjcjEL4G9tvg8AcmCtiyBb`), $11.62, all 5,667 end_turn | I confirmed the run. I asked for the Batch API specifically to halve the cost. |
| 2026-11-14 | Claude Code | Decide whether to drop the character flag | Options and a recommendation | I accepted the recommendation. The character flag is excluded from Model E. |
| 2026-11-14 | Claude Code | Run the notebook end to end, confirm Model B, draft the write-up | Notebook run, Model B confirmation, first write-up draft | I reviewed the draft and made my own edits. |
| 2026-11-14 | Claude Code | Convert the write-up to Word; suggest better wording than "analyst text" | Word conversion and wording options | I chose my own wording. |
| 2026-11-14 to 2026-11-15 | Claude Code | Score the partners and Model B against the NFL draftGrade, mapped to tiers | Mapping and comparison tables | I checked the mapping and wrote the framing for the slides. The "beat the pro" framing and the light-hearted slide copy were drafted with AI help and edited by me. |
| 2026-11-15 | Claude Code | Explain TF-IDF, macro-F1, and which models answer which research question | Plain-language explanations | I restructured the research questions to RQ1–RQ4 after this discussion. |
| 2026-11-15 | Claude Code | Review and tighten the introduction; explain "not detectable" results | Edit suggestions and an explanation of the MDD | I accepted some edits and rejected others. I wrote the final text. |
| 2026-11-16 | Claude Code | Explain MDD, the risk level, the three review rules, and why Model D was used for the subgroup check | Explanations; a finding that the subgroup check should use Model E | I told the AI to use Model E, because Model E is the model that uses the AI output we are governing. |
| 2026-11-18 | Claude Code | Review the write-up for typos, comprehension, and rubric quality; list proposed edits | 24 suggested edits, each shown for approval | I approved all 24 with my own input on four of them. |
| 2026-11-18 | Claude Code | Run the Model E subgroup analysis; export the notebook to HTML | Updated `src/governance_checks.py`, report, and `nfl_draft_text_analysis.html` | I reviewed the results and the export. |

## Notes on use

- Labels in the analysis set come from Claude Opus 5.5 at batch prices. Every label was produced by the same frozen prompt (v2). The labels were checked against the partners' labels on 100 eval reports. They were not used to train the models.
- Partner labels were produced by the two partners, not by the AI. The AI only compared them and reported agreement.
- Final wording, code choices, and conclusions in the notebook, write-up, and slides are mine. I checked AI-generated numbers against the saved outputs in `docs/` and `data/llm/` before using them.

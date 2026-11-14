# Masking limitations

Decision (2026-11-13): keep all items in the analysis and document these leaks as a limitation. The masking is not changed, because the LLM prompt is frozen at v2 and the same masked text goes to humans and Claude.

## Items with unmasked identifying text (flagged in the eval labeling notes)

| Item | Set | What leaked | Flagged by |
|---|---|---|---|
| E011 | eval | Linebacker coach and defensive coordinator names | Partner 1 |
| E031 | eval | "Tide" (identifies the school) | Partner 1 (noted); Partner 2 (noted as recognized) |
| E042 | eval | School mentioned in the weaknesses section | Partner 1 |
| E069 | eval | "three seasons at Uconn" | Partner 1 |
| E085 | eval | Team name not redacted | Partner 1 |
| E094 | eval | School name not redacted | Partner 1 |

Also known: coach names appear in 25 reports (about 0.4%), including P20 in the practice set. See `project-status-next-steps` in memory.

## How this affects the analysis

- The LLM and both partners see the same masked text, so the leaks affect all three the same way. They don't change the human–human agreement.
- Items that leak can help the LLM recognize a player or school, which may inflate its scores for those items. The write-up should say this is a possible inflation.
- With six leaking items out of 100, the effect on overall metrics should be small. Check it with a sensitivity run that drops them and compares metrics.

## For the write-up

State the leaks, the count (6 of 100 eval items, plus the coach-name reports), and the sensitivity check. Don't claim the masking is complete.

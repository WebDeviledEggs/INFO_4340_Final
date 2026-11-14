# LLM checks (notebook section 7.2-7.7)

## 7.2 Prompt pilot (2023 validation, 52 items)

- items: 52
- v2 end_turn: 52
- tier agreement v1 vs v2: 0.962
- role agreement v1 vs v2: 0.962
- tone QWK v1 vs v2: 0.942
- pilot cost v1 ($): 0.22
- pilot cost v2 ($): 0.22

## 7.3 Full run (analysis set)

- reports: 5667
- stop reasons: {'end_turn': 5667}
- batch cost ($): 11.62
- batch id: msgbatch_01EjcjEL4G9tvg8AcmCtiyBb
- tier counts: {'R4-7': 2988, 'UDFA': 1503, 'R2-3': 937, 'R1': 239}
- role counts: {'backup': 2637, 'roster_bubble': 1823, 'eventual_starter': 1029, 'starter': 178}
- injury_flag rate: 0.14
- character_flag rate (not used): 0.056
- recognized rate: 0.057

## 7.5 Contamination check: recognized vs not

| subset     | recognized   |    n |   tier accuracy vs real tier |
|:-----------|:-------------|-----:|-----------------------------:|
| all        | True         |  322 |                        0.543 |
| all        | False        | 5345 |                        0.527 |
| test split | True         |   40 |                        0.6   |
| test split | False        | 1262 |                        0.536 |

## 7.6 Self-consistency

Not run. A second pass over the 100 eval items would cost about $0.22 at batch prices and would show how stable Claude's labels are. Run it only if the write-up needs it.

## 7.7 Saved outputs

| file                                    |   rows | exists   | holds                                                   |
|:----------------------------------------|-------:|:---------|:--------------------------------------------------------|
| data/llm/pilot_items.csv                |     52 | True     | 52 pilot reports (2023 validation), masked text         |
| data/llm/pilot_v1.csv                   |     52 | True     | pilot run, prompt v1 (confidence field, retired)        |
| data/llm/pilot_v2.csv                   |     52 | True     | pilot run, prompt v2 (frozen)                           |
| data/llm/eval_v2.csv                    |    100 | True     | 100 eval reports, prompt v2 (standard API)              |
| data/llm/batch_v2_analysis.csv          |   5667 | True     | 5,667 analysis reports, prompt v2 (Message Batches API) |
| data/llm/batch_v2_analysis.manifest.csv |   5667 | True     | batch request IDs mapped to row numbers (no names)      |
| data/llm/batch_v2_analysis.batch_id.txt |    nan | True     | batch ID for reattaching to the run                     |

# Subgroups, cost and governance back-tests (sections 10-12)

Test split n = 1302. Model D errors = 452 (0.347). Grade bands are tertiles of grade_pct on the test split, cut points [0.004, 0.336, 0.68, 1.0].

## 10. Subgroup error rates (Model D, test)

### positionGroup

| positionGroup   |   n |   errors |   error_rate |   di_ratio | flag_1.25   |
|:----------------|----:|---------:|-------------:|-----------:|:------------|
| OL              | 239 |       74 |        0.31  |      1     | False       |
| RB              | 105 |       33 |        0.314 |      1.015 | False       |
| QB              |  54 |       17 |        0.315 |      1.017 | False       |
| LB              | 111 |       37 |        0.333 |      1.077 | False       |
| TE              |  85 |       29 |        0.341 |      1.102 | False       |
| DB              | 258 |       92 |        0.357 |      1.152 | False       |
| WR              | 182 |       65 |        0.357 |      1.153 | False       |
| DL              | 268 |      105 |        0.392 |      1.265 | True        |

### chi2 positionGroup

chi-square p = 0.675, dof 7, min expected 18.7

### power5

| power5   |    n |   errors |   error_rate |   di_ratio | flag_1.25   |
|:---------|-----:|---------:|-------------:|-----------:|:------------|
| False    |  245 |       65 |        0.265 |       1    | False       |
| True     | 1057 |      387 |        0.366 |       1.38 | True        |

### chi2 power5

chi-square p = 0.004, dof 1, min expected 85.1

### year_str

|   year_str |   n |   errors |   error_rate |   di_ratio | flag_1.25   |
|-----------:|----:|---------:|-------------:|-----------:|:------------|
|       2024 | 459 |      144 |        0.314 |      1     | False       |
|       2025 | 433 |      157 |        0.363 |      1.156 | False       |
|       2026 | 410 |      151 |        0.368 |      1.174 | False       |

### chi2 year_str

chi-square p = 0.171, dof 2, min expected 142.3

### grade_band

| grade_band   |   n |   errors |   error_rate |   di_ratio | flag_1.25   |
|:-------------|----:|---------:|-------------:|-----------:|:------------|
| low          | 438 |       99 |        0.226 |      1     | False       |
| high         | 430 |      170 |        0.395 |      1.749 | True        |
| mid          | 434 |      183 |        0.422 |      1.866 | True        |

### chi2 grade_band

chi-square p = 0.000, dof 2, min expected 149.3

### equalized odds (positionGroup)

|    |   class |   min_recall |   max_recall |   spread | unequal   |   groups |
|---:|--------:|-------------:|-------------:|---------:|:----------|---------:|
|  0 |       0 |        0.5   |        1     |    0.5   | True      |        7 |
|  1 |       1 |        0.533 |        0.784 |    0.25  | True      |        8 |
|  2 |       2 |        0.411 |        0.696 |    0.285 | True      |        8 |
|  3 |       3 |        0.611 |        0.788 |    0.176 | False     |        8 |

## 11. Cost

| run                                     | API      |        $ |
|:----------------------------------------|:---------|---------:|
| pilot v1 (2023, 52 items)               | standard |  0.22    |
| pilot v2 (2023, 52 items)               | standard |  0.22    |
| eval v2 (2024-26, 100 items)            | standard |  0.43    |
| batch v2 (analysis set, 5,667)          | batch    | 11.62    |
| total spent                             |          | 12.49    |
| batch set at standard prices (not paid) | standard | 23.24    |
| per report, batch                       | batch    |  0.00205 |

## 12. Governance back-tests (review rules vs Model D test errors)

| rule                                  |   flag_rate |   flagged |   errors |   caught |   catch_rate |   precision_of_flag |
|:--------------------------------------|------------:|----------:|---------:|---------:|-------------:|--------------------:|
| 1: abs(Claude tier - model tier) >= 2 |       0.008 |        11 |      452 |        5 |        0.011 |               0.455 |
| 2: Claude tier != model tier          |       0.433 |       564 |      452 |      225 |        0.498 |               0.399 |
| 3: Claude recognized = true           |       0.031 |        40 |      452 |       11 |        0.024 |               0.275 |
| 4: Claude injury_flag = 1             |       0.095 |       124 |      452 |       49 |        0.108 |               0.395 |

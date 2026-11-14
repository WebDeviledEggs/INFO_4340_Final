# Final model comparison (notebook section 8)

Protocol: C chosen on the 2023 validation split; 2024–26 test split scored once; paired bootstrap, 1000 resamples. The 2014 holdout is not used.

## Coverage of draftGrade (analysis rows)

4071 of 5667 analysis rows have a draftGrade. Share with a draftGrade by class year:

|                       |   2014 |   2015 |   2016 |   2017 |   2018 |   2019 |   2020 |   2021 |   2022 |   2023 |   2024 |   2025 |   2026 |
|:----------------------|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|
| share_with_draftGrade |   0.72 |   0.81 |   0.83 |    0.8 |   0.69 |   0.67 |   0.66 |    0.7 |   0.69 |   0.65 |   0.69 |   0.75 |   0.76 |

## Q3: draftGrade vs grade (test rows with both scores, n = 950)

| model                             |    C |   macro_f1 |   qwk |   accuracy |
|:----------------------------------|-----:|-----------:|------:|-----------:|
| A objective                       |  0.1 |      0.536 | 0.624 |      0.524 |
| B objective + grade               | 10   |      0.633 | 0.751 |      0.608 |
| N objective + draftGrade          |  1   |      0.562 | 0.697 |      0.539 |
| BN objective + grade + draftGrade |  0.1 |      0.616 | 0.742 |      0.587 |
| grade alone                       |  0.1 |      0.582 | 0.735 |      0.558 |
| draftGrade alone                  | 10   |      0.559 | 0.697 |      0.545 |

Paired differences (test, macro-F1):

| comparison                                  |   delta |   ci_low |   ci_high |
|:--------------------------------------------|--------:|---------:|----------:|
| draftGrade alone − grade alone              |  -0.023 |   -0.063 |     0.015 |
| N − B (draftGrade vs grade, with objective) |  -0.071 |   -0.105 |    -0.038 |
| BN − B (add draftGrade to grade)            |  -0.017 |   -0.045 |     0.009 |

## Q4: Claude fields and text model (all analysis rows, test n = 1302)

| model   |    C |   macro_f1 |   qwk |   accuracy |
|:--------|-----:|-----------:|------:|-----------:|
| B       | 10   |      0.662 | 0.788 |      0.657 |
| E       |  0.1 |      0.663 | 0.783 |      0.657 |
| D       |  1   |      0.661 | 0.785 |      0.653 |

Paired differences (test, macro-F1):

| comparison                                     |   delta |   ci_low |   ci_high |
|:-----------------------------------------------|--------:|---------:|----------:|
| E − B (add Claude fields to objective + grade) |   0.001 |   -0.014 |     0.017 |
| D − B (add text model to objective + grade)    |  -0.001 |   -0.013 |     0.012 |
| E − D (Claude fields vs text model)            |   0.002 |   -0.014 |     0.019 |

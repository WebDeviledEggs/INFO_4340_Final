"""Formal model comparison for research questions 3 and 4 (notebook section 8).

Q3: does the NFL model score (draftGrade) predict draft tier better than the analyst grade?
    Scored on the rows that have BOTH scores, so every model sees the same prospects.
Q4: does Claude's reading of the report add predictive power? Model E = objective + grade +
    the Claude fields in llm.CLAUDE_FEATURE_FIELDS (character_flag excluded). Compared with B
    (same without Claude) and D (text model probabilities).

Protocol: C is chosen on the 2023 validation split; the 2024-26 test split is scored once.
The 2014 holdout (holdout_nawrocki) is not used here. Paired bootstrap (1,000 resamples) on the
test split gives the CI for each difference.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import cleaning as C
from . import evaluation as E
from . import llm
from . import modeling as M

BATCH_CSV = C.ROOT / "data" / "llm" / "batch_v2_analysis.csv"
REPORT_MD = C.ROOT / "docs" / "final_model_results.md"
C_GRID = [0.1, 1.0, 10.0]
N_BOOT = 1000
CLAUDE_TIERS = ["R1", "R2-3", "R4-7", "UDFA"]
CLAUDE_ROLES = ["starter", "eventual_starter", "backup", "roster_bubble"]
CLAUDE_COLS = ([f"cl_tier_{t}" for t in CLAUDE_TIERS] + [f"cl_role_{r}" for r in CLAUDE_ROLES]
               + ["cl_injury", "cl_tone"])


def load() -> pd.DataFrame:
    d = M.add_model_columns(pd.read_parquet(C.ROOT / "data" / "prospects_clean.parquet"))
    a = d[d.in_analysis.astype(bool)].copy()
    # Same within-class percentile treatment as grade_pct; ranks only over rows with a draftGrade.
    a["draft_pct"] = a.groupby("year").draftGrade.rank(pct=True)

    cl = pd.read_csv(BATCH_CSV)
    assert not cl.duplicated(C.KEY).any(), "batch file has duplicate player keys"
    cl = cl.rename(columns={"projected_tier": "cl_tier", "role": "cl_role",
                            "injury_flag": "cl_injury", "tone": "cl_tone"})
    cl = cl[C.KEY + ["cl_tier", "cl_role", "cl_injury", "cl_tone"]]
    a = a.merge(cl, on=C.KEY, how="left", validate="one_to_one")
    assert a.cl_tier.notna().all(), "analysis rows without a Claude label"
    for t in CLAUDE_TIERS:
        a[f"cl_tier_{t}"] = (a.cl_tier == t).astype(float)
    for r in CLAUDE_ROLES:
        a[f"cl_role_{r}"] = (a.cl_role == r).astype(float)
    a["cl_injury"] = a.cl_injury.astype(float)
    a["cl_tone"] = a.cl_tone.astype(float)
    return a


def tune_fit(num_cols, tr, va, te):
    """Fit a class-weighted logit for each C in C_GRID; keep the best macro-F1 on validation."""
    best = None
    for c in C_GRID:
        m = M.tabular_model(num_cols, C_reg=c).fit(tr, tr.tier_code)
        f1 = E.macro_f1(va.tier_code.values, m.predict(va))
        if best is None or f1 > best[0]:
            best = (f1, c, m)
    _, c, m = best
    return c, m.predict(te)


def summarize(name, y, yhat_by_model, c_by_model):
    rows = []
    for k, yhat in yhat_by_model.items():
        s = M.score_tiers(y, yhat)
        rows.append({"model": k, "C": c_by_model[k], "macro_f1": round(s["macro_f1"], 3),
                     "qwk": round(s["qwk"], 3), "accuracy": round(s["accuracy"], 3)})
    return pd.DataFrame(rows)


def paired(label, metric, y, yhat_a, yhat_b):
    r = E.paired_bootstrap_delta(metric, y, yhat_a, yhat_b, n_boot=N_BOOT)
    return {"comparison": label, "delta": round(r["delta"], 3),
            "ci_low": round(r["ci_low"], 3), "ci_high": round(r["ci_high"], 3)}


def main() -> None:
    a = load()
    tr_all, va_all, te_all = a[a.split == "train"], a[a.split == "val"], a[a.split == "test"]
    out_lines = ["# Final model comparison (notebook section 8)", "",
                 "Protocol: C chosen on the 2023 validation split; 2024–26 test split scored once; "
                 f"paired bootstrap, {N_BOOT} resamples. The 2014 holdout is not used.", ""]

    # ---- Q3: draftGrade vs grade, on rows with both scores
    has = a.draft_pct.notna()
    cov = a.assign(has_draft=has).groupby("year").has_draft.mean().round(2)
    out_lines += ["## Coverage of draftGrade (analysis rows)", "",
                  f"{int(has.sum())} of {len(a)} analysis rows have a draftGrade. Share with a draftGrade by class year:", "",
                  cov.to_frame("share_with_draftGrade").T.to_markdown(), ""]
    c3 = a[has]
    tr, va, te = c3[c3.split == "train"], c3[c3.split == "val"], c3[c3.split == "test"]
    specs3 = {
        "A objective": M.OBJ_NUM,
        "B objective + grade": M.OBJ_NUM + ["grade_pct"],
        "N objective + draftGrade": M.OBJ_NUM + ["draft_pct"],
        "BN objective + grade + draftGrade": M.OBJ_NUM + ["grade_pct", "draft_pct"],
        "grade alone": ["grade_pct"],
        "draftGrade alone": ["draft_pct"],
    }
    preds3, cs3 = {}, {}
    for k, cols in specs3.items():
        cs3[k], preds3[k] = tune_fit(cols, tr, va, te)
    y3 = te.tier_code.values
    out_lines += [f"## Q3: draftGrade vs grade (test rows with both scores, n = {len(te)})", "",
                  summarize("Q3", y3, preds3, cs3).to_markdown(index=False), "",
                  "Paired differences (test, macro-F1):", ""]
    q3 = [paired("draftGrade alone − grade alone", E.macro_f1, y3, preds3["grade alone"], preds3["draftGrade alone"]),
          paired("N − B (draftGrade vs grade, with objective)", E.macro_f1, y3, preds3["B objective + grade"], preds3["N objective + draftGrade"]),
          paired("BN − B (add draftGrade to grade)", E.macro_f1, y3, preds3["B objective + grade"], preds3["BN objective + grade + draftGrade"])]
    out_lines += [pd.DataFrame(q3).to_markdown(index=False), ""]

    # ---- Q4: Claude fields (E) vs B and D, on all analysis rows
    tr, va, te = tr_all, va_all, te_all
    preds4, cs4 = {}, {}
    cs4["B"], preds4["B"] = tune_fit(M.OBJ_NUM + ["grade_pct"], tr, va, te)
    cs4["E"], preds4["E"] = tune_fit(M.OBJ_NUM + ["grade_pct"] + CLAUDE_COLS, tr, va, te)
    probs = M.text_probabilities(tr, {"val": va, "test": te}, C_reg=1.0)
    tr_d = M.attach_text_probs(tr, probs["train"])
    va_d = M.attach_text_probs(va, probs["val"])
    te_d = M.attach_text_probs(te, probs["test"])
    cs4["D"], preds4["D"] = tune_fit(M.OBJ_NUM + ["grade_pct"] + M.TEXT_PROBS, tr_d, va_d, te_d)
    y4 = te.tier_code.values
    out_lines += [f"## Q4: Claude fields and text model (all analysis rows, test n = {len(te)})", "",
                  summarize("Q4", y4, preds4, cs4).to_markdown(index=False), "",
                  "Paired differences (test, macro-F1):", ""]
    q4 = [paired("E − B (add Claude fields to objective + grade)", E.macro_f1, y4, preds4["B"], preds4["E"]),
          paired("D − B (add text model to objective + grade)", E.macro_f1, y4, preds4["B"], preds4["D"]),
          paired("E − D (Claude fields vs text model)", E.macro_f1, y4, preds4["D"], preds4["E"])]
    out_lines += [pd.DataFrame(q4).to_markdown(index=False), ""]

    REPORT_MD.write_text("\n".join(out_lines), encoding="utf-8")
    print("\n".join(out_lines))
    print(f"\nwritten: {REPORT_MD}")


if __name__ == "__main__":
    main()

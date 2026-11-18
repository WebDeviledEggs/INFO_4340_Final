"""Notebook sections 10-12: subgroup errors, cost, and governance back-tests.

Errors are judged on the 2024-26 test split, using Model D (objective + grade + text probabilities)
and Model E (adds the Claude fields). Both are tuned on 2023 validation, as in final_comparison.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import cleaning as C
from . import evaluation as E
from . import final_comparison as FC
from . import llm
from . import modeling as M

LLM_DIR = C.ROOT / "data" / "llm"
REPORT_MD = C.ROOT / "docs" / "subgroups_costs_governance.md"


def test_frame() -> pd.DataFrame:
    """Test split with model predictions, Claude fields, and the batch 'recognized' flag."""
    a = FC.load()
    assert list(C.TIERS) == ["R1", "R2-3", "R4-7", "UDFA"]
    code_of = {k: i for i, k in enumerate(C.TIERS)}
    assert (a.tier.astype(str).map(code_of).values == a.tier_code.values).all(), "tier_code != TIERS index"
    rec = pd.read_csv(LLM_DIR / "batch_v2_analysis.csv")[C.KEY + ["recognized"]]
    a = a.merge(rec, on=C.KEY, how="left", validate="one_to_one")
    tr, va, te = a[a.split == "train"], a[a.split == "val"], a[a.split == "test"]

    probs = M.text_probabilities(tr, {"val": va, "test": te}, C_reg=1.0)
    tr_d = M.attach_text_probs(tr, probs["train"])
    va_d = M.attach_text_probs(va, probs["val"])
    te_d = M.attach_text_probs(te, probs["test"])
    _, pred_d = FC.tune_fit(M.OBJ_NUM + ["grade_pct"] + M.TEXT_PROBS, tr_d, va_d, te_d)
    _, pred_e = FC.tune_fit(M.OBJ_NUM + ["grade_pct"] + FC.CLAUDE_COLS, tr, va, te)

    t = te_d.copy()
    t["pred_D"] = pred_d
    t["pred_E"] = pred_e
    t["cl_tier_code"] = t.cl_tier.map({k: i for i, k in enumerate(C.TIERS)})
    t["grade_band"] = pd.qcut(t.grade_pct, 3, labels=["low", "mid", "high"])
    t["year_str"] = t.year.astype(str)
    return t


# ---------------------------------------------------------------- section 10: subgroups
def subgroup_tables(t: pd.DataFrame, pred_col: str = "pred_E") -> dict:
    """Subgroup error tables for one model. Model E (the AI-using workflow) is the primary check;
    Model D is kept as a comparison."""
    pred = t[pred_col]
    err = (pred != t.tier_code)
    out = {}
    for col in ["positionGroup", "power5", "year_str", "grade_band"]:
        tab = E.subgroup_errors(err, t[col].astype(str))
        out[col] = tab
        out[f"chi2 {col}"] = E.chi_square_errors(err, t[col].astype(str))
    odds = E.equalized_odds_recall(pd.Series(t.tier_code.values), pd.Series(pred.values),
                                   pd.Series(t.positionGroup.astype(str).values))
    out["equalized odds (positionGroup)"] = odds
    return out


# ---------------------------------------------------------------- section 11: cost
def cost_table() -> pd.DataFrame:
    """Every LLM run so far, at the prices actually paid, plus the cost the same batch would have
    had at standard prices."""
    pilot1 = pd.read_csv(LLM_DIR / "pilot_v1.csv")
    pilot2 = pd.read_csv(LLM_DIR / "pilot_v2.csv")
    ev = pd.read_csv(LLM_DIR / "eval_v2.csv")
    b = pd.read_csv(LLM_DIR / "batch_v2_analysis.csv")
    rows = [
        {"run": "pilot v1 (2023, 52 items)", "API": "standard", "$": round(llm.cost(pilot1, "standard"), 2)},
        {"run": "pilot v2 (2023, 52 items)", "API": "standard", "$": round(llm.cost(pilot2, "standard"), 2)},
        {"run": "eval v2 (2024-26, 100 items)", "API": "standard", "$": round(llm.cost(ev, "standard"), 2)},
        {"run": "batch v2 (analysis set, 5,667)", "API": "batch", "$": round(llm.cost(b, "batch"), 2)},
    ]
    total = sum(r["$"] for r in rows)
    rows.append({"run": "total spent", "API": "", "$": round(total, 2)})
    rows.append({"run": "batch set at standard prices (not paid)", "API": "standard",
                 "$": round(llm.cost(b, "standard"), 2)})
    rows.append({"run": "per report, batch", "API": "batch",
                 "$": round(llm.cost(b, "batch") / len(b), 5)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- section 12: governance back-tests
def backtest_rules(t: pd.DataFrame) -> pd.DataFrame:
    """Three review rules, each with a numeric threshold, scored against Model D's test errors.

    Rule 1: review if Claude's tier is 2 or more tiers from the model's tier (|diff| >= 2).
    Rule 2: review if Claude's tier differs from the model's tier at all.
    Rule 3: review if Claude marks the player as recognized.
    Rule 4: review if Claude's injury flag is set (1).
    """
    err = pd.Series(t.pred_D.values != t.tier_code.values)
    rules = {
        "1: abs(Claude tier - model tier) >= 2": pd.Series(np.abs(t.cl_tier_code.values - t.pred_D.values) >= 2),
        "2: Claude tier != model tier": pd.Series(t.cl_tier_code.values != t.pred_D.values),
        "3: Claude recognized = true": t.recognized.fillna(False).astype(bool).reset_index(drop=True),
        "4: Claude injury_flag = 1": pd.Series(t.cl_injury.values == 1),
    }
    rows = []
    for name, flag in rules.items():
        r = E.backtest_flag(flag.reset_index(drop=True), err.reset_index(drop=True))
        rows.append({"rule": name, **{k: round(v, 3) if isinstance(v, float) else v for k, v in r.items()}})
    return pd.DataFrame(rows)


def main() -> None:
    t = test_frame()
    # the grade-band tercile edges come from the test split itself; report them
    edges = pd.qcut(t.grade_pct, 3, retbins=True)[1]
    st = subgroup_tables(t, "pred_E")
    st_d = subgroup_tables(t, "pred_D")
    cost = cost_table()
    bt = backtest_rules(t)

    lines = ["# Subgroups, cost and governance back-tests (sections 10-12)", "",
             f"Test split n = {len(t)}. Model D errors = {int((t.pred_D != t.tier_code).sum())} "
             f"({(t.pred_D != t.tier_code).mean():.3f}). Grade bands are tertiles of grade_pct on the test "
             f"split, cut points {np.round(edges, 3).tolist()}.", "",
             f"## 10. Subgroup error rates (Model E, test; Model D for comparison)", "",
             f"Model E errors = {int((t.pred_E != t.tier_code).sum())} ({(t.pred_E != t.tier_code).mean():.3f}).", ""]
    for label, table in [("Model E", st), ("Model D (comparison)", st_d)]:
        lines += [f"### {label}", ""]
        for key, val in table.items():
            lines += [f"#### {key}", ""]
            if isinstance(val, dict):
                lines += [f"chi-square p = {val['p']:.3f}, dof {val['dof']}, min expected {val['min_expected']:.1f}", ""]
            else:
                lines += [val.round(3).to_markdown(), ""]
    lines += ["## 11. Cost", "", cost.to_markdown(index=False), "",
              "## 12. Governance back-tests (review rules vs Model D test errors)", "",
              bt.to_markdown(index=False), ""]
    REPORT_MD.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nwritten: {REPORT_MD}")


if __name__ == "__main__":
    main()

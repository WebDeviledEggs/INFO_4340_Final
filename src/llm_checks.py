"""Checks behind notebook section 7.2-7.7: pilot, full run, contamination, saved outputs.

Self-consistency (7.6) is not run here: it needs a second full pass over the labels, which costs
money, so it is reported as not run. See docs/llm_checks.md.
"""
from __future__ import annotations

import pandas as pd

from . import cleaning as C
from . import evaluation as E
from . import llm

LLM_DIR = C.ROOT / "data" / "llm"
BATCH_CSV = LLM_DIR / "batch_v2_analysis.csv"
REPORT_MD = C.ROOT / "docs" / "llm_checks.md"


def pilot_v1_v2() -> dict:
    """7.2: prompt pilot. v1 (with confidence) and v2 (frozen) on the same 52 items."""
    v1 = pd.read_csv(LLM_DIR / "pilot_v1.csv")
    v2 = pd.read_csv(LLM_DIR / "pilot_v2.csv")
    m = v1.merge(v2, on=C.KEY, suffixes=("_v1", "_v2"))
    return {
        "items": len(m),
        "v2 end_turn": int((v2.stop_reason == "end_turn").sum()),
        "tier agreement v1 vs v2": round((m.projected_tier_v1 == m.projected_tier_v2).mean(), 3),
        "role agreement v1 vs v2": round((m.role_v1 == m.role_v2).mean(), 3),
        "tone QWK v1 vs v2": round(E.qwk(m.tone_v1.astype(int), m.tone_v2.astype(int)), 3),
        "pilot cost v1 ($)": round(llm.cost(v1, "standard"), 2),
        "pilot cost v2 ($)": round(llm.cost(v2, "standard"), 2),
    }


def full_run_summary(b: pd.DataFrame) -> dict:
    """7.3: the batch run over the analysis set."""
    return {
        "reports": len(b),
        "stop reasons": b.stop_reason.value_counts().to_dict(),
        "batch cost ($)": round(llm.cost(b, "batch"), 2),
        "batch id": b.batch_id.iloc[0],
        "tier counts": b.projected_tier.value_counts().to_dict(),
        "role counts": b.role.value_counts().to_dict(),
        "injury_flag rate": round(b.injury_flag.mean(), 3),
        "character_flag rate (not used)": round(b.character_flag.mean(), 3),
        "recognized rate": round(b.recognized.mean(), 3),
    }


def contamination(b: pd.DataFrame) -> pd.DataFrame:
    """7.5: does Claude do better on prospects it says it recognizes?

    Accuracy is Claude's tier against the real drafted tier, so a recognized player only helps if
    Claude is using outside knowledge of the outcome.
    """
    d = pd.read_parquet(C.ROOT / "data" / "prospects_clean.parquet")[C.KEY + ["tier", "split"]]
    m = b.merge(d, on=C.KEY, how="left", validate="one_to_one")
    m["match"] = m.projected_tier == m.tier
    rows = []
    for label, sub in [("all", m), ("test split", m[m.split == "test"])]:
        for flag in [True, False]:
            s = sub[sub.recognized.astype(bool) == flag]
            rows.append({"subset": label, "recognized": flag, "n": len(s),
                         "tier accuracy vs real tier": round(s.match.mean(), 3) if len(s) else None})
    return pd.DataFrame(rows)


def saved_outputs() -> pd.DataFrame:
    """7.7: every file the LLM step writes, with what it holds."""
    items = [
        ("data/llm/pilot_items.csv", "52 pilot reports (2023 validation), masked text"),
        ("data/llm/pilot_v1.csv", "pilot run, prompt v1 (confidence field, retired)"),
        ("data/llm/pilot_v2.csv", "pilot run, prompt v2 (frozen)"),
        ("data/llm/eval_v2.csv", "100 eval reports, prompt v2 (standard API)"),
        ("data/llm/batch_v2_analysis.csv", "5,667 analysis reports, prompt v2 (Message Batches API)"),
        ("data/llm/batch_v2_analysis.manifest.csv", "batch request IDs mapped to row numbers (no names)"),
        ("data/llm/batch_v2_analysis.batch_id.txt", "batch ID for reattaching to the run"),
    ]
    rows = []
    for p, what in items:
        path = C.ROOT / p
        rows.append({"file": p, "rows": (len(pd.read_csv(path)) if path.suffix == ".csv" and path.exists() else None),
                     "exists": path.exists(), "holds": what})
    return pd.DataFrame(rows)


def main() -> None:
    b = pd.read_csv(BATCH_CSV)
    p = pilot_v1_v2()
    f = full_run_summary(b)
    c = contamination(b)
    s = saved_outputs()
    lines = ["# LLM checks (notebook section 7.2-7.7)", "",
             "## 7.2 Prompt pilot (2023 validation, 52 items)", ""]
    lines += [f"- {k}: {v}" for k, v in p.items()]
    lines += ["", "## 7.3 Full run (analysis set)", ""]
    lines += [f"- {k}: {v}" for k, v in f.items()]
    lines += ["", "## 7.5 Contamination check: recognized vs not", "",
              c.to_markdown(index=False), "",
              "## 7.6 Self-consistency", "",
              "Not run. A second pass over the 100 eval items would cost about $0.22 at batch prices "
              "and would show how stable Claude's labels are. Run it only if the write-up needs it.", "",
              "## 7.7 Saved outputs", "", s.to_markdown(index=False), ""]
    REPORT_MD.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nwritten: {REPORT_MD}")


if __name__ == "__main__":
    main()

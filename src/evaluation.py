"""Evaluation and governance helpers.

Methods follow the INFO 4340 course notebooks so results are comparable to the weekly work:
- Week 7 (Class 13/14): kappa, bootstrap CIs, minimum detectable difference = mean + 1 SD of
  pairwise rater disagreement.
- Week 9 (Class 16): subgroup error rates, disparate impact ratio (1.25 rule for error rates),
  chi-square, equalized-odds recall spread (> 0.20 = unequal).
- Week 10 (Class 18): threshold back-tests (flag rate + share of errors caught).
"""
from __future__ import annotations

from itertools import combinations
from typing import Callable

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import cohen_kappa_score, f1_score

Metric = Callable[[np.ndarray, np.ndarray], float]


# ---------------------------------------------------------------- metrics
def macro_f1(y, yhat) -> float:
    return f1_score(y, yhat, average="macro", zero_division=0)


def qwk(y, yhat) -> float:
    """Quadratic-weighted kappa: ordinal agreement (being 1 tier off costs less than 3)."""
    return cohen_kappa_score(y, yhat, weights="quadratic")


# ---------------------------------------------------------------- uncertainty
def _boot_idx(n: int, n_boot: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, n, size=(n_boot, n))


def bootstrap_ci(metric: Metric, y, yhat, n_boot: int = 1000, seed: int = 4340, alpha: float = 0.05):
    """Point estimate + percentile CI from resampling the evaluation set with replacement."""
    y, yhat = np.asarray(y), np.asarray(yhat)
    s = np.array([metric(y[i], yhat[i]) for i in _boot_idx(len(y), n_boot, seed)])
    lo, hi = np.nanpercentile(s, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"estimate": metric(y, yhat), "ci_low": lo, "ci_high": hi, "n": len(y)}


def paired_bootstrap_delta(metric: Metric, y, yhat_a, yhat_b, n_boot: int = 1000, seed: int = 4340,
                           alpha: float = 0.05):
    """CI for metric(B) - metric(A) on the SAME resampled rows. CI excluding 0 = B really differs from A."""
    y, a, b = map(np.asarray, (y, yhat_a, yhat_b))
    d = np.array([metric(y[i], b[i]) - metric(y[i], a[i]) for i in _boot_idx(len(y), n_boot, seed)])
    lo, hi = np.nanpercentile(d, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"delta": metric(y, b) - metric(y, a), "ci_low": lo, "ci_high": hi}


def mdd(diffs, k_sd: float = 1.0) -> float:
    """Course minimum detectable difference: mean + k_sd * SD of |rater-vs-rater| disagreements.

    `diffs` are absolute gaps between two raters (or two runs of the same model) scoring the
    same item on the same scale. A gap between outputs smaller than this is within rater noise.
    """
    diffs = np.abs(np.asarray(diffs, dtype=float))
    return float(diffs.mean() + k_sd * diffs.std())


def pairwise_diffs(ratings: pd.DataFrame) -> np.ndarray:
    """|r1 - r2| for every pair of rater columns and every row (item) both rated."""
    out = []
    for r1, r2 in combinations(ratings.columns, 2):
        both = ratings[[r1, r2]].dropna()
        out.extend((both[r1] - both[r2]).abs())
    return np.asarray(out)


# ---------------------------------------------------------------- subgroups (Week 9)
def subgroup_errors(is_error: pd.Series, group: pd.Series) -> pd.DataFrame:
    """n, errors, error rate per group + disparate impact ratio vs. the best (lowest-error) group."""
    t = is_error.groupby(group, observed=True).agg(n="size", errors="sum")
    t["error_rate"] = t["errors"] / t["n"]
    ref = t["error_rate"].min()
    t["di_ratio"] = t["error_rate"] / ref if ref > 0 else np.inf
    t["flag_1.25"] = t["di_ratio"] > 1.25
    return t.sort_values("error_rate")


def chi_square_errors(is_error: pd.Series, group: pd.Series) -> dict:
    """Is correct/incorrect independent of group? (Unreliable when expected counts < 5.)"""
    table = pd.crosstab(group, is_error)
    chi2, p, dof, expected = stats.chi2_contingency(table)
    return {"chi2": chi2, "dof": dof, "p": p, "min_expected": float(expected.min())}


def equalized_odds_recall(y_true: pd.Series, y_pred: pd.Series, group: pd.Series,
                          min_support: int = 3, max_spread: float = 0.20) -> pd.DataFrame:
    """Simplified equalized odds: per-class recall in each group; spread > max_spread = unequal."""
    rows = []
    for cls in sorted(pd.unique(y_true)):
        rec = {}
        for g in pd.unique(group):
            m = (group == g) & (y_true == cls)
            if m.sum() >= min_support:
                rec[g] = float((y_pred[m] == cls).mean())
        if len(rec) >= 2:
            spread = max(rec.values()) - min(rec.values())
            rows.append({"class": cls, "min_recall": min(rec.values()), "max_recall": max(rec.values()),
                         "spread": spread, "unequal": spread > max_spread, "groups": len(rec)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- threshold back-tests (Week 10)
def backtest_flag(flag: pd.Series, is_error: pd.Series) -> dict:
    """How often a review rule fires, and how many of the actual errors it catches."""
    flag, is_error = flag.astype(bool), is_error.astype(bool)
    n_err = int(is_error.sum())
    caught = int((flag & is_error).sum())
    return {"flag_rate": float(flag.mean()), "flagged": int(flag.sum()), "errors": n_err,
            "caught": caught, "catch_rate": caught / n_err if n_err else np.nan,
            "precision_of_flag": caught / flag.sum() if flag.sum() else np.nan}

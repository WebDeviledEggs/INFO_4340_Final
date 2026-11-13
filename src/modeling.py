"""Feature sets and model pipelines for the baseline models (notebook sections 3-6, 9).

Every tier model is a class-weighted multinomial logistic regression so the only thing that
changes between models A-D is the information they get. Gradient boosting is a robustness check.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import cleaning as C
from . import evaluation as E

SEED = 4340

# ---------------------------------------------------------------- feature sets
MEASURABLES = ["height", "weight", "armLength", "wingspan", "handSize"]
NGS_SCORES = ["productionScore", "athleticismScore"]   # NFL Next Gen Stats draft model scores
OBJ_NUM = MEASURABLES + NGS_SCORES
OBJ_CAT = ["positionGroup"]
OBJ_BOOL = ["power5"]
GRADE = ["grade_pct"]                                   # analyst grade, percentile within class
LENGTH = ["log_len_overview", "log_len_strengths", "log_len_weaknesses"]
TEXT_PROBS = ["text_p_R1", "text_p_R2-3", "text_p_R4-7"]   # out-of-fold text-model probabilities

# Tabular models: name -> (numeric columns, include position + Power-5?)
# C (text only) is the TF-IDF model itself; D0/D add its out-of-fold probabilities.
TABULAR_SPECS = {
    "A":   (OBJ_NUM, True),                          # objective
    "B":   (OBJ_NUM + GRADE, True),                  # objective + grade
    "L":   (LENGTH, False),                          # report length only
    "B+L": (OBJ_NUM + GRADE + LENGTH, True),         # is text's value just length?
    "D0":  (OBJ_NUM + TEXT_PROBS, True),             # objective + text (no grade)
    "D":   (OBJ_NUM + GRADE + TEXT_PROBS, True),     # objective + grade + text
}
MODEL_LABELS = {
    "A": "A: objective", "B": "B: objective + grade", "C": "C: text only (TF-IDF)",
    "L": "L: report length only", "B+L": "B+L: objective + grade + length",
    "D0": "D0: objective + text", "D": "D: objective + grade + text",
}


def add_model_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Masked text (same input the LLM will see) and per-section length features."""
    out = df.copy()
    out["text_masked"] = [C.mask_text(t, n, s) for t, n, s in
                          zip(out.scout_text, out.displayName, out.collegeNames)]
    for f in C.TEXT_FIELDS:
        out[f"log_len_{f}"] = np.log1p(out[f].fillna("").str.len())
    return out


def tabular_preprocessor(num_cols: list[str], with_position: bool = True,
                         indicators: bool = True) -> ColumnTransformer:
    """Median-impute + missing-value indicators + scale; one-hot position; Power-5 flag."""
    parts = [("num", make_pipeline(SimpleImputer(strategy="median", add_indicator=indicators),
                                   StandardScaler()), num_cols)]
    if with_position:
        parts += [("pos", OneHotEncoder(handle_unknown="ignore"), OBJ_CAT),
                  ("p5", "passthrough", OBJ_BOOL)]
    return ColumnTransformer(parts)


def tabular_model(num_cols: list[str], C_reg: float = 1.0, with_position: bool = True,
                  kind: str = "logit", indicators: bool = True) -> Pipeline:
    if kind == "logit":
        clf = LogisticRegression(C=C_reg, class_weight="balanced", max_iter=5000)
    elif kind == "gbm":
        clf = HistGradientBoostingClassifier(learning_rate=0.05, max_iter=300, max_leaf_nodes=15,
                                             class_weight="balanced", random_state=SEED)
    else:
        raise ValueError(kind)
    return Pipeline([("prep", tabular_preprocessor(num_cols, with_position, indicators)), ("clf", clf)])


def text_model(C_reg: float = 1.0) -> Pipeline:
    """TF-IDF (words + bigrams) -> class-weighted logistic regression. No stop-word list:
    'not', 'lacks', 'no' carry meaning in scouting text."""
    return Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=5, max_df=0.9, sublinear_tf=True,
                                  max_features=30_000)),
        ("clf", LogisticRegression(C=C_reg, class_weight="balanced", max_iter=5000)),
    ])


def text_probabilities(train: pd.DataFrame, others: dict[str, pd.DataFrame], C_reg: float,
                       text_col: str = "text_masked") -> dict[str, np.ndarray]:
    """Stacking input for models D/D0.

    Train rows get OUT-OF-FOLD probabilities (GroupKFold by class year), so the downstream model
    never sees a text score that was fit on the same player's outcome. Other splits get
    probabilities from a text model fit on all training rows.
    """
    y = train.tier_code.values
    oof = np.zeros((len(train), len(C.TIERS)))
    for tr, te in GroupKFold(n_splits=train.year.nunique()).split(train, y, groups=train.year):
        m = text_model(C_reg).fit(train[text_col].iloc[tr], y[tr])
        oof[te] = m.predict_proba(train[text_col].iloc[te])
    full = text_model(C_reg).fit(train[text_col], y)
    out = {"train": oof}
    out.update({k: full.predict_proba(v[text_col]) for k, v in others.items()})
    return out


def attach_text_probs(df: pd.DataFrame, probs: np.ndarray) -> pd.DataFrame:
    out = df.copy()
    for j, col in enumerate(TEXT_PROBS):   # UDFA column dropped (probabilities sum to 1)
        out[col] = probs[:, j]
    return out


# ---------------------------------------------------------------- scoring
def score_tiers(y, yhat) -> dict:
    y, yhat = np.asarray(y), np.asarray(yhat)
    return {"macro_f1": E.macro_f1(y, yhat), "qwk": E.qwk(y, yhat), "accuracy": float((y == yhat).mean()),
            "within_1_tier": float((np.abs(y - yhat) <= 1).mean())}


def per_class_recall(y, yhat) -> pd.Series:
    y, yhat = np.asarray(y), np.asarray(yhat)
    return pd.Series({t: float((yhat[y == k] == k).mean()) for k, t in enumerate(C.TIERS)})


# ---------------------------------------------------------------- secondary target
def pick_model(num_cols: list[str], alpha: float = 1.0) -> Pipeline:
    """Ridge on log(overall pick), drafted players only."""
    return Pipeline([("prep", tabular_preprocessor(num_cols)), ("reg", Ridge(alpha=alpha))])


def text_pick_model(alpha: float = 3.0) -> Pipeline:
    """TF-IDF -> Ridge on log(overall pick), drafted players only (text-only pick check)."""
    return Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=5, max_df=0.9, sublinear_tf=True,
                                  max_features=30_000)),
        ("reg", Ridge(alpha=alpha)),
    ])


def pick_mae(log_pick_true, log_pick_pred) -> float:
    return float(np.mean(np.abs(np.exp(log_pick_true) - np.exp(log_pick_pred))))

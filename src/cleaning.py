"""Cleaning and feature-prep for the NFL.com prospect data.

Every step is a small, pure function so the notebook can show (and assert on)
each one. `load_clean()` chains them all.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_CSV = ROOT / "nfl_prospects_2014_2026.csv"
CLEAN_PATH = ROOT / "data" / "prospects_clean.parquet"

KEY = ["displayName", "year", "position"]

# Ordered best -> worst. Mirrors the NFL's own three draft days + undrafted.
TIERS = ["R1", "R2-3", "R4-7", "UDFA"]
TIER_LABELS = {"R1": "Day 1 (Rd 1)", "R2-3": "Day 2 (Rd 2-3)", "R4-7": "Day 3 (Rd 4-7)", "UDFA": "Undrafted"}

SPLITS = {  # year -> split
    2014: "holdout_nawrocki",
    **{y: "train" for y in range(2015, 2023)},
    2023: "val",
    **{y: "test" for y in range(2024, 2027)},
}

# Power-5 membership as of 2014-2023 (pre-2024 realignment), plus Notre Dame.
POWER5 = {
    # ACC
    "Boston College", "Clemson", "Duke", "Florida State", "Georgia Tech", "Louisville", "Miami",
    "North Carolina", "N.C. State", "NC State", "North Carolina State", "Pittsburgh", "Syracuse",
    "Virginia", "Virginia Tech", "Wake Forest", "Notre Dame",
    # Big Ten
    "Illinois", "Indiana", "Iowa", "Maryland", "Michigan", "Michigan State", "Minnesota", "Nebraska",
    "Northwestern", "Ohio State", "Penn State", "Purdue", "Rutgers", "Wisconsin",
    # Big 12
    "Baylor", "Iowa State", "Kansas", "Kansas State", "Oklahoma", "Oklahoma State", "TCU", "Texas",
    "Texas Tech", "West Virginia",
    # Pac-12
    "Arizona", "Arizona State", "California", "Colorado", "Oregon", "Oregon State", "Stanford", "UCLA",
    "USC", "Southern California", "Utah", "Washington", "Washington State",
    # SEC
    "Alabama", "Arkansas", "Auburn", "Florida", "Georgia", "Kentucky", "LSU", "Mississippi", "Ole Miss",
    "Mississippi State", "Missouri", "South Carolina", "Tennessee", "Texas A&M", "Vanderbilt",
}

# Raw position -> group, used only to fill the few rows missing positionGroup.
POS_TO_GROUP = {
    "QB": "QB", "RB": "RB", "FB": "RB", "WR": "WR", "TE": "TE",
    "OT": "OL", "T": "OL", "G": "OL", "OG": "OL", "C": "OL", "OL": "OL",
    "DT": "DL", "NT": "DL", "DE": "DL", "DL": "DL", "EDGE": "DL",
    "LB": "LB", "ILB": "LB", "OLB": "LB",
    "CB": "DB", "S": "DB", "SAF": "DB", "FS": "DB", "SS": "DB", "DB": "DB", "WR/CB": "DB",
    "K": "SPEC", "P": "SPEC", "LS": "SPEC",
}

# NFL franchises, incl. names used during 2014-2026 (for masking + leakage check).
NFL_TEAMS = {
    "Arizona Cardinals": ["Cardinals"], "Atlanta Falcons": ["Falcons"], "Baltimore Ravens": ["Ravens"],
    "Buffalo Bills": ["Bills"], "Carolina Panthers": ["Panthers"], "Chicago Bears": ["Bears"],
    "Cincinnati Bengals": ["Bengals"], "Cleveland Browns": ["Browns"], "Dallas Cowboys": ["Cowboys"],
    "Denver Broncos": ["Broncos"], "Detroit Lions": ["Lions"], "Green Bay Packers": ["Packers"],
    "Houston Texans": ["Texans"], "Indianapolis Colts": ["Colts"], "Jacksonville Jaguars": ["Jaguars"],
    "Kansas City Chiefs": ["Chiefs"], "Las Vegas Raiders": ["Raiders", "Oakland Raiders"],
    "Los Angeles Chargers": ["Chargers", "San Diego Chargers"], "Los Angeles Rams": ["Rams", "St. Louis Rams"],
    "Miami Dolphins": ["Dolphins"], "Minnesota Vikings": ["Vikings"], "New England Patriots": ["Patriots"],
    "New Orleans Saints": ["Saints"], "New York Giants": ["Giants"], "New York Jets": ["Jets"],
    "Philadelphia Eagles": ["Eagles"], "Pittsburgh Steelers": ["Steelers"], "San Francisco 49ers": ["49ers"],
    "Seattle Seahawks": ["Seahawks"], "Tampa Bay Buccaneers": ["Buccaneers", "Bucs"],
    "Tennessee Titans": ["Titans"],
    "Washington Commanders": ["Commanders", "Redskins", "Washington Football Team"],
}
TEXT_FIELDS = ["overview", "strengths", "weaknesses"]


# ---------------------------------------------------------------- steps
def load_raw(path: Path | str = RAW_CSV) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig")


def clean_authors(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize detailAuthor typos/whitespace; non-names ('Round 7/PFA') -> NA."""
    a = df["detailAuthor"].str.strip().str.rstrip(".")
    a = a.replace({"Lance Zierlien": "Lance Zierlein", "Round 7/PFA": pd.NA})
    return df.assign(detailAuthor=a)


def normalize_positions(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing positionGroup from the raw position (UNK if no position at all); flag specialists."""
    grp = df["positionGroup"].fillna(df["position"].map(POS_TO_GROUP)).fillna("UNK")
    return df.assign(positionGroup=grp, is_spec=grp.eq("SPEC"))


def clean_grade(df: pd.DataFrame) -> pd.DataFrame:
    """grade <= 0 is a placeholder -> NA. grade_pct = within-class percentile.

    The grade scale tightens over time (the IQR narrows after ~2021), so a raw 6.0 doesn't
    mean the same thing in every class. Models use grade_pct; raw grade is kept for display.
    """
    g = df["grade"].where(df["grade"] > 0)
    return df.assign(grade=g, grade_pct=g.groupby(df["year"]).rank(pct=True))


def add_targets(df: pd.DataFrame) -> pd.DataFrame:
    """drafted (bool), tier (ordered categorical, best->worst), tier_code (0=R1..3=UDFA), log_pick."""
    rnd = df["draftRound"]
    tier = np.select([rnd.eq(1), rnd.between(2, 3), rnd.between(4, 7)], TIERS[:3], default="UDFA")
    tier = pd.Categorical(tier, categories=TIERS, ordered=True)
    return df.assign(
        drafted=rnd.notna(),
        tier=tier,
        tier_code=tier.codes,
        log_pick=np.log(df["draftOverallPick"]),
    )


def add_school_tier(df: pd.DataFrame) -> pd.DataFrame:
    """power5 = any listed school (transfers are '/'-joined) is a 2014-23 Power-5 program."""
    schools = df["collegeNames"].fillna("").str.split("/")
    return df.assign(power5=schools.apply(lambda s: any(x.strip() in POWER5 for x in s)))


def build_text(df: pd.DataFrame) -> pd.DataFrame:
    """scout_text = labeled overview + strengths + weaknesses (the analyst's written evaluation)."""
    parts = [df[f].fillna("").str.strip() for f in TEXT_FIELDS]
    text = ("Overview: " + parts[0] + "\nStrengths: " + parts[1] + "\nWeaknesses: " + parts[2])
    has_text = df[TEXT_FIELDS].notna().any(axis=1)
    return df.assign(
        scout_text=text.where(has_text),
        has_text=has_text,
        text_len=text.where(has_text).str.len(),
    )


def assign_split(df: pd.DataFrame) -> pd.DataFrame:
    return df.assign(split=df["year"].map(SPLITS))


def analysis_flags(df: pd.DataFrame) -> pd.DataFrame:
    """in_analysis = has text AND has grade AND not a specialist (K/P/LS) AND position known."""
    has_grade = df["grade"].notna()
    known = df["positionGroup"].ne("UNK")
    return df.assign(has_grade=has_grade, in_analysis=df["has_text"] & has_grade & ~df["is_spec"] & known)


def load_clean(path: Path | str = RAW_CSV) -> pd.DataFrame:
    return (load_raw(path)
            .pipe(clean_authors)
            .pipe(normalize_positions)
            .pipe(clean_grade)
            .pipe(add_targets)
            .pipe(add_school_tier)
            .pipe(build_text)
            .pipe(assign_split)
            .pipe(analysis_flags))


# ---------------------------------------------------------------- leakage + masking
def _word_re(terms) -> re.Pattern:
    terms = sorted({t for t in terms if t}, key=len, reverse=True)
    return re.compile(r"\b(?:" + "|".join(re.escape(t) for t in terms) + r")\b")


def team_aliases(full_name: str) -> list[str]:
    return [full_name, *NFL_TEAMS.get(full_name, [])]


def mentions_team(text: str, full_name: str) -> bool:
    if not isinstance(text, str) or not isinstance(full_name, str):
        return False
    return bool(_word_re(team_aliases(full_name)).search(text))


_ALL_TEAMS_RE = _word_re([a for t in NFL_TEAMS for a in team_aliases(t)])


def mask_text(text: str, name: str | None, colleges: str | None) -> str:
    """Hide identity cues before sending text to the LLM (contamination control).

    Replaces the player's full/first/last name with [PLAYER], their school(s) with
    [SCHOOL], and any NFL team name/nickname with [TEAM].
    """
    if not isinstance(text, str):
        return text
    if isinstance(colleges, str):
        schools = [c.strip() for c in colleges.split("/") if c.strip()]
        text = _word_re(schools).sub("[SCHOOL]", text)
    if isinstance(name, str):
        parts = [p for p in re.split(r"\s+", name.strip()) if len(p) > 1]
        text = _word_re([name, *parts]).sub("[PLAYER]", text)
    return _ALL_TEAMS_RE.sub("[TEAM]", text)


if __name__ == "__main__":
    d = load_clean()
    print(d.shape)
    print(d.tier.value_counts().reindex(TIERS))
    print(d.groupby("split").in_analysis.sum())

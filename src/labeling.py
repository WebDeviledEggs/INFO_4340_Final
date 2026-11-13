"""Human labeling sample and workbooks (codebook: docs/codebook.md).

Practice items come from the validation class (2023), and evaluation items come from the test
classes (2024-26). Both are stratified by actual tier so every class has enough items for kappa
and per-class agreement. The workbooks show only the position and the masked text, never names,
schools, grades or outcomes.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl import Workbook
from openpyxl.utils import get_column_letter
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.worksheet.datavalidation import DataValidation

from . import cleaning as C

LABEL_DIR = C.ROOT / "data" / "labels"
KEY_PATH = LABEL_DIR / "label_items_key.csv"
LABELERS = ["partner1", "partner2"]

FIELDS = {  # column -> allowed values (dropdowns)
    "projected_tier": C.TIERS,
    "role": ["starter", "eventual_starter", "backup", "roster_bubble"],
    "injury_flag": ["0", "1"],
    "character_flag": ["0", "1"],
    "tone": ["1", "2", "3", "4", "5"],
}
ORDINAL = {"projected_tier": {t: i for i, t in enumerate(C.TIERS)},
           "role": {r: i for i, r in enumerate(FIELDS["role"])}}


def draw_label_sample(A: pd.DataFrame, n_practice_per_tier: int = 5, n_eval_per_tier: int = 25,
                      seed: int = 4340) -> pd.DataFrame:
    """Stratified sample: practice from val, evaluation from test; shuffled within each set."""
    rng = np.random.default_rng(seed)
    parts = []
    for set_name, split, k, prefix in [("practice", "val", n_practice_per_tier, "P"),
                                       ("eval", "test", n_eval_per_tier, "E")]:
        pool = A[A.split == split]
        s = pd.concat([pool[pool.tier == t].sample(k, random_state=int(rng.integers(1e9)))
                       for t in C.TIERS])
        s = s.sample(frac=1, random_state=int(rng.integers(1e9)))
        width = 2 if set_name == "practice" else 3
        s = s.assign(item_id=[f"{prefix}{i:0{width}d}" for i in range(1, len(s) + 1)], set=set_name)
        parts.append(s)
    out = pd.concat(parts)
    return out[["item_id", "set", *C.KEY, "positionGroup", "split", "text_masked"]].reset_index(drop=True)


def write_workbook(items: pd.DataFrame, path: Path, labeler: str) -> None:
    """One sheet per labeler with dropdown validation. Never overwrites an existing file."""
    if path.exists():
        raise FileExistsError(f"{path} exists; refusing to overwrite labels")
    wb = Workbook()
    ws = wb.active
    ws.title = "labels"
    cols = ["item_id", "set", "position", "report", *FIELDS, "notes"]
    ws.append(cols)
    for _, r in items.iterrows():
        ws.append([r.item_id, r.set, r.position, r.text_masked, *[None] * len(FIELDS), None])

    head = PatternFill("solid", fgColor="DDEBF7")
    for c in ws[1]:
        c.font, c.fill = Font(bold=True), head
    field_cols = [get_column_letter(5 + j) for j in range(len(FIELDS))]   # fields start in column E
    notes_col = get_column_letter(5 + len(FIELDS))
    for col, w in {"A": 8, "B": 9, "C": 9, "D": 100, notes_col: 30}.items():
        ws.column_dimensions[col].width = w
    for col in field_cols:
        ws.column_dimensions[col].width = 15
    for row in ws.iter_rows(min_row=2):
        row[3].alignment = Alignment(wrap_text=True, vertical="top")
        for c in row:
            if c.column != 4:
                c.alignment = Alignment(vertical="top")
    ws.freeze_panes = "E2"

    n = len(items) + 1
    for j, (field, allowed) in enumerate(FIELDS.items()):
        col = field_cols[j]
        dv = DataValidation(type="list", formula1='"' + ",".join(allowed) + '"', allow_blank=True,
                            showErrorMessage=True, errorTitle="Not in codebook",
                            error=f"{field} must be one of: {', '.join(allowed)}")
        ws.add_data_validation(dv)
        dv.add(f"{col}2:{col}{n}")

    info = wb.create_sheet("read me")
    for line in [f"Labeler: {labeler}",
                 "Codebook: docs/codebook.md (read it first).",
                 "Label from the text only. Do not look players up.",
                 "Do the practice items (P01-P20) first, then compare with your partner and update the codebook.",
                 "Then label E001-E100 independently. Do not change the codebook during the evaluation round.",
                 "If you recognize the player anyway, write 'recognized' in notes.",
                 "Save this file in place (same name, .xlsx)."]:
        info.append([line])
    info.column_dimensions["A"].width = 110
    wb.save(path)


def workbook_path(labeler: str) -> Path:
    return LABEL_DIR / f"labels_{labeler}.xlsx"


def read_labels(labeler: str) -> pd.DataFrame:
    """Completed labels for one labeler (blank rows dropped), with numeric codes for kappa."""
    path = workbook_path(labeler)
    df = pd.read_excel(path, sheet_name="labels", dtype=str)
    df = df.drop(columns=["confidence"], errors="ignore")   # removed from the codebook in v1.1 (older workbooks still have the column)
    df = df.dropna(subset=["projected_tier"])
    for f in ["injury_flag", "character_flag", "tone"]:
        df[f] = pd.to_numeric(df[f], errors="coerce")
    for f, codes in ORDINAL.items():
        df[f"{f}_code"] = df[f].map(codes)
    return df.assign(labeler=labeler)

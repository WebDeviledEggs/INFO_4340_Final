"""Claude API access for the LLM step (notebook section 7).

The key is read from `.env` at the repo root (gitignored). python-dotenv isn't needed;
the tiny loader below accepts KEY=value, with or without quotes.

The prompt mirrors docs/codebook.md so the LLM and the two human labelers answer the same
questions under the same decision rules.
"""
from __future__ import annotations

import json
import os

import anthropic
import numpy as np
import pandas as pd

from . import cleaning as C

MODEL = "claude-opus-5-5"
EFFORT = "low"
MAX_TOKENS = 2000            # thinking + JSON; JSON alone is ~120 tokens
PROMPT_VERSION = "v2"   # v2 (2026-11-12): confidence field removed to match codebook v1.1
ENV_PATH = C.ROOT / ".env"

# $ per million tokens (Claude Opus 5.5). Batch API is 50% off; cache reads are 0.05x input.
PRICE = {"standard": {"in": 4.0, "out": 20.0, "cache_read": 0.20, "cache_write": 5.0},
         "batch":    {"in": 2.0, "out": 10.0, "cache_read": 0.10, "cache_write": 2.5}}


# ---------------------------------------------------------------- client
def load_env(path=ENV_PATH) -> None:
    """Put KEY=value lines from .env into os.environ (existing variables win)."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip("'\""))


def get_client() -> anthropic.Anthropic:
    load_env()
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError("ANTHROPIC_API_KEY not set. Copy .env.example to .env and add your key.")
    return anthropic.Anthropic()


# ---------------------------------------------------------------- prompt (v1)
SYSTEM_PROMPT = """You are labeling NFL draft scouting reports for a research study. Each report was written by a professional analyst before the draft. Your labels are compared with labels from two human readers who follow the same codebook, so apply the rules below exactly as written.

The player's name, school, and any NFL team names have been replaced with [PLAYER], [SCHOOL], and [TEAM]. Label from the text alone. Do not try to work out who the player is, and do not use anything you may know about real draft results. If you believe you recognize the player anyway, set "recognized" to true and still label from the text only.

Fields:

1. projected_tier: where the TEXT expects the player to be drafted.
   R1 = Round 1; R2-3 = Rounds 2-3; R4-7 = Rounds 4-7; UDFA = not drafted.
   Rules, in order:
   a. Explicit draft language wins ("Day 2 pick", "first-round talent", "late-round flier", "priority free agent", "camp body").
   b. Otherwise use the projected NFL role:
      - immediate / Day 1 starter, high-end starter, Pro Bowl potential -> R1
      - should become a starter, starter by year 2, early contributor -> R2-3
      - backup with upside, rotational, special teams value, developmental, could compete for a roster spot -> R4-7
      - will need to fight for a roster spot, camp competition, practice squad, hard to project as more than depth -> UDFA
   c. If still tied, a physical-tools concern for the position in the weaknesses (for example "below-average speed for a corner") moves the projection one tier lower.
   d. Do not adjust for position value (for example "quarterbacks go early"). Label what the analyst says about this player.
   e. If the report is too short to judge, give your best guess.

2. role: the NFL role the text projects. If the text gives a range, take the LOWER end.
   starter = will start early, plug-and-play, immediate impact
   eventual_starter = can become a starter, starter with development, starter in the right scheme
   backup = backup, rotational, sub-package or situational player, depth
   roster_bubble = special teams only, practice squad, camp competition, fighting for a spot

3. injury_flag: 1 if the text mentions this player's injury history, surgery, games missed because of injury, or a durability concern. 0 otherwise ("plays through contact" or "tough" alone is 0).

4. character_flag: 1 if the text raises an off-field or work-habit concern (arrest, suspension, dismissal, failed test, maturity, coachability, "teams will need to vet him"). 0 for praise of character and for on-field effort or motor comments.

5. tone: how positive the evaluation is overall, judged mainly from the overview (the strengths and weaknesses lists always contain both sides; use them only to break a tie).
   1 = mostly negative, doubts he sticks; 2 = more negative than positive; 3 = balanced; 4 = more positive than negative; 5 = glowing, weaknesses minor.

6. evidence: the short phrase from the report (at most 25 words, quoted exactly) that most drove projected_tier.

7. recognized: true only if you believe you know who this player is despite the masking."""

OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "projected_tier": {"type": "string", "enum": C.TIERS},
        "role": {"type": "string", "enum": ["starter", "eventual_starter", "backup", "roster_bubble"]},
        "injury_flag": {"type": "integer", "enum": [0, 1]},
        "character_flag": {"type": "integer", "enum": [0, 1]},
        "tone": {"type": "integer", "enum": [1, 2, 3, 4, 5]},
        "evidence": {"type": "string"},
        "recognized": {"type": "boolean"},
    },
    "required": ["projected_tier", "role", "injury_flag", "character_flag", "tone", "evidence",
                 "recognized"],
    "additionalProperties": False,
}


def user_message(position: str, text_masked: str) -> str:
    return f"Position: {position}\n\nReport:\n{text_masked}"


def request_params(position: str, text_masked: str) -> dict:
    """Keyword arguments for messages.create (also the `params` of a batch request)."""
    return {
        "model": MODEL,
        "max_tokens": MAX_TOKENS,
        # The system prompt is identical for every report, so cache it (cuts its input cost ~95%).
        "system": [{"type": "text", "text": SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}],
        "messages": [{"role": "user", "content": user_message(position, text_masked)}],
        "output_config": {"effort": EFFORT, "format": {"type": "json_schema", "schema": OUTPUT_SCHEMA}},
    }


def parse_response(msg) -> dict:
    """JSON fields + token usage from one Message. Refusals and truncations are recorded, not raised."""
    u = msg.usage
    row = {"stop_reason": msg.stop_reason, "input_tokens": u.input_tokens, "output_tokens": u.output_tokens,
           "cache_read_tokens": getattr(u, "cache_read_input_tokens", 0) or 0,
           "cache_write_tokens": getattr(u, "cache_creation_input_tokens", 0) or 0}
    if msg.stop_reason == "end_turn":
        text = next((b.text for b in msg.content if b.type == "text"), None)
        if text:
            row.update(json.loads(text))
    return row


def cost(df: pd.DataFrame, mode: str = "standard") -> float:
    """Dollar cost from logged token counts."""
    p = PRICE[mode]
    return float((df.input_tokens * p["in"] + df.output_tokens * p["out"]
                  + df.cache_read_tokens * p["cache_read"] + df.cache_write_tokens * p["cache_write"]).sum() / 1e6)


# ---------------------------------------------------------------- pilot sample
def pilot_sample(A: pd.DataFrame, label_items: pd.DataFrame, n: int = 50, seed: int = 4340) -> pd.DataFrame:
    """Pilot reports from the 2023 validation class: the 20 human practice items plus a
    tier-stratified fill to n. Never touches the 2024-26 test classes."""
    val = A[A.split == "val"]
    practice = val.merge(label_items.loc[label_items.set == "practice", C.KEY], on=C.KEY)
    rest = val.merge(practice[C.KEY], on=C.KEY, how="left", indicator=True)
    rest = rest[rest._merge == "left_only"].drop(columns="_merge")
    per_tier = int(np.ceil((n - len(practice)) / len(C.TIERS)))
    fill = pd.concat([rest[rest.tier == t].sample(per_tier, random_state=seed) for t in C.TIERS])
    out = pd.concat([practice.assign(in_practice=True), fill.assign(in_practice=False)])
    return out.reset_index(drop=True)


# ---------------------------------------------------------------- standard-API runs (pilot)
LLM_DIR = C.ROOT / "data" / "llm"


def run_standard(items: pd.DataFrame, out_csv, budget_left: float, workers: int = 4) -> pd.DataFrame:
    """Label `items` (needs KEY cols, position, text_masked) one request each; append to `out_csv`.

    Resumable: rows already in out_csv are skipped. The first call runs alone so it writes the
    prompt cache; the rest run in parallel and read it. Stops before exceeding `budget_left` ($),
    using the most expensive call so far as the estimate for the next one.
    """
    import time
    from concurrent.futures import ThreadPoolExecutor

    client = get_client()
    done = pd.read_csv(out_csv) if out_csv.exists() else pd.DataFrame(columns=C.KEY)
    todo = items.merge(done[C.KEY], on=C.KEY, how="left", indicator=True)
    todo = todo[todo._merge == "left_only"].drop(columns="_merge")

    def one(r):
        t0 = time.time()
        msg = client.messages.create(**request_params(r.position, r.text_masked))
        return {**{k: getattr(r, k) for k in C.KEY}, **parse_response(msg),
                "latency_s": round(time.time() - t0, 2), "prompt_version": PROMPT_VERSION, "model": MODEL}

    rows, spent, worst = [], 0.0, 0.0
    def record(row):
        nonlocal spent, worst
        c = cost(pd.DataFrame([row]))
        spent += c; worst = max(worst, c); rows.append(row)

    recs = list(todo.itertuples(index=False))
    if recs:
        record(one(recs[0]))
        with ThreadPoolExecutor(workers) as ex:
            for i in range(1, len(recs), workers):
                if spent + worst * workers * 1.5 > budget_left:
                    print(f"budget stop: spent ${spent:.3f}, {len(recs) - i} items left")
                    break
                for row in ex.map(one, recs[i:i + workers]):
                    record(row)
    new = pd.DataFrame(rows)
    out = pd.concat([done, new], ignore_index=True) if len(done) else new
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)
    print(f"{len(new)} new calls, ${spent:.3f} this run; file total ${cost(out):.3f}")
    return out

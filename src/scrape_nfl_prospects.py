"""Scrape NFL.com Draft Tracker prospect profiles (2014-2026) into a CSV.

Each https://www.nfl.com/draft/tracker/{year}/prospects page embeds the full
prospect dataset in its Next.js (React Server Components) payload, so one
request per year is enough.

Usage:
    python scrape_nfl_prospects.py [--start 2014] [--end 2026] [--out file.csv]
"""
import argparse
import json
import random
import re
import time
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

URL = "https://www.nfl.com/draft/tracker/{year}/prospects"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/130.0 Safari/537.36"
}
COLUMNS = [
    "displayName", "position", "positionGroup", "collegeNames", "year",
    "draftRound", "draftPick", "draftOverallPick", "draftTeam",
    "height", "weight", "armLength", "wingspan", "handSize",
    "grade", "productionScore", "athleticismScore", "draftGrade",
    "bio", "overview", "strengths", "weaknesses", "nflComparison", "detailAuthor",
]
HTML_FIELDS = ["bio", "overview", "strengths", "weaknesses"]


def fetch(year):
    resp = requests.get(URL.format(year=year), headers=HEADERS, timeout=60)
    resp.raise_for_status()
    return resp.text


def decode_flight(html):
    """Join the RSC payload chunks pushed via self.__next_f.push([1, "..."])."""
    chunks = re.findall(r'self\.__next_f\.push\(\[1,(".*?")\]\)</script>', html, re.S)
    return "".join(json.loads(c) for c in chunks)


def text_refs(flight):
    """Map RSC text-row ids ("5f:T<hexlen>,<text>") to their text."""
    refs = {}
    for m in re.finditer(r"([0-9a-f]+):T([0-9a-f]+),", flight):
        n_bytes = int(m.group(2), 16)
        # Length is in UTF-8 bytes; n chars always covers at least n bytes.
        raw = flight[m.end():m.end() + n_bytes].encode("utf-8")[:n_bytes]
        refs[m.group(1)] = raw.decode("utf-8", errors="ignore")
    return refs


def extract_array(flight, key):
    start = flight.index(f'"{key}":[') + len(f'"{key}":')
    array, _ = json.JSONDecoder().raw_decode(flight, start)
    return array


def pick_teams(flight):
    """Map prospect personId -> drafting teamId from the draft-order picks list.

    Used as a fallback: a few profiles have a draft slot but no draftTeamId.
    """
    try:
        picks = extract_array(flight, "picks")
    except ValueError:
        return {}
    return {(pk.get("prospect") or {}).get("personId"): pk.get("teamId") for pk in picks}


def extract_teams(flight):
    pairs = re.findall(r'\{"id":"([0-9a-f-]{36})"[^{}]*?"fullName":"([^"]+)"', flight)
    return dict(pairs)


def html_to_text(value, refs):
    if not value:
        return None
    if re.fullmatch(r"\$[0-9a-f]+", value):
        value = refs.get(value[1:], "")
    soup = BeautifulSoup(value, "html.parser")
    items = soup.find_all("li")
    if items:
        return " | ".join(li.get_text(" ", strip=True) for li in items)
    return soup.get_text(" ", strip=True) or None


def parse_profile(p, teams, refs, picked_by):
    person = p.get("person") or {}
    row = {k: p.get(k) for k in COLUMNS}
    row["displayName"] = person.get("displayName")
    row["collegeNames"] = "/".join(person.get("collegeNames") or []) or None
    row["draftTeam"] = teams.get(p.get("draftTeamId") or picked_by.get(person.get("id")))
    for k in HTML_FIELDS:
        row[k] = html_to_text(p.get(k), refs)
    return row


def scrape_year(year):
    flight = decode_flight(fetch(year))
    teams, refs, picked_by = extract_teams(flight), text_refs(flight), pick_teams(flight)
    return [parse_profile(p, teams, refs, picked_by) for p in extract_array(flight, "profiles")]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--start", type=int, default=2014)
    parser.add_argument("--end", type=int, default=2026)
    parser.add_argument("--out", default=str(Path(__file__).parent / "nfl_prospects_2014_2026.csv"))
    args = parser.parse_args()

    rows = []
    for year in range(args.start, args.end + 1):
        try:
            year_rows = scrape_year(year)
            rows.extend(year_rows)
            print(f"{year}: {len(year_rows)} prospects")
        except Exception as e:
            print(f"{year}: FAILED ({e})")
        if year < args.end:
            time.sleep(random.uniform(2, 3))

    df = pd.DataFrame(rows, columns=COLUMNS)
    df.to_csv(args.out, index=False, encoding="utf-8-sig")
    print(f"Wrote {len(df)} rows to {args.out}")


if __name__ == "__main__":
    main()

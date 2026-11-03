"""
Summarize Project Implicit Race IAT data (OSF) by state and respondent race.

Reads the big files in chunks, keeps only a few columns, and writes a small
JSON file (a few KB) you can upload instead of the full dataset.

Setup (once):
    pip install pandas pyreadstat

Usage (one or more files, .sav or .csv):
    python summarize_race_iat_by_state.py Race.IAT.2023.sav Race.IAT.2024.sav

Output: state_cohort_iat.json in the current folder.
"""
import argparse
import json
import math
import os
import sys
from collections import defaultdict

import pandas as pd

STATES = set("""AL AK AZ AR CA CO CT DE DC FL GA HI ID IL IN IA KS KY LA ME MD MA MI MN MS
MO MT NE NV NH NJ NM NY NC ND OH OK OR PA RI SC SD TN TX UT VT VA WA WV WI WY""".split())

D_CANDIDATES = ["D_biep.White_Good_all", "D_biep.White_Good_36", "D_biep.White_Good_47"]
STATE_CANDIDATES = ["STATE", "state"]
RACE_CANDIDATES = ["raceomb_002", "raceomb", "raceombmulti", "ethnicityomb"]
CHUNK = 200_000


def pick(cols, candidates, override, what):
    if override:
        if override not in cols:
            sys.exit(f"Column '{override}' not found. Available columns include: {cols[:40]}")
        return override
    for c in candidates:
        if c in cols:
            return c
    sys.exit(f"Couldn't find a {what} column. Pass it with --{what}-col. Columns: {cols[:60]}")


def find_code(labels, word, override):
    if override is not None:
        return override
    hits = [k for k, v in labels.items()
            if word in str(v).lower() and "non" not in str(v).lower() and "multi" not in str(v).lower()]
    return hits[0] if len(hits) == 1 else None


def iter_chunks(path, usecols):
    if path.lower().endswith(".sav"):
        import pyreadstat
        for df, _ in pyreadstat.read_file_in_chunks(
                pyreadstat.read_sav, path, chunksize=CHUNK, usecols=usecols):
            yield df
    else:
        yield from pd.read_csv(path, usecols=usecols, chunksize=CHUNK, low_memory=False)


def columns_and_labels(path, race_override):
    if path.lower().endswith(".sav"):
        import pyreadstat
        _, meta = pyreadstat.read_sav(path, metadataonly=True)
        return list(meta.column_names), meta.variable_value_labels
    return list(pd.read_csv(path, nrows=0).columns), {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--d-col"); ap.add_argument("--state-col"); ap.add_argument("--race-col")
    ap.add_argument("--black-code", type=float); ap.add_argument("--white-code", type=float)
    ap.add_argument("--min-n", type=int, default=100,
                    help="Flag state/cohort cells with fewer respondents than this")
    ap.add_argument("--out", default="state_cohort_iat.json")
    a = ap.parse_args()

    # running sums per (state, cohort): n, sum, sum of squares
    acc = defaultdict(lambda: [0, 0.0, 0.0])
    report = []

    for path in a.files:
        cols, all_labels = columns_and_labels(path, a.race_col)
        d_col = pick(cols, D_CANDIDATES, a.d_col, "d")
        st_col = pick(cols, STATE_CANDIDATES, a.state_col, "state")
        race_col = pick(cols, RACE_CANDIDATES, a.race_col, "race")
        labels = all_labels.get(race_col, {})
        black = find_code(labels, "black", a.black_code)
        white = find_code(labels, "white", a.white_code)
        print(f"\n{os.path.basename(path)}: D={d_col}  state={st_col}  race={race_col}")
        if labels:
            print("  Race codes in this file:", labels)
        if black is None or white is None:
            sys.exit("  Couldn't tell which race codes mean Black and White. "
                     "Re-run with --black-code and --white-code using the codes listed above.")
        print(f"  Using Black = {black}, White = {white}")

        usecols = [d_col, st_col, race_col] + (["session_status"] if "session_status" in cols else [])
        kept = 0
        for df in iter_chunks(path, usecols):
            if "session_status" in df:
                df = df[df["session_status"].astype(str).str.upper().str.strip() == "C"]
            df = df[pd.to_numeric(df[d_col], errors="coerce").notna()]
            d = pd.to_numeric(df[d_col])
            st = df[st_col].astype(str).str.strip().str.upper()
            race = pd.to_numeric(df[race_col], errors="coerce")
            ok = st.isin(STATES) & d.between(-2, 2)
            d, st, race = d[ok], st[ok], race[ok]
            kept += len(d)
            for cohort, mask in (("all", slice(None)), ("black", race == black), ("white", race == white)):
                g = pd.DataFrame({"s": st[mask], "d": d[mask]}).groupby("s")["d"]
                for s, n, sm, sq in zip(g.size().index, g.size(), g.sum(), (g.apply(lambda x: (x ** 2).sum()))):
                    cell = acc[(s, cohort)]
                    cell[0] += int(n); cell[1] += float(sm); cell[2] += float(sq)
        report.append((os.path.basename(path), kept))
        print(f"  Kept {kept:,} completed sessions with a valid state and D-score")

    out = {"source_files": [r[0] for r in report], "min_n": a.min_n, "states": {}}
    for (s, cohort), (n, sm, sq) in sorted(acc.items()):
        mean = sm / n
        sd = math.sqrt(max(sq / n - mean ** 2, 0))
        out["states"].setdefault(s, {})[cohort] = {
            "n": n, "mean_d": round(mean, 4), "sd": round(sd, 4), "low_n": n < a.min_n}
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)
    print(f"\nDone. Wrote {a.out} ({os.path.getsize(a.out):,} bytes). Upload that file.")


if __name__ == "__main__":
    main()

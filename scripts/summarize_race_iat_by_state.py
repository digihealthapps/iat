"""
Summarize Project Implicit Race IAT data (OSF) by state and respondent race.

Reads the big files in chunks, keeps only a few columns, and writes a small
JSON file (a few KB) you can upload instead of the full dataset.

Setup (once):
    pip install pandas pyreadstat

Usage (one or more files, .sav or .csv):
    python summarize_race_iat_by_state.py Race.IAT.2023.sav Race.IAT.2024.sav

Output: state_cohort_iat.json in the current folder. For each state (plus a
"US" total) and each cohort (all, black, white) it records the respondent
count, mean and SD of the D-score, a low-sample flag, how many respondents
fall in each D-score band (strong/moderate/slight pro-White, neutral,
slight/moderate/strong pro-Black) and, where the file has block latencies,
the average response time in the White+good and Black+good blocks.

Cohorts: all, white, black, asian, hispanic.
  - white / black / asian: that race only, and not Hispanic.
  - hispanic: Hispanic or Latino, of any race.
Race is recorded three ways across the years:
  - 2002-2006 "ethnic": one combined answer ("White-Not of Hispanic Origin",
    "Asian or Pacific Islander", "Hispanic", ...). Asian includes Pacific
    Islander in these years.
  - 2007-2022 "raceomb" / "raceomb_002" / "raceomb002": one race answer
    (East Asian and South Asian both count as Asian), plus a separate
    Hispanic question "ethnicityomb".
  - 2023-2025 tick boxes, one per race including Hispanic (raceomb_003_white, ...).

Response times: the test randomises whether White+good or Black+good comes
first (column "Order"). Rather than assume what each Order code means, the
script checks it against the D-score (positive D = faster at White+good) and
prints how well each code agrees.
"""
import argparse
import json
import math
import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

STATES = set("""AL AK AZ AR CA CO CT DE DC FL GA HI ID IL IN IA KS KY LA ME MD MA MI MN MS
MO MT NE NV NH NJ NM NY NC ND OH OK OR PA RI SC SD TN TX UT VT VA WA WV WI WY""".split())

D_CANDIDATES = ["D_biep.White_Good_all", "D_biep.White_Good_36", "D_biep.White_Good_47"]
STATE_CANDIDATES = ["STATE", "state"]
RACE_CANDIDATES = ["raceomb_002", "raceomb002", "raceomb", "ethnic"]  # "ethnic": 2002-2006 combined race/ethnicity
RACE_TICK_PREFIX = "raceomb_003_"   # one tick-box column per race, e.g. raceomb_003_white
RT_COLS = ["Mn_RT_all_3", "Mn_RT_all_4", "Mn_RT_all_6", "Mn_RT_all_7"]
ORDER_COL = "Order"
RT_RANGE = (200, 10_000)  # ms; block means outside this are treated as invalid
CHUNK = 200_000

# D-score bands, using the same cut-offs as the dashboard (0.15 / 0.35 / 0.65).
# |D| <= 0.15 is neutral; beyond that, slight up to 0.35, moderate up to 0.65, strong above.
BANDS = ["strong_white", "moderate_white", "slight_white", "neutral",
         "slight_black", "moderate_black", "strong_black"]


def band_of(d):
    mag = d.abs()
    white = d > 0
    return pd.Series(np.select(
        [mag <= 0.15,
         white & (mag > 0.65), white & (mag > 0.35), white,
         mag > 0.65, mag > 0.35],
        ["neutral", "strong_white", "moderate_white", "slight_white",
         "strong_black", "moderate_black"],
        default="slight_black"), index=d.index)


def pick(cols, candidates, override, what):
    if override:
        if override not in cols:
            sys.exit(f"Column '{override}' not found. Available columns include: {cols[:40]}")
        return override
    for c in candidates:
        if c in cols:
            return c
    sys.exit(f"Couldn't find a {what} column. Pass it with --{what}-col. Columns: {cols[:60]}")


def find_codes(labels, word):
    """All codes whose label names `word`, skipping multi-race answers and negations.

    "Not Hispanic or Latino" is skipped; "White-Not of Hispanic Origin" (2002-2006)
    still counts as White but not as Hispanic.
    """
    skip = ("multi", "more than one", "mixed", "biracial")
    def wanted(v):
        v = str(v).lower().strip()
        if word not in v or any(x in v for x in skip) or v.startswith(("not ", "non")):
            return False
        return not (word == "hispanic" and "not" in v)
    return [k for k, v in labels.items() if wanted(v)]


def find_code(labels, word, override):
    if override is not None:
        return override
    hits = find_codes(labels, word)
    return hits[0] if len(hits) == 1 else None


def acc_total(acc, cohort):
    return sum(v[0] for (s, c), v in acc.items() if c == cohort and s != "US")


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
    # respondent counts per (state, cohort) and D-score band
    band_acc = defaultdict(lambda: dict.fromkeys(BANDS, 0))
    # response times per (state, cohort): n, sum White+good ms, sum Black+good ms
    rt_acc = defaultdict(lambda: [0, 0.0, 0.0])
    report = []

    for path in a.files:
        name = os.path.basename(path)
        cols, all_labels = columns_and_labels(path, a.race_col)
        d_col = pick(cols, D_CANDIDATES, a.d_col, "d")
        st_col = pick(cols, STATE_CANDIDATES, a.state_col, "state")

        tick_cols = [c for c in cols if c.startswith(RACE_TICK_PREFIX)]
        tick_mode = (not a.race_col and f"{RACE_TICK_PREFIX}white" in cols
                     and f"{RACE_TICK_PREFIX}black" in cols)
        print(f"\n{name}: D={d_col}  state={st_col}")
        if tick_mode:
            race_cols = tick_cols
            print(f"  Race: tick boxes ({', '.join(c[len(RACE_TICK_PREFIX):] for c in tick_cols)}); "
                  "a box counts as ticked when its value is 1")
            for c in (f"{RACE_TICK_PREFIX}white", f"{RACE_TICK_PREFIX}black",
                      f"{RACE_TICK_PREFIX}asian", f"{RACE_TICK_PREFIX}hispanic"):
                if all_labels.get(c):
                    print(f"  Codes for {c}: {all_labels[c]}")
        else:
            race_col = pick(cols, RACE_CANDIDATES, a.race_col, "race")
            race_cols = [race_col]
            # Hispanic is a separate question in these files; leave Hispanic respondents out of
            # the White/Black groups so they match the tick-box definition ("only that race").
            eth_col = "ethnicityomb" if "ethnicityomb" in cols else None
            hisp = find_code(all_labels.get(eth_col, {}), "hispanic", None) if eth_col else None
            if eth_col and hisp is not None:
                race_cols = race_cols + [eth_col]
            labels = all_labels.get(race_col, {})
            black = find_code(labels, "black", a.black_code)
            white = find_code(labels, "white", a.white_code)
            asian = find_codes(labels, "asian")
            # Hispanic: from ethnicityomb where it exists, else from the combined 2002-2006 answer
            hisp_in_race = find_code(labels, "hispanic", None) if not (eth_col and hisp is not None) else None
            print(f"  Race: {race_col}")
            if labels:
                print("  Race codes in this file:", labels)
            if black is None or white is None:
                sys.exit("  Couldn't tell which race codes mean Black and White. "
                         "Re-run with --black-code and --white-code using the codes listed above.")
            print(f"  Using Black = {black}, White = {white}, Asian = {asian}; Hispanic = "
                  + (f"{eth_col} = {hisp} (excluded from the race groups)" if eth_col and hisp is not None
                     else f"{race_col} = {hisp_in_race}" if hisp_in_race is not None else "not available"))

        has_rt = all(c in cols for c in RT_COLS + [ORDER_COL])
        if has_rt:
            print(f"  Response times: {', '.join(RT_COLS)} with {ORDER_COL}"
                  + (f" (codes: {all_labels[ORDER_COL]})" if all_labels.get(ORDER_COL) else ""))
        else:
            print("  Response times: block latency columns not found; skipping")

        # A session counts when it has a D-score: only participants who completed the IAT
        # blocks have one (OSF readme). session_status is not used, because its codes
        # change between years (2025 marks nearly every session "R", not "C").
        usecols = [d_col, st_col] + race_cols + (RT_COLS + [ORDER_COL] if has_rt else [])
        # Per Order code: how often "later blocks slower" agrees with D > 0, and
        # per (state, cohort, order): n, sum of early-block ms, sum of late-block ms.
        order_votes = defaultdict(lambda: [0, 0])
        rt_raw = defaultdict(lambda: [0, 0.0, 0.0])
        kept = 0
        for df in iter_chunks(path, usecols):
            d = pd.to_numeric(df[d_col], errors="coerce")
            st = df[st_col].astype(str).str.strip().str.upper()
            ok = d.notna() & st.isin(STATES) & d.between(-2, 2)
            df, d, st = df[ok], d[ok], st[ok]
            band = band_of(d)
            kept += len(d)

            if tick_mode:
                ticked = {c: pd.to_numeric(df[c], errors="coerce") == 1 for c in race_cols}
                n_ticked = sum(ticked.values())
                only = lambda r: ticked.get(f"{RACE_TICK_PREFIX}{r}", False) & (n_ticked == 1)
                is_white, is_black, is_asian = only("white"), only("black"), only("asian")
                is_hisp = ticked.get(f"{RACE_TICK_PREFIX}hispanic", pd.Series(False, index=df.index))
            else:
                race = pd.to_numeric(df[race_col], errors="coerce")
                is_white, is_black, is_asian = race == white, race == black, race.isin(asian)
                if eth_col and hisp is not None:
                    is_hisp = pd.to_numeric(df[eth_col], errors="coerce") == hisp
                    is_white, is_black, is_asian = is_white & ~is_hisp, is_black & ~is_hisp, is_asian & ~is_hisp
                elif hisp_in_race is not None:
                    is_hisp = race == hisp_in_race
                else:
                    is_hisp = pd.Series(False, index=df.index)
            cohorts = (("all", pd.Series(True, index=d.index)), ("black", is_black), ("white", is_white),
                       ("asian", is_asian), ("hispanic", is_hisp))

            if has_rt:
                rt = df[RT_COLS].apply(pd.to_numeric, errors="coerce")
                order = pd.to_numeric(df[ORDER_COL], errors="coerce")
                rt_ok = rt.notna().all(axis=1) & rt.ge(RT_RANGE[0]).all(axis=1) \
                    & rt.le(RT_RANGE[1]).all(axis=1) & order.notna() & (order == order.round())  # whole-number codes only
                early = (rt["Mn_RT_all_3"] + rt["Mn_RT_all_4"]) / 2   # blocks 3 & 4
                late = (rt["Mn_RT_all_6"] + rt["Mn_RT_all_7"]) / 2    # blocks 6 & 7
                vote = rt_ok & (d != 0) & (late != early)
                agree = ((late > early) == (d > 0))
                for o, grp in agree[vote].groupby(order[vote]):
                    order_votes[o][0] += int(grp.sum()); order_votes[o][1] += int(len(grp))

            for cohort, mask in cohorts:
                g = pd.DataFrame({"s": st[mask], "d": d[mask]}).groupby("s")["d"]
                for s, n, sm, sq in zip(g.size().index, g.size(), g.sum(), (g.apply(lambda x: (x ** 2).sum()))):
                    cell = acc[(s, cohort)]
                    cell[0] += int(n); cell[1] += float(sm); cell[2] += float(sq)
                counts = pd.DataFrame({"s": st[mask], "b": band[mask]}).groupby(["s", "b"]).size()
                for (s, b), n in counts.items():
                    band_acc[(s, cohort)][b] += int(n)
                if has_rt:
                    m = mask & rt_ok
                    rg = pd.DataFrame({"s": st[m], "o": order[m], "e": early[m], "l": late[m]}).groupby(["s", "o"])
                    for (s, o), row in rg.agg(n=("e", "size"), e=("e", "sum"), l=("l", "sum")).iterrows():
                        cell = rt_raw[(s, cohort, o)]
                        cell[0] += int(row["n"]); cell[1] += float(row["e"]); cell[2] += float(row["l"])

        report.append((name, kept))
        print(f"  Kept {kept:,} US sessions with a state and a D-score")
        print("  Running totals: " + "  ".join(f"{c} {acc_total(acc, c):,}"
                                               for c in ("white", "black", "asian", "hispanic")))

        if has_rt:
            # Decide, per Order code, whether blocks 3 & 4 were White+good.
            early_is_white = {}
            for o, (ag, n) in sorted(order_votes.items()):
                share = ag / n if n else 0.5
                if n < 100:
                    pass  # too few sessions to tell
                elif share >= 0.65:
                    early_is_white[o] = True
                elif share <= 0.35:
                    early_is_white[o] = False
                verdict = ("blocks 3&4 = White+good" if early_is_white.get(o) is True else
                           "blocks 3&4 = Black+good" if early_is_white.get(o) is False else
                           "UNCLEAR, response times for this code skipped")
                print(f"  Order {o:g}: {share:.0%} of {n:,} sessions agree with the D-score -> {verdict}")
            for (s, cohort, o), (n, e, l) in rt_raw.items():
                if o not in early_is_white:
                    continue
                w, b = (e, l) if early_is_white[o] else (l, e)
                cell = rt_acc[(s, cohort)]
                cell[0] += n; cell[1] += w; cell[2] += b

    # National ("US") totals: all states combined, per cohort
    for (s, cohort), (n, sm, sq) in list(acc.items()):
        us = acc[("US", cohort)]
        us[0] += n; us[1] += sm; us[2] += sq
        for b, c in band_acc[(s, cohort)].items():
            band_acc[("US", cohort)][b] += c
    for (s, cohort), (n, w, b) in list(rt_acc.items()):
        us = rt_acc[("US", cohort)]
        us[0] += n; us[1] += w; us[2] += b

    out = {"source_files": [r[0] for r in report], "min_n": a.min_n,
           "bands": BANDS, "states": {}}
    for (s, cohort), (n, sm, sq) in sorted(acc.items()):
        mean = sm / n
        sd = math.sqrt(max(sq / n - mean ** 2, 0))
        out["states"].setdefault(s, {})[cohort] = {
            "n": n, "mean_d": round(mean, 4), "sd": round(sd, 4), "low_n": n < a.min_n,
            "band_counts": band_acc[(s, cohort)]}
        rn, w, b = rt_acc.get((s, cohort), (0, 0, 0))
        if rn:
            out["states"][s][cohort].update({
                "n_rt": rn, "rt_white_good_ms": round(w / rn), "rt_black_good_ms": round(b / rn)})
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)
    print(f"\nDone. Wrote {a.out} ({os.path.getsize(a.out):,} bytes). Upload that file.")


if __name__ == "__main__":
    main()

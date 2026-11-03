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
slight/moderate/strong pro-Black), for the US total the mean D-score by age
group and by education level, where the file has block latencies the
average response time in the White+good and Black+good blocks, and the mean
self-reported preference ("explicit_pref") rescaled to -1 (strongly prefer
African Americans) .. 0 (no preference) .. +1 (strongly prefer European
Americans), so the 5-point (2002-2006) and 7-point versions share a scale.

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
# Self-reported preference ("Which statement best describes you?"), from strongly
# prefer African Americans (1) to strongly prefer European Americans (5 or 7).
# 2002-2006 use a 5-point version (att_5); later years a 7-point one (att_7, att7).
EXPLICIT_COLS = ["att7", "att_7", "att_5"]

# National breakdowns by age and education (US totals only).
# Age: "age" where recorded (2002-2015), otherwise session year minus birth year.
AGE_GROUPS = [(18, 24, "18-24"), (25, 34, "25-34"), (35, 44, "35-44"),
              (45, 54, "45-54"), (55, 64, "55-64"), (65, 99, "65+")]
# Education: edu_14 (same 14 codes in every year).
EDU_GROUPS = {"High school or less": [1, 2, 3, 4], "Some college / associate": [5, 6],
              "Bachelor's": [7], "Master's / other graduate": [8, 9, 13, 14],
              "Ph.D. / J.D. / M.D.": [10, 11, 12]}
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


def num(code):
    """Codes are sometimes stored as text ("1" in 2022); compare them as numbers."""
    if code is None:
        return None
    try:
        return float(code)
    except (TypeError, ValueError):
        return code


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
    # explicit preference per (state, cohort): n, sum of rescaled answers
    exp_acc = defaultdict(lambda: [0, 0.0])
    # US totals per (cohort, "age"/"edu", group): n, sum of D
    grp_acc = defaultdict(lambda: [0, 0.0])
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
        # Some years switch race question mid-year (2006: "ethnic" then "raceomb"; 2016:
        # "raceomb" all year plus "raceomb_002" from September; 2023: "raceomb002" until
        # March, then tick boxes). Each respondent is classified by the first of these
        # they actually answered, tick boxes first.
        race_cols = list(tick_cols) if tick_mode else []
        if tick_mode:
            print(f"  Race: tick boxes ({', '.join(c[len(RACE_TICK_PREFIX):] for c in tick_cols)}); "
                  "a box counts as ticked when its value is 1")
            for c in (f"{RACE_TICK_PREFIX}white", f"{RACE_TICK_PREFIX}black",
                      f"{RACE_TICK_PREFIX}asian", f"{RACE_TICK_PREFIX}hispanic"):
                if all_labels.get(c):
                    print(f"  Codes for {c}: {all_labels[c]}")
        race_names = [a.race_col] if a.race_col else [c for c in RACE_CANDIDATES if c in cols]
        if not race_names and not tick_mode:
            pick(cols, RACE_CANDIDATES, None, "race")  # exits with the column list
        # Hispanic (any race) from the separate ethnicity question, where it exists.
        eth_col = "ethnicityomb" if "ethnicityomb" in cols else None
        hisp = num(find_code(all_labels.get(eth_col, {}), "hispanic", None)) if eth_col else None
        if hisp is None:
            eth_col = None
        schemes = []
        for col in race_names:
            labels = all_labels.get(col, {})
            sch = {"col": col,
                   "black": num(find_code(labels, "black", a.black_code)),
                   "white": num(find_code(labels, "white", a.white_code)),
                   "asian": [num(c) for c in find_codes(labels, "asian")],
                   "hisp": num(find_code(labels, "hispanic", None))}  # 2002-2006 combined answer
            print(f"  Race: {col}" + (f"  codes: {labels}" if labels else ""))
            if sch["black"] is None or sch["white"] is None:
                sys.exit(f"  Couldn't tell which {col} codes mean Black and White. "
                         "Re-run with --race-col, --black-code and --white-code.")
            print(f"    Using Black = {sch['black']:g}, White = {sch['white']:g}, "
                  f"Asian = {[f'{c:g}' for c in sch['asian']]}, Hispanic = "
                  + (f"{sch['hisp']:g}" if sch["hisp"] is not None else
                     f"{eth_col} = {hisp:g}" if eth_col else "not available"))
            schemes.append(sch)
        race_cols += race_names + ([eth_col] if eth_col else [])

        has_rt = all(c in cols for c in RT_COLS + [ORDER_COL])
        if has_rt:
            print(f"  Response times: {', '.join(RT_COLS)} with {ORDER_COL}"
                  + (f" (codes: {all_labels[ORDER_COL]})" if all_labels.get(ORDER_COL) else ""))
        else:
            print("  Response times: block latency columns not found; skipping")

        age_cols = [c for c in ("age", "birthyear", "year") if c in cols]
        # edu_14 in every year from 2004; part of 2015-2016 records it only as
        # "edunotstudent" (same 14 codes), used where edu_14 is blank.
        edu_cols = [c for c in ("edu_14", "edunotstudent") if c in cols]
        print(f"  Age: {', '.join(age_cols) or 'not found'}   Education: {', '.join(edu_cols) or 'not found'}")

        explicit = []  # (column, number of scale points)
        for c in EXPLICIT_COLS:
            if c not in cols:
                continue
            lab = {num(k): str(v).lower() for k, v in all_labels.get(c, {}).items()}
            k = int(max(lab)) if lab else None
            if k in (5, 7) and "african" in lab.get(1.0, "") and "european" in lab.get(float(k), ""):
                explicit.append((c, k))
            elif not lab and c in ("att7", "att_7"):
                # 2015 and 2017 files carry this item without value labels; it is the same
                # 7-point question as in the labelled years on either side.
                explicit.append((c, 7))
                print(f"  Explicit: {c} has no value labels; using the 7-point coding of other years")
            else:
                print(f"  Explicit: skipping {c} (unexpected labels: {all_labels.get(c)})")
        print("  Explicit preference: " + (", ".join(f"{c} ({k}-point)" for c, k in explicit) or "not found"))

        # A session counts when it has a D-score: only participants who completed the IAT
        # blocks have one (OSF readme). session_status is not used, because its codes
        # change between years (2025 marks nearly every session "R", not "C").
        usecols = ([d_col, st_col] + race_cols + (RT_COLS + [ORDER_COL] if has_rt else [])
                   + [c for c, _ in explicit] + age_cols + edu_cols)
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

            none = pd.Series(False, index=df.index)
            is_white, is_black, is_asian, is_hisp = none.copy(), none.copy(), none.copy(), none.copy()
            done = none.copy()
            if tick_mode:
                ticked = {c: pd.to_numeric(df[c], errors="coerce") == 1 for c in tick_cols}
                n_ticked = sum(ticked.values())
                only = lambda r: ticked.get(f"{RACE_TICK_PREFIX}{r}", none) & (n_ticked == 1)
                is_white, is_black, is_asian = only("white"), only("black"), only("asian")
                is_hisp = ticked.get(f"{RACE_TICK_PREFIX}hispanic", none).copy()
                done = n_ticked > 0
            if eth_col:
                is_hisp |= ~done & (pd.to_numeric(df[eth_col], errors="coerce") == hisp)
            for sch in schemes:
                race = pd.to_numeric(df[sch["col"]], errors="coerce")
                rows = race.notna() & ~done
                is_white |= rows & (race == sch["white"])
                is_black |= rows & (race == sch["black"])
                is_asian |= rows & race.isin(sch["asian"])
                if sch["hisp"] is not None:
                    is_hisp |= rows & (race == sch["hisp"])
                done |= rows
            # race groups are "that race only, not Hispanic"
            is_white, is_black, is_asian = is_white & ~is_hisp, is_black & ~is_hisp, is_asian & ~is_hisp
            age = pd.Series(np.nan, index=df.index)
            if "age" in df:
                age = pd.to_numeric(df["age"], errors="coerce")
            if "birthyear" in df and "year" in df:
                age = age.fillna(pd.to_numeric(df["year"], errors="coerce")
                                 - pd.to_numeric(df["birthyear"], errors="coerce"))
            age_grp = pd.Series(None, index=df.index, dtype=object)
            for lo, hi, label in AGE_GROUPS:
                age_grp[(age >= lo) & (age <= hi)] = label
            edu_grp = pd.Series(None, index=df.index, dtype=object)
            if edu_cols:
                edu = pd.Series(np.nan, index=df.index)
                for c in edu_cols:
                    edu = edu.fillna(pd.to_numeric(df[c], errors="coerce"))
                for label, codes in EDU_GROUPS.items():
                    edu_grp[edu.isin(codes)] = label

            expl = pd.Series(np.nan, index=df.index)
            for c, k in explicit:  # first column answered wins
                mid = (k + 1) / 2
                v = pd.to_numeric(df[c], errors="coerce")
                v = v.where(v.between(1, k))
                expl = expl.fillna((v - mid) / (mid - 1))

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
                for kind, grp in (("age", age_grp), ("edu", edu_grp)):
                    gg = pd.DataFrame({"g": grp[mask], "d": d[mask]}).dropna().groupby("g")["d"]
                    for gname, n, sm in zip(gg.size().index, gg.size(), gg.sum()):
                        cell = grp_acc[(cohort, kind, gname)]
                        cell[0] += int(n); cell[1] += float(sm)
                if explicit:
                    eg = pd.DataFrame({"s": st[mask], "e": expl[mask]}).dropna().groupby("s")["e"]
                    for s, n, sm in zip(eg.size().index, eg.size(), eg.sum()):
                        cell = exp_acc[(s, cohort)]
                        cell[0] += int(n); cell[1] += float(sm)
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
    for (s, cohort), (n, sm) in list(exp_acc.items()):
        us = exp_acc[("US", cohort)]
        us[0] += n; us[1] += sm
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
        if s == "US":
            for kind, groups in (("age", [g for _, _, g in AGE_GROUPS]), ("edu", list(EDU_GROUPS))):
                vals = {g: grp_acc[(cohort, kind, g)] for g in groups}
                out["states"][s][cohort][f"{kind}_groups"] = {
                    g: {"n": n, "mean_d": round(sm / n, 4) if n else None} for g, (n, sm) in vals.items()}
        en, esum = exp_acc.get((s, cohort), (0, 0))
        if en:
            out["states"][s][cohort].update({"n_explicit": en, "explicit_pref": round(esum / en, 4)})
        rn, w, b = rt_acc.get((s, cohort), (0, 0, 0))
        if rn:
            out["states"][s][cohort].update({
                "n_rt": rn, "rt_white_good_ms": round(w / rn), "rt_black_good_ms": round(b / rn)})
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)
    print(f"\nDone. Wrote {a.out} ({os.path.getsize(a.out):,} bytes). Upload that file.")


if __name__ == "__main__":
    main()

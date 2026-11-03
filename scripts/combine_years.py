"""
Combine per-year outputs of summarize_race_iat_by_state.py into one file
for the dashboard.

Usage:
    python combine_years.py y2007.json y2008.json ... --from 2007 --to 2025 --out ../data/state_cohort_iat.json

Each input file name must contain its year (e.g. y2019.json, y2002-2003.json;
a range counts as its last year). The output has:
  - "states": the same per-state / per-cohort fields as the yearly files,
    pooled over the chosen years (every session weighted equally);
  - "us_by_year": national mean D-score and respondent count per year and cohort.
"""
import argparse
import json
import math
import os
import re
from collections import defaultdict


def year_of(path):
    m = re.findall(r"(\d{4})", os.path.basename(path))
    if not m:
        raise SystemExit(f"Can't find a year in the file name: {path}")
    return int(m[-1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--from", dest="start", type=int, default=0)
    ap.add_argument("--to", dest="end", type=int, default=9999)
    ap.add_argument("--min-n", type=int, default=100)
    ap.add_argument("--out", default="state_cohort_iat.json")
    a = ap.parse_args()

    # pooled sums per (state, cohort)
    pool = defaultdict(lambda: {"n": 0, "sum": 0.0, "sumsq": 0.0, "bands": defaultdict(int),
                                "n_rt": 0, "w_sum": 0.0, "b_sum": 0.0})
    us_by_year = {}
    used, bands_order = [], None

    for path in sorted(a.files, key=year_of):
        y = year_of(path)
        data = json.load(open(path))
        cells = data["states"]
        us_by_year[str(y)] = {c: {"n": v["n"], "mean_d": v["mean_d"]}
                              for c, v in cells.get("US", {}).items()}
        if not (a.start <= y <= a.end):
            continue
        used.append(y)
        bands_order = data.get("bands", bands_order)
        for s, cohorts in cells.items():
            for c, v in cohorts.items():
                p = pool[(s, c)]
                n, mean, sd = v["n"], v["mean_d"], v["sd"]
                p["n"] += n
                p["sum"] += mean * n
                p["sumsq"] += (sd ** 2 + mean ** 2) * n
                for b, k in v.get("band_counts", {}).items():
                    p["bands"][b] += k
                if v.get("n_rt"):
                    p["n_rt"] += v["n_rt"]
                    p["w_sum"] += v["rt_white_good_ms"] * v["n_rt"]
                    p["b_sum"] += v["rt_black_good_ms"] * v["n_rt"]

    out = {"years": [min(used), max(used)] if used else [], "years_included": used,
           "min_n": a.min_n, "bands": bands_order, "states": {}, "us_by_year": us_by_year}
    for (s, c), p in sorted(pool.items()):
        n = p["n"]
        mean = p["sum"] / n
        cell = {"n": n, "mean_d": round(mean, 4),
                "sd": round(math.sqrt(max(p["sumsq"] / n - mean ** 2, 0)), 4),
                "low_n": n < a.min_n,
                "band_counts": {b: p["bands"].get(b, 0) for b in (bands_order or p["bands"])}}
        if p["n_rt"]:
            cell.update({"n_rt": p["n_rt"],
                         "rt_white_good_ms": round(p["w_sum"] / p["n_rt"]),
                         "rt_black_good_ms": round(p["b_sum"] / p["n_rt"])})
        out["states"].setdefault(s, {})[c] = cell

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(out, f, separators=(",", ":"))
    print(f"Combined {len(used)} year files ({out['years']}) -> {a.out} "
          f"({os.path.getsize(a.out):,} bytes)")


if __name__ == "__main__":
    main()

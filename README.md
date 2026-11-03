# Race IAT Trends & Insights Dashboard

An interactive dashboard of Race Implicit Association Test (IAT) results in the United States, built from Project Implicit's public datasets on the Open Science Framework (OSF).

## What's here

- `index.html` — the dashboard. Open it in a browser; no build step needed.
- `scripts/summarize_race_iat_by_state.py` — summarizes the OSF Race IAT files by state and respondent race (all, Black, White) into a small `state_cohort_iat.json`.

## Running the summary script

```bash
pip install pandas pyreadstat
python scripts/summarize_race_iat_by_state.py path/to/Race.IAT.2024.sav
```

On Python 3.9, install `pyreadstat==1.2.8`: newer versions need Python 3.10 or later.

The script reads both race formats: the single-answer question in older files (`raceomb_002`) and the tick boxes in newer ones (`raceomb_003_white`, `raceomb_003_black`, …). With tick boxes, "White" and "Black" mean that was the only race selected. Where a file has block latencies (`Mn_RT_all_3/4/6/7` and `Order`), the output also includes average White+good and Black+good response times.

Raw OSF files are not stored in this repository (they exceed GitHub's file size limits). Download them from Project Implicit's OSF page.

## Notes on the data

- Positive D-scores mean respondents were faster at associating White with good; negative D-scores mean they were faster at associating Black with good.
- The "White Respondents Only" and "Black Respondents Only" map views are currently scaled from each state's overall average. They will be replaced with measured per-state values from `state_cohort_iat.json`.
- The Implicit Bias Score Distribution chart currently shows fixed national figures that don't match the trend chart. `state_cohort_iat.json` also records how many respondents fall in each D-score band, per state and for the US as a whole, so the chart can show measured, state-specific breakdowns.

## Source

Project Implicit, Race IAT public datasets, Open Science Framework (osf.io).

# Race IAT Trends & Insights Dashboard

An interactive dashboard of Race Implicit Association Test (IAT) results in the United States, built from Project Implicit's public datasets on the Open Science Framework (OSF).

## What's here

- `index.html` — the dashboard. Open it in a browser; no build step needed.
- `data/iat_data.js` (and the same data as `data/state_cohort_iat.json`) — measured values from every public Race IAT file on OSF, 2002–2003 through 2025: per state and for the US as a whole, for all respondents and for White, Black, Asian and Hispanic/Latino respondents, plus national values per year.
- `scripts/summarize_race_iat_by_state.py` — summarizes one or more OSF Race IAT files (`.sav` or `.csv`) by state and respondent group.
- `scripts/combine_years.py` — pools the yearly summaries into the dashboard's data file.

## Rebuilding the data

```bash
pip install pandas pyreadstat
python scripts/summarize_race_iat_by_state.py Race_IAT.public.2024.sav --out y2024.json
# ...one file per year, then:
python scripts/combine_years.py y*.json --out data/state_cohort_iat.json --js data/iat_data.js
```

Use Python 3.10 or later: the 2023 and 2024 `.sav` files need pyreadstat 1.3 or newer, which doesn't support Python 3.9. Raw OSF files are not stored in this repository (they exceed GitHub's file size limits). Download them from Project Implicit's OSF page.

## How the numbers are computed

- **Sessions:** a session counts when it has a D-score (only participants who completed the IAT blocks have one) and a US state. `session_status` is not used, because its codes change between years.
- **Groups:** White, Black and Asian mean that race only and not Hispanic. Hispanic/Latino covers any race. The race question changes over the years (a combined race/ethnicity answer in 2002–2006, a single race answer plus a separate Hispanic question in 2007–2022, tick boxes from 2023, with switches part-way through 2006, 2016 and 2023); each respondent is classified by the question they answered. In 2002–2006, Asian includes Pacific Islander.
- **D-score:** positive means faster at associating White with good. Bands use the standard 0.15 / 0.35 / 0.65 cut-offs.
- **Response times:** mean latency of blocks 3 & 4 and blocks 6 & 7, matched to White+good / Black+good using the `Order` column. Each Order code's meaning is checked against the D-score rather than assumed; block means outside 200–10,000 ms are skipped.
- **Explicit preference:** the "Which statement best describes you?" item, rescaled to −1 (strongly prefer African Americans) … 0 … +1 (strongly prefer European Americans). 2002–2006 asked a 5-point version, later years a 7-point one.
- **Map:** all years pooled, so years with more sessions weigh more. Groups with fewer than 100 respondents in a state are marked as low sample.
- **Not yet from the data:** the age and education charts show illustrative values.

## Source

Project Implicit, Race IAT public datasets, Open Science Framework (osf.io).

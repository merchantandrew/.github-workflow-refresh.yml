# NFL model: automated dashboard

Python port of the "Magic Sheet" scoring model, with a walk-forward backtest, a
Streamlit dashboard, and a GitHub Actions job that refreshes everything on a schedule.

## How it fits together

    nflverse (free CSVs)  ->  nfl_model.pipeline  ->  outputs/predictions.csv  ->  app.py (dashboard)
      schedule, scores,        projects every game,       (committed to the repo
      lines, team stats        grades past picks           by the scheduled job)
                                       |
                                       +-> optional: "Dash - Slate" / "Dash - Record" tabs in your Google Sheet

## Run it locally

    pip install -r requirements.txt
    python -m pytest                  # confirms the port reproduces the sheet's Rams @ Panthers numbers
    python -m nfl_model.pipeline      # writes outputs/predictions.csv and outputs/meta.json
    streamlit run app.py

Options: `--seasons 2025 2026`, `--shrink 4` (pull early-season averages toward league
average), `--hfa 2` (home-field points), `--spread-edge 4`, `--total-edge 4`.
The defaults (`--shrink 0 --hfa 0`) reproduce the sheet exactly.

## Put it online and automate it

1. Create a **private** GitHub repo and push this folder to it (the free plan is enough).
2. The workflow in `.github/workflows/refresh.yml` runs Tuesday mornings (after Monday
   Night Football) and Thursday/Saturday/Sunday mornings (fresh lines). It runs the
   tests, rebuilds `outputs/`, and commits the result. "Run workflow" in the Actions tab
   triggers it by hand.
3. Host the dashboard free on Streamlit Community Cloud: connect the repo, set the main
   file to `app.py`. It redeploys each time the job commits new outputs.

## Optional: write into your Google Sheet too

See the docstring in `nfl_model/sheets_push.py` (service account, share the sheet with
it, add two repo secrets, switch the workflow step to `--push-sheet`).

## Things to know

- **Sign convention:** nflverse `spread_line` is the expected HOME margin (positive = home
  favoured). The sheet shows the same number from the away side. Picks are identical; the
  dashboard displays spreads the sportsbook way (negative = favourite).
- **No lookahead:** each game is projected from weeks before it only, so the history is an
  honest backtest. This is tested in `tests/test_model.py`.
- **Lines:** nflverse's `spread_line`/`total_line` are good for grading and backtests, but
  for upcoming games they may lag the market. For bet-time lines, add an odds API as a
  second source and overwrite those two columns before `add_picks`.
- **2-point conversions:** passing + rushing only. Passing and receiving credit the same
  conversion, so adding receiving as well would double count.
- **Column names:** if an upstream nflverse file changes a column, the loader raises a
  message naming it; the fix lives in `nfl_model/data.py`.
- **Sample size:** a 42-game record is not evidence either way. Use the "Edge size" tab to
  check that bigger edges win more often, and "Model vs Vegas" to see whether the model's
  error beats the market's.

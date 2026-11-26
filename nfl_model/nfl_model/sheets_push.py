"""Optional: write the current slate and record tables into your Google Sheet.

Setup (one time):
  1. Google Cloud console -> create a service account, download its JSON key.
  2. Share your Google Sheet with the service account's email (Editor).
  3. GitHub repo -> Settings -> Secrets: SHEET_ID and GOOGLE_SERVICE_ACCOUNT_JSON
     (paste the whole JSON). Then run the pipeline with --push-sheet.

It only writes to tabs named "Dash - Slate" and "Dash - Record" (created if
missing), so your existing model tabs are untouched.
"""
from __future__ import annotations

import json

import pandas as pd

from .summarize import by_week, edge_buckets


def _write(sh, title: str, df: pd.DataFrame) -> None:
    import gspread

    try:
        ws = sh.worksheet(title)
        ws.clear()
    except gspread.WorksheetNotFound:
        ws = sh.add_worksheet(title, rows=max(len(df) + 5, 100), cols=max(len(df.columns), 10))
    df = df.astype(object).where(df.notna(), "")
    ws.update([df.columns.tolist()] + df.values.tolist())


def push(df: pd.DataFrame, sheet_id: str, service_account_json: str) -> None:
    import gspread

    gc = gspread.service_account_from_dict(json.loads(service_account_json))
    sh = gc.open_by_key(sheet_id)

    latest = df[df["season"] == df["season"].max()]
    upcoming = latest[latest["spread_result"].isna()]
    if not upcoming.empty:
        upcoming = upcoming[upcoming["week"] == upcoming["week"].min()]
    slate = upcoming[
        ["week", "gameday", "away_team", "home_team", "spread_line", "model_home_margin",
         "spread_edge", "spread_pick", "total_line", "model_total", "total_edge", "total_pick"]
    ].round(2)
    _write(sh, "Dash - Slate", slate)

    parts = [
        by_week(latest, "spread_result").assign(market="Spread"),
        by_week(latest, "total_result").assign(market="Total"),
    ]
    buckets = edge_buckets(df, "spread_edge", "spread_result").assign(market="Spread by edge")
    rec = pd.concat(parts + [buckets], ignore_index=True)
    _write(sh, "Dash - Record", rec.round(3))

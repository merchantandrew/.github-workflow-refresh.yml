"""Refresh job: load data, project every game, grade, write outputs.

    python -m nfl_model.pipeline                   # current + previous season
    python -m nfl_model.pipeline --seasons 2025 2026 --shrink 4 --hfa 2

Each game is projected using ONLY weeks before that game's week, so the
history it writes is an honest walk-forward backtest, not hindsight.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path

import pandas as pd

from .model import add_picks, grade, project_game, team_rates


def default_season(today: dt.date | None = None) -> int:
    today = today or dt.date.today()
    return today.year if today.month >= 3 else today.year - 1


def build_predictions(
    games: pd.DataFrame,
    stats: pd.DataFrame,
    seasons: list[int],
    shrink_games: float = 0.0,
    home_field_pts: float = 0.0,
    min_games: int = 3,
    spread_edge_min: float = 4.0,
    total_edge_min: float = 4.0,
) -> pd.DataFrame:
    g = games[(games["game_type"] == "REG") & games["season"].isin(seasons)]
    rows = []
    for (season, week), wk in g.groupby(["season", "week"]):
        if week <= min_games:  # not enough games played for stable averages
            continue
        rates = team_rates(stats, season, week, shrink_games)
        if rates.empty:
            continue
        for r in wk.itertuples(index=False):
            if r.away_team not in rates.index or r.home_team not in rates.index:
                continue
            p = project_game(rates, r.away_team, r.home_team, home_field_pts)
            rows.append(
                {
                    "game_id": r.game_id,
                    "season": season,
                    "week": week,
                    "gameday": r.gameday,
                    "gametime": r.gametime,
                    "away_team": r.away_team,
                    "home_team": r.home_team,
                    "spread_line": r.spread_line,
                    "total_line": r.total_line,
                    "away_score": r.away_score,
                    "home_score": r.home_score,
                    **p,
                }
            )
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = add_picks(df, spread_edge_min, total_edge_min)
    return grade(df)


def main() -> None:
    from .data import load_games, load_team_stats

    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", type=int, nargs="+")
    ap.add_argument("--shrink", type=float, default=0.0, help="extra league-average games")
    ap.add_argument("--hfa", type=float, default=0.0, help="home-field points (sheet = 0)")
    ap.add_argument("--min-games", type=int, default=3)
    ap.add_argument("--spread-edge", type=float, default=4.0)
    ap.add_argument("--total-edge", type=float, default=4.0)
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--push-sheet", action="store_true",
                    help="also write summary tabs to Google Sheets (needs env vars)")
    a = ap.parse_args()

    cur = default_season()
    seasons = a.seasons or [cur - 1, cur]
    games = load_games(seasons)
    stats = load_team_stats(seasons)
    df = build_predictions(
        games, stats, seasons, a.shrink, a.hfa, a.min_games, a.spread_edge, a.total_edge
    )

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "predictions.csv", index=False)
    meta = {
        "updated_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "seasons": seasons,
        "shrink_games": a.shrink,
        "home_field_pts": a.hfa,
        "spread_edge_min": a.spread_edge,
        "total_edge_min": a.total_edge,
        "games": int(len(df)),
        "graded_games": int(df["spread_result"].notna().sum()) if len(df) else 0,
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))

    if a.push_sheet:
        from .sheets_push import push

        push(df, os.environ["SHEET_ID"], os.environ["GOOGLE_SERVICE_ACCOUNT_JSON"])


if __name__ == "__main__":
    main()

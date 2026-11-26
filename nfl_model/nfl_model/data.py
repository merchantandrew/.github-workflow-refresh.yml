"""Load schedule/lines and team-week stats from nflverse (free, no API key).

If a column name below ever changes upstream, fix it in one place here.
"""
from __future__ import annotations

import pandas as pd

GAMES_URL = "https://github.com/nflverse/nfldata/raw/master/data/games.csv"
TEAM_STATS_URL = (
    "https://github.com/nflverse/nflverse-data/releases/download/"
    "stats_team/stats_team_week_{season}.csv"
)

GAME_COLS = [
    "game_id", "season", "game_type", "week", "gameday", "gametime",
    "away_team", "away_score", "home_team", "home_score",
    "spread_line", "total_line",
]
STAT_COLS = [
    "season", "week", "team", "opponent_team", "attempts", "passing_tds",
    "carries", "rushing_tds", "fg_made", "pat_made",
    "passing_2pt_conversions", "rushing_2pt_conversions",
]


def _require(df: pd.DataFrame, cols: list[str], what: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(
            f"{what} is missing expected columns {missing}. "
            "nflverse may have renamed them; update nfl_model/data.py."
        )


def load_games(seasons: list[int]) -> pd.DataFrame:
    """Schedule, final scores and closing/current lines.

    nflverse convention: spread_line > 0 means the HOME team is favoured.
    """
    g = pd.read_csv(GAMES_URL)
    _require(g, GAME_COLS, "games.csv")
    g = g[g["season"].isin(seasons)][GAME_COLS].copy()
    return g.reset_index(drop=True)


def load_team_stats(seasons: list[int]) -> pd.DataFrame:
    frames = []
    for season in seasons:
        try:
            f = pd.read_csv(TEAM_STATS_URL.format(season=season))
        except Exception as exc:  # season not published yet
            print(f"warning: no team stats for {season}: {exc}")
            continue
        _require(f, STAT_COLS, f"stats_team_week_{season}.csv")
        keep = STAT_COLS + (["season_type"] if "season_type" in f.columns else [])
        frames.append(f[keep])
    if not frames:
        raise RuntimeError("No team stats could be loaded.")
    s = pd.concat(frames, ignore_index=True)
    s = s.fillna(
        {c: 0 for c in STAT_COLS if c not in ("season", "week", "team", "opponent_team")}
    )
    # A 2-point conversion is credited to both passer and receiver, so count
    # passing + rushing only (counting receiving too would double count).
    s["two_pt"] = s["passing_2pt_conversions"] + s["rushing_2pt_conversions"]
    return s

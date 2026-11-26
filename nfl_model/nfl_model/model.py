"""Port of the "2025 NFL Season Magic Sheet" scoring model.

Sheet logic (Output tab), reproduced exactly:

  matchup blend   = (team offensive rate + opponent rate allowed) / 2
                    for TD-per-pass-attempt, pass attempts/game,
                    TD-per-rush-attempt, rush attempts/game
  kicking         = team's own FG, XP and 2pt per game (no opponent blend)
  projected score = passTD% * passAtt * 6 + rushTD% * rushAtt * 6
                    + FG * 3 + XP * 1 + 2PT * 2

Sign convention in this package follows nflverse: ``spread_line`` is the
expected HOME margin (positive = home favoured). The sheet shows the same
number from the away team's point of view; picks are identical.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TD_PTS, FG_PTS, XP_PTS, TWO_PTS = 6, 3, 1, 2

OFF_COLS = ["pass_td_rate", "pass_att", "rush_td_rate", "rush_att", "fg", "xp", "two"]
DEF_COLS = [
    "pass_td_rate_allowed",
    "pass_att_allowed",
    "rush_td_rate_allowed",
    "rush_att_allowed",
]
RATE_COLS = OFF_COLS + DEF_COLS


def team_rates(
    stats: pd.DataFrame, season: int, before_week: int, shrink_games: float = 0.0
) -> pd.DataFrame:
    """Per-team offensive and defensive rates using only weeks < before_week.

    ``stats`` is one row per team-week with columns: season, week, team,
    opponent_team, attempts, passing_tds, carries, rushing_tds, fg_made,
    pat_made, two_pt. Defence-allowed numbers are derived from the opponent's
    offensive row, so no separate defensive file is needed.

    shrink_games > 0 pulls each rate toward the league average as if the team
    had that many extra average games (0 reproduces the sheet exactly).
    """
    s = stats[(stats["season"] == season) & (stats["week"] < before_week)]
    if "season_type" in s.columns:
        s = s[s["season_type"] == "REG"]
    if s.empty:
        return pd.DataFrame(columns=RATE_COLS + ["games"])

    off = s.groupby("team").agg(
        games=("week", "nunique"),
        pass_att=("attempts", "sum"),
        pass_td=("passing_tds", "sum"),
        rush_att=("carries", "sum"),
        rush_td=("rushing_tds", "sum"),
        fg=("fg_made", "sum"),
        xp=("pat_made", "sum"),
        two=("two_pt", "sum"),
    )
    dfn = (
        s.assign(team=s["opponent_team"])
        .groupby("team")
        .agg(
            pass_att_a=("attempts", "sum"),
            pass_td_a=("passing_tds", "sum"),
            rush_att_a=("carries", "sum"),
            rush_td_a=("rushing_tds", "sum"),
        )
    )
    t = off.join(dfn, how="inner")
    n = t["games"]

    out = pd.DataFrame(index=t.index)
    out["games"] = n
    out["pass_td_rate"] = t["pass_td"] / t["pass_att"]
    out["pass_att"] = t["pass_att"] / n
    out["rush_td_rate"] = t["rush_td"] / t["rush_att"]
    out["rush_att"] = t["rush_att"] / n
    out["fg"] = t["fg"] / n
    out["xp"] = t["xp"] / n
    out["two"] = t["two"] / n
    out["pass_td_rate_allowed"] = t["pass_td_a"] / t["pass_att_a"]
    out["pass_att_allowed"] = t["pass_att_a"] / n
    out["rush_td_rate_allowed"] = t["rush_td_a"] / t["rush_att_a"]
    out["rush_att_allowed"] = t["rush_att_a"] / n

    if shrink_games > 0:
        league = out[RATE_COLS].mean()
        w = n / (n + shrink_games)
        for c in RATE_COLS:
            out[c] = w * out[c] + (1 - w) * league[c]
    return out


def _points(off: pd.Series, opp: pd.Series) -> float:
    pass_td = (off["pass_td_rate"] + opp["pass_td_rate_allowed"]) / 2
    pass_att = (off["pass_att"] + opp["pass_att_allowed"]) / 2
    rush_td = (off["rush_td_rate"] + opp["rush_td_rate_allowed"]) / 2
    rush_att = (off["rush_att"] + opp["rush_att_allowed"]) / 2
    return (
        pass_td * pass_att * TD_PTS
        + rush_td * rush_att * TD_PTS
        + off["fg"] * FG_PTS
        + off["xp"] * XP_PTS
        + off["two"] * TWO_PTS
    )


def project_game(
    rates: pd.DataFrame, away: str, home: str, home_field_pts: float = 0.0
) -> dict:
    """Projected score for one game. home_field_pts=0 matches the sheet."""
    a = _points(rates.loc[away], rates.loc[home])
    h = _points(rates.loc[home], rates.loc[away])
    a -= home_field_pts / 2
    h += home_field_pts / 2
    return {
        "model_away_pts": a,
        "model_home_pts": h,
        "model_total": a + h,
        "model_home_margin": h - a,
    }


def add_picks(
    df: pd.DataFrame, spread_edge_min: float = 4.0, total_edge_min: float = 4.0
) -> pd.DataFrame:
    """Add edges, picks and top-pick flags. Needs spread_line / total_line."""
    df = df.copy()
    se = df["model_home_margin"] - df["spread_line"]  # >0: model likes home
    te = df["model_total"] - df["total_line"]  # >0: model likes over
    df["spread_edge"] = se
    df["total_edge"] = te
    df["spread_pick"] = np.where(se > 0, df["home_team"], df["away_team"])
    df.loc[se.isna(), "spread_pick"] = None
    df["total_pick"] = np.where(te > 0, "Over", "Under")
    df.loc[te.isna(), "total_pick"] = None
    df["top_spread"] = se.abs() > spread_edge_min
    df["top_total"] = te.abs() > total_edge_min
    return df


def grade(df: pd.DataFrame) -> pd.DataFrame:
    """Grade picks against final scores. Pushes are labelled 'Push'."""
    df = df.copy()
    played = df["home_score"].notna() & df["away_score"].notna()
    actual_margin = df["home_score"] - df["away_score"]
    actual_total = df["home_score"] + df["away_score"]

    cover = actual_margin - df["spread_line"]  # >0 home covered
    over = actual_total - df["total_line"]  # >0 went over

    def label(outcome: pd.Series, edge: pd.Series) -> pd.Series:
        res = np.where(
            outcome == 0, "Push", np.where((outcome > 0) == (edge > 0), "Win", "Loss")
        )
        res = pd.Series(res, index=df.index, dtype="object")
        res[~played | outcome.isna() | edge.isna()] = None
        return res

    df["actual_home_margin"] = actual_margin.where(played)
    df["actual_total"] = actual_total.where(played)
    df["spread_result"] = label(cover, df["spread_edge"])
    df["total_result"] = label(over, df["total_edge"])
    return df

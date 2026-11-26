import numpy as np
import pandas as pd

from nfl_model.model import RATE_COLS, add_picks, grade, project_game
from nfl_model.pipeline import build_predictions


def test_matches_spreadsheet_rams_at_panthers():
    """Inputs are the Adjusted Stats values from the sheet (Game 273).
    The sheet shows Rams 24.7, Panthers 18.3, total 43.0, projected spread 6.4."""
    rows = {
        "Los Angeles Rams": dict(
            pass_td_rate=0.081, pass_att=33.308, rush_td_rate=0.038, rush_att=26.615,
            fg=1.615, xp=1.538, two=0.0,
            pass_td_rate_allowed=0.039, pass_att_allowed=35.538,
            rush_td_rate_allowed=0.012, rush_att_allowed=26.385),
        "Carolina Panthers": dict(
            pass_td_rate=0.048, pass_att=30.769, rush_td_rate=0.019, rush_att=28.462,
            fg=1.538, xp=2.538, two=0.0,
            pass_td_rate_allowed=0.040, pass_att_allowed=30.692,
            rush_td_rate_allowed=0.047, rush_att_allowed=26.385),
    }
    rates = pd.DataFrame(rows).T[RATE_COLS]
    p = project_game(rates, away="Los Angeles Rams", home="Carolina Panthers")
    assert abs(p["model_away_pts"] - 24.7) < 0.3
    assert abs(p["model_home_pts"] - 18.3) < 0.3
    assert abs(-p["model_home_margin"] - 6.4) < 0.4  # sheet's away-minus-home spread


def test_pick_logic_matches_sheet():
    """Sheet: Vegas has Rams -10, model says Rams by 6.4 -> pick Panthers; total 43 vs 45 -> Under."""
    df = pd.DataFrame([dict(
        away_team="Los Angeles Rams", home_team="Carolina Panthers",
        model_home_margin=-6.4, model_total=43.0, spread_line=-10.0, total_line=45.0,
        away_score=np.nan, home_score=np.nan)])
    out = add_picks(df)
    assert out.loc[0, "spread_pick"] == "Carolina Panthers"
    assert out.loc[0, "total_pick"] == "Under"


def test_grading_including_push():
    df = pd.DataFrame([
        # home -3 favourite wins by 7 -> home covers; model liked home -> Win
        dict(away_team="A", home_team="H", model_home_margin=5, model_total=50, spread_line=3, total_line=44,
             home_score=27, away_score=20),
        # exact push on spread, total over (model said under -> Loss)
        dict(away_team="A", home_team="H", model_home_margin=5, model_total=40, spread_line=3, total_line=44,
             home_score=26, away_score=23),
        # unplayed
        dict(away_team="A", home_team="H", model_home_margin=5, model_total=40, spread_line=3, total_line=44,
             home_score=np.nan, away_score=np.nan),
    ])
    out = grade(add_picks(df))
    assert list(out["spread_result"].fillna("-")) == ["Win", "Push", "-"]
    assert list(out["total_result"].fillna("-")) == ["Win", "Loss", "-"]


def _synthetic(seed=0, n_teams=8, weeks=9):
    rng = np.random.default_rng(seed)
    teams = [f"T{i}" for i in range(n_teams)]
    games, stats = [], []
    for wk in range(1, weeks + 1):
        order = rng.permutation(teams)
        for i in range(0, n_teams, 2):
            a, h = order[i], order[i + 1]
            res = {}
            for t, o in ((a, h), (h, a)):
                res[t] = dict(season=2026, week=wk, team=t, opponent_team=o, season_type="REG",
                              attempts=rng.integers(25, 45), passing_tds=rng.integers(0, 4),
                              carries=rng.integers(18, 35), rushing_tds=rng.integers(0, 3),
                              fg_made=rng.integers(0, 4), pat_made=rng.integers(0, 5), two_pt=0)
                stats.append(res[t])
            hs, as_ = rng.integers(10, 35, 2)
            played = wk < weeks  # last week is "upcoming"
            games.append(dict(game_id=f"{wk}_{a}_{h}", season=2026, game_type="REG", week=wk,
                              gameday="2026-10-01", gametime="13:00", away_team=a, home_team=h,
                              away_score=as_ if played else np.nan, home_score=hs if played else np.nan,
                              spread_line=float(rng.integers(-7, 8)), total_line=float(rng.integers(38, 52))))
    return pd.DataFrame(games), pd.DataFrame(stats)


def test_pipeline_runs_and_has_no_lookahead():
    games, stats = _synthetic()
    df = build_predictions(games, stats, [2026], min_games=3)
    assert df["week"].min() == 4 and df["week"].max() == 9
    assert df.loc[df["week"] == 9, "spread_result"].isna().all()  # upcoming stays ungraded
    assert df.loc[df["week"] < 9, "spread_result"].notna().all()

    # Changing stats from week 6 onward must not change week-6 projections.
    s2 = stats.copy()
    s2.loc[s2["week"] >= 6, ["attempts", "carries"]] += 50
    df2 = build_predictions(games, s2, [2026], min_games=3)
    a = df[df["week"] == 6].set_index("game_id")["model_total"]
    b = df2[df2["week"] == 6].set_index("game_id")["model_total"]
    assert np.allclose(a, b)
    # ...but it should change week 8 (which uses week 6-7 data).
    a8 = df[df["week"] == 8].set_index("game_id")["model_total"]
    b8 = df2[df2["week"] == 8].set_index("game_id")["model_total"]
    assert not np.allclose(a8, b8)


def test_shrinkage_pulls_toward_league_average():
    games, stats = _synthetic()
    raw = build_predictions(games, stats, [2026], shrink_games=0)
    shr = build_predictions(games, stats, [2026], shrink_games=8)
    assert shr["model_total"].std() < raw["model_total"].std()

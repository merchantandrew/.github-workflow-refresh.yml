"""Record tables used by both the dashboard and the Google Sheets push."""
from __future__ import annotations

import numpy as np
import pandas as pd


def record(results: pd.Series) -> dict:
    w = int((results == "Win").sum())
    l = int((results == "Loss").sum())
    p = int((results == "Push").sum())
    return {"W": w, "L": l, "Push": p, "Win %": w / (w + l) if (w + l) else np.nan}


def by_week(df: pd.DataFrame, result_col: str, top_col: str | None = None) -> pd.DataFrame:
    d = df[df[result_col].notna()]
    if top_col:
        d = d[d[top_col]]
    rows = []
    for (season, week), g in d.groupby(["season", "week"]):
        rows.append({"season": season, "week": week, **record(g[result_col])})
    return pd.DataFrame(rows)


def edge_buckets(df: pd.DataFrame, edge_col: str, result_col: str) -> pd.DataFrame:
    d = df[df[result_col].isin(["Win", "Loss"])].copy()
    if d.empty:
        return pd.DataFrame(columns=["edge bucket", "W", "L", "Push", "Win %", "Games"])
    bins = [0, 1, 2, 3, 4, 6, 8, np.inf]
    labels = ["0-1", "1-2", "2-3", "3-4", "4-6", "6-8", "8+"]
    d["edge bucket"] = pd.cut(d[edge_col].abs(), bins=bins, labels=labels, right=False)
    out = []
    for b, g in d.groupby("edge bucket", observed=True):
        r = record(g[result_col])
        out.append({"edge bucket": str(b), **r, "Games": len(g)})
    return pd.DataFrame(out)


def error_vs_market(df: pd.DataFrame) -> dict:
    """Mean absolute error of model and market on margin and total."""
    d = df[df["actual_home_margin"].notna()]
    if d.empty:
        return {}
    return {
        "Games": len(d),
        "Model margin MAE": float((d["model_home_margin"] - d["actual_home_margin"]).abs().mean()),
        "Vegas margin MAE": float((d["spread_line"] - d["actual_home_margin"]).abs().mean()),
        "Model total MAE": float((d["model_total"] - d["actual_total"]).abs().mean()),
        "Vegas total MAE": float((d["total_line"] - d["actual_total"]).abs().mean()),
    }

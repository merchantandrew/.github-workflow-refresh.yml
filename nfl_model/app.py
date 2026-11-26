"""Streamlit dashboard. Run locally:  streamlit run app.py"""
import json
from pathlib import Path

import pandas as pd
import streamlit as st

from nfl_model.summarize import by_week, edge_buckets, error_vs_market, record

OUT = Path(__file__).parent / "outputs"
BREAKEVEN = 110 / 210  # 52.38% at -110

st.set_page_config(page_title="NFL Model Dashboard", layout="wide")


@st.cache_data(ttl=600)
def load():
    df = pd.read_csv(OUT / "predictions.csv")
    meta = json.loads((OUT / "meta.json").read_text())
    return df, meta


try:
    df, meta = load()
except FileNotFoundError:
    st.error("No outputs yet. Run `python -m nfl_model.pipeline` first.")
    st.stop()

st.title("NFL Model Dashboard")
st.caption(
    f"Last refreshed {meta['updated_utc']} UTC · picks need an edge over "
    f"{meta['spread_edge_min']} pts for 'top' status · breakeven at -110 is {BREAKEVEN:.1%}"
)

seasons = sorted(df["season"].unique())
season = st.sidebar.selectbox("Season", seasons, index=len(seasons) - 1)
edge_min = st.sidebar.slider("Top-pick edge threshold (pts)", 0.0, 10.0, float(meta["spread_edge_min"]), 0.5)
sdf = df[df["season"] == season].copy()
sdf["top_spread"] = sdf["spread_edge"].abs() > edge_min
sdf["top_total"] = sdf["total_edge"].abs() > edge_min

tab1, tab2, tab3, tab4 = st.tabs(["This week", "Record", "Edge size", "Model vs Vegas"])

with tab1:
    pending = sdf[sdf["spread_result"].isna()]
    if pending.empty:
        st.info("No upcoming games with lines in the data yet.")
    else:
        weeks = sorted(pending["week"].unique())
        week = st.selectbox("Week", weeks, index=0)
        w = pending[pending["week"] == week].copy()
        # show lines the way sportsbooks do: negative = favourite
        w["Vegas spread (home)"] = -w["spread_line"]
        w["Model spread (home)"] = -w["model_home_margin"]
        w["Game"] = w["away_team"] + " @ " + w["home_team"]
        w = w.sort_values("spread_edge", key=lambda s: -s.abs())
        st.subheader("Spread")
        st.dataframe(
            w[["Game", "gameday", "Vegas spread (home)", "Model spread (home)",
               "spread_edge", "spread_pick", "top_spread"]]
            .rename(columns={"spread_edge": "Edge (pts)", "spread_pick": "Pick", "top_spread": "Top pick"})
            .round(1),
            hide_index=True, use_container_width=True,
        )
        st.subheader("Totals")
        w = w.sort_values("total_edge", key=lambda s: -s.abs())
        st.dataframe(
            w[["Game", "total_line", "model_total", "total_edge", "total_pick", "top_total"]]
            .rename(columns={"total_line": "Vegas total", "model_total": "Model total",
                             "total_edge": "Edge (pts)", "total_pick": "Pick", "top_total": "Top pick"})
            .round(1),
            hide_index=True, use_container_width=True,
        )

with tab2:
    c = st.columns(4)
    for col, (label, rcol, top) in zip(
        c,
        [("Spread, all", "spread_result", None), ("Spread, top picks", "spread_result", "top_spread"),
         ("Totals, all", "total_result", None), ("Totals, top picks", "total_result", "top_total")],
    ):
        d = sdf[sdf[rcol].notna()]
        if top:
            d = d[d[top]]
        r = record(d[rcol])
        col.metric(label, f"{r['W']}-{r['L']}" + (f"-{r['Push']}" if r["Push"] else ""),
                   f"{r['Win %']:.1%}" if r["W"] + r["L"] else "n/a", delta_color="off")
    wk = by_week(sdf, "spread_result").merge(
        by_week(sdf, "total_result"), on=["season", "week"], suffixes=(" spread", " total"), how="outer"
    )
    st.subheader("By week")
    st.dataframe(wk.drop(columns="season").round(3), hide_index=True, use_container_width=True)
    if len(wk):
        st.line_chart(wk.set_index("week")[["Win % spread", "Win % total"]])

with tab3:
    st.caption("All seasons loaded. A working model should win more often as the edge grows.")
    a, b = st.columns(2)
    a.subheader("Spread")
    a.dataframe(edge_buckets(df, "spread_edge", "spread_result").round(3), hide_index=True)
    b.subheader("Totals")
    b.dataframe(edge_buckets(df, "total_edge", "total_result").round(3), hide_index=True)

with tab4:
    err = error_vs_market(sdf)
    if err:
        c = st.columns(2)
        c[0].metric("Margin error: model", f"{err['Model margin MAE']:.2f}",
                    f"{err['Model margin MAE'] - err['Vegas margin MAE']:+.2f} vs Vegas", delta_color="inverse")
        c[1].metric("Total error: model", f"{err['Model total MAE']:.2f}",
                    f"{err['Model total MAE'] - err['Vegas total MAE']:+.2f} vs Vegas", delta_color="inverse")
        st.caption(f"Mean absolute error in points over {err['Games']} graded games. Lower is better; "
                   "beating Vegas error is the real test of a betting model.")
    else:
        st.info("No graded games yet.")

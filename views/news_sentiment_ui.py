"""Beautified charts and table for the Live News Sentiment page."""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from services.news import get_news_sentiment, summarize_sentiment

SENTIMENT_COLORS = {
    "Positive": "#22C55E",
    "Neutral": "#F59E0B",
    "Negative": "#EF4444",
}


def _apply_theme(fig, height=380):
    is_light = st.session_state.get("ui_theme_is_light", True)
    template = "plotly_white" if is_light else "plotly_dark"
    grid = "rgba(30,41,59,0.08)" if is_light else "rgba(255,255,255,0.08)"
    text = "#1E293B" if is_light else "#E8EEF5"
    fig.update_layout(
        template=template,
        height=height,
        margin=dict(l=12, r=12, t=48, b=12),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=text, family="Inter, system-ui, sans-serif"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    fig.update_xaxes(showgrid=True, gridcolor=grid)
    fig.update_yaxes(showgrid=True, gridcolor=grid)
    return fig


def _prepare_display_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["date"] = out["date"].astype(str).replace("NaT", "—")
    out["compound"] = pd.to_numeric(out["compound"], errors="coerce").round(3)
    out = out.rename(columns={
        "date": "Date",
        "time": "Time",
        "headline": "Headline",
        "Sentiment": "Sentiment",
        "compound": "Score",
    })
    return out[["Date", "Time", "Headline", "Sentiment", "Score"]]


def _build_donut_chart(df: pd.DataFrame) -> go.Figure:
    counts = df["Sentiment"].value_counts().reindex(
        ["Positive", "Neutral", "Negative"], fill_value=0
    )
    pie_df = counts.reset_index()
    pie_df.columns = ["Sentiment", "Count"]

    fig = px.pie(
        pie_df,
        values="Count",
        names="Sentiment",
        hole=0.58,
        color="Sentiment",
        color_discrete_map=SENTIMENT_COLORS,
        category_orders={"Sentiment": ["Positive", "Neutral", "Negative"]},
    )
    fig.update_traces(
        textposition="inside",
        textinfo="percent+label",
        textfont_size=13,
        marker=dict(line=dict(color="rgba(255,255,255,0.8)", width=2)),
        pull=[0.03 if c > 0 else 0 for c in pie_df["Count"]],
    )
    total = int(pie_df["Count"].sum())
    fig.update_layout(
        title=dict(text=f"Sentiment mix ({total} headlines)", x=0.5, xanchor="center"),
        showlegend=True,
    )
    return _apply_theme(fig, height=360)


def _build_timeline_chart(df: pd.DataFrame) -> go.Figure:
    """Stacked daily sentiment counts + average compound line."""
    if df.empty or "date" not in df.columns:
        return go.Figure()

    work = df.copy()
    work["date"] = pd.to_datetime(work["date"], errors="coerce")
    work = work.dropna(subset=["date"])
    if work.empty:
        return go.Figure()

    daily_counts = (
        work.groupby(["date", "Sentiment"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=["Positive", "Neutral", "Negative"], fill_value=0)
    )
    daily_compound = work.groupby("date")["compound"].mean()

    fig = go.Figure()
    for sentiment in ["Positive", "Neutral", "Negative"]:
        if sentiment in daily_counts.columns:
            fig.add_trace(
                go.Bar(
                    x=daily_counts.index,
                    y=daily_counts[sentiment],
                    name=sentiment,
                    marker_color=SENTIMENT_COLORS[sentiment],
                    opacity=0.88,
                )
            )

    fig.add_trace(
        go.Scatter(
            x=daily_compound.index,
            y=daily_compound.values,
            name="Avg compound score",
            mode="lines+markers",
            yaxis="y2",
            line=dict(color="#2563EB", width=2.5),
            marker=dict(size=7, color="#2563EB"),
        )
    )

    fig.update_layout(
        title="Sentiment over time",
        barmode="stack",
        xaxis_title="Date",
        yaxis_title="Headline count",
        yaxis2=dict(
            title="Compound score",
            overlaying="y",
            side="right",
            range=[-1, 1],
            showgrid=False,
        ),
        hovermode="x unified",
    )
    return _apply_theme(fig, height=400)


def _build_compound_histogram(df: pd.DataFrame) -> go.Figure:
    fig = px.histogram(
        df,
        x="compound",
        nbins=20,
        color="Sentiment",
        color_discrete_map=SENTIMENT_COLORS,
        labels={"compound": "VADER compound score"},
        title="Score distribution by headline",
    )
    fig.add_vline(x=0, line_dash="dot", line_color="rgba(128,128,128,0.6)")
    return _apply_theme(fig, height=320)


def _style_dataframe(df: pd.DataFrame):
    """Color-code sentiment cells for readability."""

    def _sentiment_bg(val):
        v = str(val)
        if v == "Positive":
            return "background-color: #DCFCE7; color: #166534; font-weight: 600"
        if v == "Negative":
            return "background-color: #FEE2E2; color: #991B1B; font-weight: 600"
        if v == "Neutral":
            return "background-color: #FEF3C7; color: #92400E; font-weight: 600"
        return ""

    def _score_color(val):
        try:
            v = float(val)
        except (TypeError, ValueError):
            return ""
        if v > 0.05:
            return "color: #166534; font-weight: 600"
        if v < -0.05:
            return "color: #991B1B; font-weight: 600"
        return "color: #92400E"

    is_light = st.session_state.get("ui_theme_is_light", True)
    if not is_light:
        def _sentiment_bg_dark(val):
            v = str(val)
            if v == "Positive":
                return "background-color: #14532D; color: #86EFAC; font-weight: 600"
            if v == "Negative":
                return "background-color: #7F1D1D; color: #FCA5A5; font-weight: 600"
            if v == "Neutral":
                return "background-color: #78350F; color: #FCD34D; font-weight: 600"
            return ""

        def _score_color_dark(val):
            try:
                v = float(val)
            except (TypeError, ValueError):
                return ""
            if v > 0.05:
                return "color: #86EFAC; font-weight: 600"
            if v < -0.05:
                return "color: #FCA5A5; font-weight: 600"
            return "color: #FCD34D"

        return df.style.map(_sentiment_bg_dark, subset=["Sentiment"]).map(
            _score_color_dark, subset=["Score"]
        )

    return df.style.map(_sentiment_bg, subset=["Sentiment"]).map(
        _score_color, subset=["Score"]
    )


def render_news_sentiment_content(
    st_module,
    ticker: str,
    render_section_card_start,
    render_section_card_end,
):
    st = st_module

    with st.spinner(f"Fetching headlines for {ticker}..."):
        try:
            df = get_news_sentiment(ticker)
        except Exception as exc:
            st.error(f"Could not load news for {ticker}: {exc}")
            return

    if df is None or df.empty:
        st.info(f"No recent headlines found for **{ticker}**.")
        return

    summary = summarize_sentiment(df)
    overall = summary.get("overall", "Neutral")

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Overall", overall)
    m2.metric("Positive", summary.get("positive", 0))
    m3.metric("Neutral", summary.get("neutral", 0))
    m4.metric("Negative", summary.get("negative", 0))
    m5.metric("Avg score", f"{summary.get('avg_compound', 0):+.3f}")

    col_pie, col_hist = st.columns([1.1, 1])
    with col_pie:
        render_section_card_start(st, "Sentiment breakdown", "Share of positive, neutral, and negative headlines")
        st.plotly_chart(_build_donut_chart(df), use_container_width=True)
        render_section_card_end(st)

    with col_hist:
        render_section_card_start(st, "Score distribution", "How headline compound scores are spread")
        st.plotly_chart(_build_compound_histogram(df), use_container_width=True)
        render_section_card_end(st)

    render_section_card_start(st, "Sentiment timeline", "Daily headline volume stacked by mood + average compound")
    st.plotly_chart(_build_timeline_chart(df), use_container_width=True)
    render_section_card_end(st)

    render_section_card_start(st, "Headlines table", "Sortable view with VADER scores")
    display = _prepare_display_df(df)
    st.dataframe(
        _style_dataframe(display),
        use_container_width=True,
        hide_index=True,
        height=min(520, 44 + len(display) * 38),
        column_config={
            "Headline": st.column_config.TextColumn("Headline", width="large"),
            "Sentiment": st.column_config.TextColumn("Sentiment", width="small"),
            "Score": st.column_config.NumberColumn("Score", format="%.3f", width="small"),
            "Date": st.column_config.TextColumn("Date", width="small"),
            "Time": st.column_config.TextColumn("Time", width="small"),
        },
    )
    render_section_card_end(st)

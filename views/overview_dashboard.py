"""Open-source analytics dashboard (Plotly + pandas) — replaces Power BI embed."""

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
import yfinance as yf

from config.settings import get_settings

WATCHLIST = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "JPM", "V", "JNJ"]


def _sp500_path() -> Path:
    return get_settings().sp500_path


def _apply_theme(fig, height=380):
    is_light = st.session_state.get("ui_theme_is_light", True)
    template = "plotly_white" if is_light else "plotly_dark"
    grid = "rgba(30,41,59,0.08)" if is_light else "rgba(255,255,255,0.08)"
    text = "#1E293B" if is_light else "#E8EEF5"
    fig.update_layout(
        template=template,
        height=height,
        margin=dict(l=16, r=16, t=44, b=16),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=text, family="Inter, system-ui, sans-serif"),
        title_font=dict(size=16, color=text),
    )
    fig.update_xaxes(showgrid=True, gridcolor=grid)
    fig.update_yaxes(showgrid=True, gridcolor=grid)
    return fig


@st.cache_data(ttl=3600, show_spinner=False)
def _load_sp500() -> pd.DataFrame:
    df = pd.read_csv(_sp500_path())
    df["Sector"] = df["Sector"].fillna("Unknown")
    return df


@st.cache_data(ttl=900, show_spinner=False)
def _fetch_watchlist_returns() -> pd.DataFrame:
    """30-day % return for major watchlist tickers via Yahoo Finance."""
    try:
        data = yf.download(
            WATCHLIST,
            period="1mo",
            interval="1d",
            auto_adjust=True,
            progress=False,
            group_by="ticker",
        )
    except Exception:
        return pd.DataFrame()

    if data is None or data.empty:
        return pd.DataFrame()

    rows = []
    multi = isinstance(data.columns, pd.MultiIndex)
    for sym in WATCHLIST:
        try:
            if multi:
                if sym not in data.columns.get_level_values(0):
                    continue
                close = data[sym]["Close"].dropna()
            else:
                close = data["Close"].dropna()
            if len(close) < 2:
                continue
            ret = (float(close.iloc[-1]) / float(close.iloc[0]) - 1.0) * 100
            rows.append({"Symbol": sym, "Return_30d_pct": round(ret, 2)})
        except Exception:
            continue
    return pd.DataFrame(rows)


@st.cache_data(ttl=900, show_spinner=False)
def _fetch_index_snapshot() -> pd.DataFrame:
    """Recent close series for SPY, QQQ, DIA benchmarks."""
    benchmarks = {"SPY": "S&P 500", "QQQ": "Nasdaq 100", "DIA": "Dow 30"}
    data = yf.download(
        list(benchmarks.keys()),
        period="3mo",
        interval="1d",
        auto_adjust=True,
        progress=False,
        group_by="ticker",
    )
    frames = []
    for sym, label in benchmarks.items():
        try:
            if isinstance(data.columns, pd.MultiIndex):
                s = data[sym]["Close"].dropna()
            else:
                s = data["Close"].dropna()
            if s.empty:
                continue
            norm = (s / s.iloc[0] - 1.0) * 100
            part = pd.DataFrame({"Date": norm.index, "Return_pct": norm.values, "Index": label})
            frames.append(part)
        except Exception:
            continue
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def render_overview_dashboard(st_module):
    """Render the open-source market overview dashboard."""
    st = st_module

    st.markdown(
        """
        <div style="margin-bottom:0.75rem;">
          <span class="feature-pill">Plotly</span>
          <span class="feature-pill">Open source</span>
          <span class="feature-pill">Yahoo Finance</span>
          <span class="feature-pill">No login required</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    sp500 = _load_sp500()
    sector_counts = (
        sp500.groupby("Sector")
        .size()
        .reset_index(name="Companies")
        .sort_values("Companies", ascending=False)
    )

    tab_sectors, tab_returns, tab_indices, tab_table = st.tabs(
        ["Sector map", "Watchlist returns", "Index trends", "S&P 500 universe"]
    )

    with tab_sectors:
        fig_tree = px.treemap(
            sector_counts,
            path=["Sector"],
            values="Companies",
            title="S&P 500 composition by sector",
            color="Companies",
            color_continuous_scale="Tealgrn",
        )
        _apply_theme(fig_tree, height=420)
        st.plotly_chart(fig_tree, use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            fig_pie = px.pie(
                sector_counts.head(8),
                names="Sector",
                values="Companies",
                title="Top sectors (share)",
                hole=0.45,
            )
            _apply_theme(fig_pie, height=340)
            st.plotly_chart(fig_pie, use_container_width=True)
        with c2:
            st.markdown("##### Sector breakdown")
            st.dataframe(sector_counts, use_container_width=True, hide_index=True)

    with tab_returns:
        returns_df = _fetch_watchlist_returns()
        if returns_df.empty:
            st.warning("Could not load watchlist returns. Try again in a moment.")
        else:
            returns_df = returns_df.sort_values("Return_30d_pct", ascending=True)
            colors = ["#16A34A" if v >= 0 else "#DC2626" for v in returns_df["Return_30d_pct"]]
            fig_bar = go.Figure(
                go.Bar(
                    x=returns_df["Return_30d_pct"],
                    y=returns_df["Symbol"],
                    orientation="h",
                    marker_color=colors,
                    text=[f"{v:+.1f}%" for v in returns_df["Return_30d_pct"]],
                    textposition="outside",
                )
            )
            fig_bar.update_layout(title="30-day return — mega-cap watchlist")
            _apply_theme(fig_bar, height=400)
            st.plotly_chart(fig_bar, use_container_width=True)

    with tab_indices:
        idx_df = _fetch_index_snapshot()
        if idx_df.empty:
            st.warning("Benchmark data unavailable right now.")
        else:
            fig_line = px.line(
                idx_df,
                x="Date",
                y="Return_pct",
                color="Index",
                title="Benchmark performance (3 months, rebased to 0%)",
            )
            fig_line.add_hline(y=0, line_dash="dot", line_color="rgba(128,128,128,0.5)")
            _apply_theme(fig_line, height=400)
            st.plotly_chart(fig_line, use_container_width=True)

    with tab_table:
        sector_filter = st.multiselect(
            "Filter by sector",
            sorted(sp500["Sector"].unique()),
            default=[],
        )
        search = st.text_input("Search symbol or company", "")
        view = sp500.copy()
        if sector_filter:
            view = view[view["Sector"].isin(sector_filter)]
        if search.strip():
            q = search.strip().upper()
            view = view[
                view["Symbol"].str.contains(q, na=False)
                | view["Name"].str.upper().str.contains(q, na=False)
            ]
        st.caption(f"Showing {len(view)} of {len(sp500)} S&P 500 companies")
        st.dataframe(view, use_container_width=True, hide_index=True, height=420)

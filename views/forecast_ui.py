"""Stock price forecast page — themed Plotly charts, no external GIFs."""

from __future__ import annotations

from datetime import date
from typing import Any, Callable

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def _chart_palette(is_light: bool) -> dict[str, str]:
    if is_light:
        return {
            "close": "#2563EB",
            "open": "#94A3B8",
            "forecast": "#0D9488",
            "band": "rgba(124, 58, 237, 0.14)",
            "grid": "rgba(30,41,59,0.08)",
        }
    return {
        "close": "#60A5FA",
        "open": "#64748B",
        "forecast": "#5EEAD4",
        "band": "rgba(167, 139, 250, 0.22)",
        "grid": "rgba(255,255,255,0.08)",
    }


def _style_fig(fig: go.Figure, apply_theme: Callable, height: int = 440) -> go.Figure:
    fig = apply_theme(fig, height=height)
    is_light = fig.layout.template and "white" in str(fig.layout.template)
    if not is_light and fig.layout.template is None:
        is_light = True
    return fig


def _history_figure(data: pd.DataFrame, apply_theme: Callable, *, is_light: bool) -> go.Figure:
    colors = _chart_palette(is_light)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=data["Date"],
            y=data["Close"],
            name="Close",
            mode="lines",
            line=dict(color=colors["close"], width=2.5),
            fill="tozeroy",
            fillcolor="rgba(37, 99, 235, 0.12)" if is_light else "rgba(96, 165, 250, 0.15)",
        )
    )
    if "Open" in data.columns:
        fig.add_trace(
            go.Scatter(
                x=data["Date"],
                y=data["Open"],
                name="Open",
                mode="lines",
                line=dict(color=colors["open"], width=1.2, dash="dot"),
                opacity=0.85,
            )
        )
    fig.update_layout(
        title="Historical prices",
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        hovermode="x unified",
        xaxis_rangeslider=dict(visible=True, thickness=0.05),
    )
    return _style_fig(fig, apply_theme, height=460)


def _forecast_figure(
    df_train: pd.DataFrame,
    forecast: pd.DataFrame,
    *,
    ticker: str,
    method: str,
    apply_theme: Callable,
    is_light: bool,
    has_bands: bool = False,
) -> go.Figure:
    colors = _chart_palette(is_light)
    hist = df_train.dropna(subset=["ds", "y"])
    last_hist = hist["ds"].max()

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=hist["ds"],
            y=hist["y"],
            name="Historical close",
            mode="lines",
            line=dict(color=colors["close"], width=2.5),
        )
    )

    fc = forecast[forecast["ds"] > last_hist].copy() if len(forecast) else forecast
    if fc.empty:
        fc = forecast

    if has_bands and {"yhat_lower", "yhat_upper"}.issubset(forecast.columns):
        band_x = fc["ds"].tolist() + fc["ds"].tolist()[::-1]
        band_y = fc["yhat_upper"].tolist() + fc["yhat_lower"].tolist()[::-1]
        fig.add_trace(
            go.Scatter(
                x=band_x,
                y=band_y,
                fill="toself",
                fillcolor=colors["band"],
                line=dict(width=0),
                name="Confidence interval",
                hoverinfo="skip",
                showlegend=True,
            )
        )

    fig.add_trace(
        go.Scatter(
            x=fc["ds"],
            y=fc["yhat"] if "yhat" in fc.columns else fc.iloc[:, 1],
            name="Forecast",
            mode="lines",
            line=dict(color=colors["forecast"], width=2.8, dash="dash"),
        )
    )

    fig.add_vline(
        x=last_hist,
        line_width=1,
        line_dash="dot",
        line_color="rgba(148,163,184,0.9)",
    )
    fig.update_layout(
        title=f"{ticker} — {method} forecast",
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        hovermode="x unified",
    )
    return _style_fig(fig, apply_theme, height=480)


def _components_figure(
    df_train: pd.DataFrame,
    forecast: pd.DataFrame | None,
    apply_theme: Callable,
) -> go.Figure:
    """Monthly pattern + forecast trend (Plotly replacement for matplotlib components)."""
    hist = df_train.dropna(subset=["ds", "y"]).copy()
    monthly = hist.set_index("ds")["y"].resample("ME").mean()

    fig = make_subplots(
        rows=2,
        cols=1,
        subplot_titles=("Monthly average close", "Forecast trend (daily)"),
        vertical_spacing=0.12,
    )
    fig.add_trace(
        go.Bar(
            x=monthly.index,
            y=monthly.values,
            name="Monthly avg",
            marker_color="#0D9488",
            opacity=0.85,
        ),
        row=1,
        col=1,
    )

    if forecast is not None and not forecast.empty and "yhat" in forecast.columns:
        last_hist = hist["ds"].max()
        fc = forecast[forecast["ds"] > last_hist]
        fig.add_trace(
            go.Scatter(
                x=fc["ds"],
                y=fc["yhat"],
                name="Projected",
                line=dict(color="#0D9488", width=2),
            ),
            row=2,
            col=1,
        )
    else:
        t0 = hist["ds"].min()
        x = (hist["ds"] - t0).dt.days.values.astype(float)
        y = hist["y"].values.astype(float)
        coef = np.polyfit(x, y, 1)
        poly = np.poly1d(coef)
        fig.add_trace(
            go.Scatter(
                x=hist["ds"],
                y=poly(x),
                name="Linear trend",
                line=dict(color="#0D9488", width=2, dash="dot"),
            ),
            row=2,
            col=1,
        )

    fig.update_layout(height=520, showlegend=False)
    return _style_fig(fig, apply_theme, height=520)


def _linear_forecast_df(df_train: pd.DataFrame, period_days: int) -> pd.DataFrame:
    d = df_train.dropna(subset=["ds", "y"]).copy()
    t0 = d["ds"].min()
    x = (d["ds"] - t0).dt.days.values.astype(float)
    y = d["y"].values.astype(float)
    coef = np.polyfit(x, y, 1)
    poly = np.poly1d(coef)
    last = d["ds"].max()
    future = pd.date_range(last + pd.Timedelta(days=1), periods=int(period_days), freq="D")
    all_ds = pd.concat([d["ds"].reset_index(drop=True), pd.Series(future)], ignore_index=True)
    x_all = (pd.to_datetime(all_ds) - t0).dt.days.values.astype(float)
    return pd.DataFrame({"ds": pd.to_datetime(all_ds), "yhat": np.maximum(poly(x_all), 1e-6)})


def render_stock_forecast_page(
    st,
    *,
    page_title: Callable,
    render_page_hero: Callable,
    render_section_card_start: Callable,
    render_section_card_end: Callable,
    apply_plotly_theme: Callable,
    sp500_csv_path: str,
    safe_yf_download: Callable,
    prophet_available: bool,
    Prophet: Any,
    plot_plotly: Any,
) -> None:
    page_title(st, "Stock forecast", "Historical prices and forward projection")
    render_page_hero(
        st,
        icon="🔮",
        kicker="Price projection",
        title="S&P 500 stock forecast",
        subtitle=(
            "Select any S&P 500 ticker, review historical prices, and project forward using "
            "**Prophet** (when available) or a **linear trend** fallback."
        ),
        pills=["S&P 500 universe", "Prophet / trend", "Confidence bands", "2015 → today"],
    )

    snp500 = pd.read_csv(sp500_csv_path)
    symbols = snp500["Symbol"].sort_values().tolist()

    ticker = st.sidebar.selectbox("Choose a S&P 500 stock", symbols, key="forecast_ticker")
    n_years = st.sidebar.slider("Forecast horizon (years)", 1, 4, 2, key="forecast_years")
    period = n_years * 365
    start = "2015-01-01"
    today = date.today().strftime("%Y-%m-%d")

    with st.spinner(f"Loading {ticker} market data…"):
        data = safe_yf_download(ticker, start, today)
    if data.empty:
        st.error(f"No market data for {ticker}. Try another symbol.")
        st.stop()
    data = data.reset_index(drop=True)

    last_close = float(data["Close"].iloc[-1])
    df_train = data[["Date", "Close"]].copy()
    df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})
    df_train["ds"] = pd.to_datetime(df_train["ds"], errors="coerce")
    df_train["y"] = pd.to_numeric(df_train["y"], errors="coerce").ffill().bfill()

    method = "Linear trend"
    forecast_full: pd.DataFrame | None = None
    projected_end = None
    prophet_fig = None

    if prophet_available and plot_plotly is not None:
        try:
            m = Prophet()
            m.fit(df_train)
            future = m.make_future_dataframe(periods=period)
            forecast_full = m.predict(future)
            method = "Prophet"
            projected_end = float(forecast_full["yhat"].iloc[-1])
            prophet_fig = plot_plotly(m, forecast_full)
            if prophet_fig is not None:
                prophet_fig = apply_plotly_theme(prophet_fig, height=500)
                prophet_fig.update_layout(
                    title=f"{ticker} — Prophet forecast ({n_years}y horizon)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                )
        except Exception:
            forecast_full = None

    if forecast_full is None:
        forecast_full = _linear_forecast_df(df_train, period)
        projected_end = float(forecast_full["yhat"].iloc[-1])

    pct_change = ((projected_end - last_close) / last_close * 100) if last_close else 0.0
    is_light = st.session_state.get("ui_theme_is_light", True)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Last close", f"${last_close:,.2f}")
    m2.metric(f"Projected (+{n_years}y)", f"${projected_end:,.2f}")
    m3.metric("Implied change", f"{pct_change:+.1f}%")
    m4.metric("Model", method)

    render_section_card_start(st, "Price history", "Open & close with range slider")
    st.plotly_chart(_history_figure(data, apply_plotly_theme, is_light=is_light), use_container_width=True)
    render_section_card_end(st)

    render_section_card_start(st, "Forecast", f"{method} projection with confidence context")
    has_bands = method == "Prophet" and forecast_full is not None
    if prophet_fig is not None and method == "Prophet":
        st.plotly_chart(prophet_fig, use_container_width=True)
    else:
        st.plotly_chart(
            _forecast_figure(
                df_train,
                forecast_full,
                ticker=ticker,
                method=method,
                apply_theme=apply_plotly_theme,
                is_light=is_light,
                has_bands=has_bands,
            ),
            use_container_width=True,
        )
    render_section_card_end(st)

    render_section_card_start(st, "Trend components", "Monthly averages and projected path")
    st.plotly_chart(
        _components_figure(df_train, forecast_full, apply_plotly_theme),
        use_container_width=True,
    )
    render_section_card_end(st)

    with st.expander("Forecast table (last 30 rows)"):
        if forecast_full is not None:
            cols = [c for c in ["ds", "yhat", "yhat_lower", "yhat_upper"] if c in forecast_full.columns]
            st.dataframe(forecast_full[cols].tail(30), use_container_width=True)

    with st.expander("Raw OHLCV sample"):
        st.dataframe(data.tail(20), use_container_width=True)

    st.caption(
        "Forecasts are statistical estimates only — not financial advice. "
        "Past performance does not guarantee future results."
    )

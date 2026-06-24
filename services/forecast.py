"""Price and trend forecasting (Prophet with polynomial fallback)."""

from __future__ import annotations

import logging
from datetime import date, datetime, timedelta
from typing import Any, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_prophet_available: Optional[bool] = None


def _check_prophet() -> bool:
    global _prophet_available
    if _prophet_available is not None:
        return _prophet_available
    try:
        from prophet import Prophet

        Prophet()
        _prophet_available = True
    except Exception:
        _prophet_available = False
    return _prophet_available


def _poly_forecast(
    ds: pd.Series,
    y: pd.Series,
    periods: int,
    clip_min: Optional[float] = None,
    clip_max: Optional[float] = None,
) -> pd.DataFrame:
    d = pd.DataFrame({"ds": pd.to_datetime(ds), "y": pd.to_numeric(y, errors="coerce")})
    d = d.dropna()
    if len(d) < 3:
        raise ValueError("Not enough data points for forecast")

    last_date = d["ds"].max()
    future_dates = pd.date_range(
        start=last_date + pd.Timedelta(days=1),
        periods=int(periods),
        freq="D",
    )
    all_ds = pd.concat([d["ds"].reset_index(drop=True), pd.Series(future_dates)], ignore_index=True)

    t0 = d["ds"].min()
    x = (d["ds"] - t0).dt.days.values.astype(float)
    y_vals = d["y"].values.astype(float)
    deg = min(2, max(1, len(x) - 2))
    coef = np.polyfit(x, y_vals, deg=deg)
    poly = np.poly1d(coef)
    x_all = (pd.to_datetime(all_ds) - t0).dt.days.values.astype(float)
    yhat = poly(x_all)
    if clip_min is not None or clip_max is not None:
        yhat = np.clip(yhat, clip_min or -np.inf, clip_max or np.inf)
    else:
        yhat = np.maximum(yhat, 1e-6)

    return pd.DataFrame({"ds": pd.to_datetime(all_ds), "yhat": yhat})


def forecast_trend_series(
    df: pd.DataFrame,
    periods: int = 90,
) -> dict[str, Any]:
    """Forecast a ds/y time series (e.g. Finviz activity proxy)."""
    d = df.copy()
    d["ds"] = pd.to_datetime(d["ds"], errors="coerce")
    d["y"] = pd.to_numeric(d["y"], errors="coerce").ffill().bfill()
    d = d.dropna(subset=["ds"])

    method = "polynomial"
    if _check_prophet():
        try:
            from prophet import Prophet

            m = Prophet()
            m.fit(d[["ds", "y"]])
            future = m.make_future_dataframe(periods=periods)
            forecast = m.predict(future)
            method = "prophet"
            result = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(periods + 5)
            return {
                "method": method,
                "periods": periods,
                "forecast": result.to_dict(orient="records"),
                "last_actual": float(d["y"].iloc[-1]),
                "projected_end": float(forecast["yhat"].iloc[-1]),
            }
        except Exception as exc:
            logger.warning("Prophet failed: %s", exc)

    forecast = _poly_forecast(d["ds"], d["y"], periods, clip_min=0.0, clip_max=100.0)
    return {
        "method": method,
        "periods": periods,
        "forecast": forecast.tail(periods + 5).to_dict(orient="records"),
        "last_actual": float(d["y"].iloc[-1]),
        "projected_end": float(forecast["yhat"].iloc[-1]),
    }


def forecast_price(
    ticker: str,
    years: int = 1,
    start: date | datetime | str = "2015-01-01",
    end: date | datetime | str | None = None,
) -> dict[str, Any]:
    """Forecast stock close price for N years ahead."""
    from services.market import download_ohlcv, get_price_column

    if end is None:
        end = date.today()

    data = download_ohlcv(ticker, start=start, end=end)
    if data.empty:
        raise ValueError(f"No price data for {ticker}")

    price_col = get_price_column(data)
    if price_col is None:
        raise ValueError("No Close/Adj Close column in price data")

    df_train = data[["Date", price_col]].copy()
    df_train = df_train.rename(columns={"Date": "ds", price_col: "y"})
    df_train["ds"] = pd.to_datetime(df_train["ds"], errors="coerce")
    df_train["y"] = pd.to_numeric(df_train["y"], errors="coerce").ffill().bfill()
    df_train = df_train.dropna()

    period_days = int(years) * 365
    method = "polynomial"

    if _check_prophet():
        try:
            from prophet import Prophet

            m = Prophet()
            m.fit(df_train)
            future = m.make_future_dataframe(periods=period_days)
            forecast = m.predict(future)
            method = "prophet"
            tail = forecast.tail(min(period_days, 30))
            return {
                "ticker": ticker.upper(),
                "method": method,
                "years": years,
                "last_close": float(df_train["y"].iloc[-1]),
                "projected_end": float(forecast["yhat"].iloc[-1]),
                "forecast_sample": tail[["ds", "yhat", "yhat_lower", "yhat_upper"]].to_dict(orient="records"),
                "disclaimer": "Forecasts are statistical projections, not investment advice.",
            }
        except Exception as exc:
            logger.warning("Prophet price forecast failed: %s", exc)

    forecast = _poly_forecast(df_train["ds"], df_train["y"], period_days)
    tail = forecast.tail(min(period_days, 30))
    return {
        "ticker": ticker.upper(),
        "method": method,
        "years": years,
        "last_close": float(df_train["y"].iloc[-1]),
        "projected_end": float(forecast["yhat"].iloc[-1]),
        "forecast_sample": tail.to_dict(orient="records"),
        "disclaimer": "Forecasts are statistical projections, not investment advice.",
    }

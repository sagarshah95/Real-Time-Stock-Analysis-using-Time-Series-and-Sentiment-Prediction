"""Technical indicators: SMA, EMA, MACD, RSI, Bollinger Bands, trading signals."""

from __future__ import annotations

from typing import Any

import pandas as pd


def compute_sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=1).mean()


def compute_ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def compute_macd(
    series: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> dict[str, pd.Series]:
    ema_fast = compute_ema(series, fast)
    ema_slow = compute_ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = compute_ema(macd_line, signal)
    histogram = macd_line - signal_line
    return {
        "macd": macd_line,
        "signal": signal_line,
        "histogram": histogram,
    }


def _rsi_signal(rsi_val: float | None) -> str:
    from services.indicators import rsi_signal
    return rsi_signal(rsi_val)


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    from services.indicators import compute_rsi as _compute_rsi
    return _compute_rsi(series, period=period)


def compute_bollinger_bands(
    series: pd.Series,
    window: int = 20,
    num_std: float = 2.0,
) -> dict[str, pd.Series]:
    from services.indicators import compute_bollinger_bands as _compute_bollinger_bands
    return _compute_bollinger_bands(series, window=window, num_std=num_std)


def compute_signal(price_df: pd.DataFrame, price_col: str = "Close") -> dict[str, Any]:
    """Derive bullish/bearish/neutral signal from SMA crossover."""
    if price_df is None or price_df.empty or price_col not in price_df.columns:
        return {"signal": "Neutral", "sma_5": None, "sma_20": None, "macd_histogram": None}

    close = price_df[price_col].dropna()
    if len(close) < 5:
        return {"signal": "Neutral", "sma_5": None, "sma_20": None, "macd_histogram": None}

    sma_5 = float(close.tail(5).mean())
    sma_20 = float(close.tail(min(20, len(close))).mean())

    if sma_5 > sma_20:
        signal = "Bullish"
    elif sma_5 < sma_20:
        signal = "Bearish"
    else:
        signal = "Neutral"

    macd_data = compute_macd(close)
    hist_val = float(macd_data["histogram"].iloc[-1]) if len(macd_data["histogram"]) else None

    return {
        "signal": signal,
        "sma_5": round(sma_5, 2),
        "sma_20": round(sma_20, 2),
        "macd_histogram": round(hist_val, 4) if hist_val is not None else None,
    }


def get_technicals_summary(price_df: pd.DataFrame, price_col: str = "Close") -> dict[str, Any]:
    """Full technical snapshot for agent/MCP tools."""
    if price_df is None or price_df.empty or price_col not in price_df.columns:
        return {"error": "No price data available"}

    close = price_df[price_col].dropna()
    sig = compute_signal(price_df, price_col)
    macd = compute_macd(close)
    from services.indicators import compute_rsi, compute_bollinger_bands

    rsi_series = compute_rsi(close)
    bb = compute_bollinger_bands(close)

    rsi_val = float(rsi_series.iloc[-1]) if len(rsi_series) else None
    last_close = float(close.iloc[-1])
    bb_upper = float(bb["upper"].iloc[-1]) if bb["upper"].notna().any() else None
    bb_lower = float(bb["lower"].iloc[-1]) if bb["lower"].notna().any() else None
    bb_pct = None
    if bb_upper is not None and bb_lower is not None and bb_upper != bb_lower:
        bb_pct = round((last_close - bb_lower) / (bb_upper - bb_lower) * 100, 1)

    return {
        **sig,
        "last_close": round(last_close, 2),
        "sma_50": round(float(compute_sma(close, 50).iloc[-1]), 2) if len(close) >= 50 else None,
        "sma_200": round(float(compute_sma(close, 200).iloc[-1]), 2) if len(close) >= 200 else None,
        "macd_line": round(float(macd["macd"].iloc[-1]), 4),
        "macd_signal": round(float(macd["signal"].iloc[-1]), 4),
        "rsi_14": round(rsi_val, 2) if rsi_val is not None else None,
        "rsi_signal": _rsi_signal(rsi_val),
        "bb_upper": round(bb_upper, 2) if bb_upper is not None else None,
        "bb_middle": round(float(bb["middle"].iloc[-1]), 2) if bb["middle"].notna().any() else None,
        "bb_lower": round(bb_lower, 2) if bb_lower is not None else None,
        "bb_percent_b": bb_pct,
    }


__all__ = [
    "compute_sma",
    "compute_ema",
    "compute_macd",
    "compute_rsi",
    "compute_bollinger_bands",
    "compute_signal",
    "get_technicals_summary",
]

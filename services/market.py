"""Yahoo Finance market data access."""

from __future__ import annotations

import logging
from datetime import date, datetime
from typing import Any, Optional

import pandas as pd
import yfinance as yf
from yfinance.exceptions import YFRateLimitError

logger = logging.getLogger(__name__)

_OHLCV = {"Open", "High", "Low", "Close", "Adj Close", "Volume"}
_price_cache: dict[str, pd.DataFrame] = {}
_info_cache: dict[str, dict[str, Any]] = {}


def normalize_market_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    normalized = df.copy()
    if isinstance(normalized.columns, pd.MultiIndex):
        level0 = set(normalized.columns.get_level_values(0).unique())
        level1 = set(normalized.columns.get_level_values(1).unique())
        if _OHLCV.intersection(level1) and not _OHLCV.intersection(level0):
            normalized.columns = normalized.columns.get_level_values(1)
        elif _OHLCV.intersection(level0) and not _OHLCV.intersection(level1):
            normalized.columns = normalized.columns.get_level_values(0)
        else:
            if len(_OHLCV.intersection(level1)) >= len(_OHLCV.intersection(level0)):
                normalized.columns = normalized.columns.get_level_values(1)
            else:
                normalized.columns = normalized.columns.get_level_values(0)
    return normalized


def get_price_column(df: pd.DataFrame) -> Optional[str]:
    if "Adj Close" in df.columns:
        return "Adj Close"
    if "Close" in df.columns:
        return "Close"
    return None


def _format_date(d: date | datetime | str | None) -> Optional[str]:
    if d is None:
        return None
    if hasattr(d, "strftime"):
        return d.strftime("%Y-%m-%d")
    return str(d)


def _ohlcv_from_history(sym: str, start: Optional[str], end: Optional[str]) -> pd.DataFrame:
    hist = yf.Ticker(sym).history(
        start=start,
        end=end,
        interval="1d",
        auto_adjust=False,
        actions=False,
    )
    if hist is None or hist.empty:
        return pd.DataFrame()

    out = hist.reset_index()
    if "Date" not in out.columns:
        if "Datetime" in out.columns:
            out = out.rename(columns={"Datetime": "Date"})
        elif len(out.columns) > 0:
            first = out.columns[0]
            if str(first).lower() in ("date", "datetime", "index") or pd.api.types.is_datetime64_any_dtype(out[first]):
                out = out.rename(columns={first: "Date"})
    return normalize_market_df(out)


def download_ohlcv(
    ticker: str,
    start: date | datetime | str | None = None,
    end: date | datetime | str | None = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Download OHLCV history with session-level fallback cache."""
    sym = str(ticker).strip().upper()
    if not sym:
        return pd.DataFrame()

    start_s = _format_date(start)
    end_s = _format_date(end)

    try:
        data = _ohlcv_from_history(sym, start_s, end_s)
        if not data.empty:
            if use_cache:
                _price_cache[sym] = data.copy()
            return data

        data = yf.download(sym, start=start_s, end=end_s, auto_adjust=False, progress=False)
        data = normalize_market_df(data)
        if data.empty:
            cached = _price_cache.get(sym)
            if cached is not None and not cached.empty:
                logger.warning("Using cached prices for %s", sym)
                return cached.copy()
            return pd.DataFrame()

        if "Date" not in data.columns:
            data = data.reset_index()
            if data.columns[0] != "Date":
                data = data.rename(columns={data.columns[0]: "Date"})

        if use_cache:
            _price_cache[sym] = data.copy()
        return data

    except YFRateLimitError:
        cached = _price_cache.get(sym)
        if cached is not None and not cached.empty:
            logger.warning("Yahoo rate limit; using cache for %s", sym)
            return cached.copy()
        raise
    except Exception:
        cached = _price_cache.get(sym)
        if cached is not None and not cached.empty:
            logger.warning("Fetch failed; using cache for %s", sym)
            return cached.copy()
        raise


def get_company_info(ticker: str, use_cache: bool = True) -> dict[str, Any]:
    sym = str(ticker).strip().upper()
    if not sym:
        return {}
    try:
        info = yf.Ticker(sym).info or {}
        if info and use_cache:
            _info_cache[sym] = info
        return info
    except Exception:
        return _info_cache.get(sym, {})


def compute_quick_stats(info: dict[str, Any], price_df: pd.DataFrame) -> dict[str, str]:
    stats = {
        "price": "—",
        "change_pct": "—",
        "volume": "—",
        "signal": "Neutral",
    }

    try:
        price = info.get("regularMarketPrice") or info.get("currentPrice")
        prev = info.get("previousClose")
        if price is not None:
            stats["price"] = f"${float(price):,.2f}"
        if price is not None and prev not in (None, 0):
            pct = ((float(price) - float(prev)) / float(prev)) * 100
            stats["change_pct"] = f"{pct:+.2f}%"

        vol = info.get("volume")
        if vol is not None:
            if vol >= 1_000_000_000:
                stats["volume"] = f"{vol / 1_000_000_000:.2f}B"
            elif vol >= 1_000_000:
                stats["volume"] = f"{vol / 1_000_000:.2f}M"
            elif vol >= 1_000:
                stats["volume"] = f"{vol / 1_000:.1f}K"
            else:
                stats["volume"] = str(vol)

        if price_df is not None and not price_df.empty and "Close" in price_df.columns:
            recent = price_df["Close"].dropna().tail(20)
            if len(recent) >= 5:
                sma_5 = recent.tail(5).mean()
                sma_20 = recent.mean()
                if sma_5 > sma_20:
                    stats["signal"] = "Bullish"
                elif sma_5 < sma_20:
                    stats["signal"] = "Bearish"
    except Exception:
        pass

    return stats

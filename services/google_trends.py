"""Google Trends news search interest and related news headlines for stocks."""

from __future__ import annotations

import logging
import re
import urllib.parse
from typing import Any
from urllib.request import Request, urlopen
from xml.etree import ElementTree as ET

import pandas as pd

logger = logging.getLogger(__name__)

_TRENDS_ALIAS = {
    "amazon": "AMZN",
    "google": "GOOGL",
    "alphabet": "GOOGL",
    "apple": "AAPL",
    "microsoft": "MSFT",
    "meta": "META",
    "facebook": "META",
    "netflix": "NFLX",
    "tesla": "TSLA",
    "nvidia": "NVDA",
    "amd": "AMD",
    "intel": "INTC",
    "disney": "DIS",
    "walmart": "WMT",
    "nike": "NKE",
    "coca cola": "KO",
    "coca-cola": "KO",
    "jpmorgan": "JPM",
    "jp morgan": "JPM",
    "bank of america": "BAC",
    "goldman": "GS",
    "berkshire": "BRK-B",
    "berkshire hathaway": "BRK-B",
}

_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)


def keyword_to_trends_query(keyword: str) -> str:
    """Map ticker or company name to a Google Trends news search phrase."""
    s = str(keyword or "").strip() or "Amazon"
    low = s.lower()
    if low in _TRENDS_ALIAS:
        ticker = _TRENDS_ALIAS[low]
        if low != ticker.lower():
            return s if any(ch.islower() for ch in s) else s.title()
        return f"{ticker} stock"
    compact = re.sub(r"\s+", "", s).upper()
    if re.fullmatch(r"[A-Z.\-]{1,6}", compact):
        return f"{compact.replace('.', '-')} stock"
    return s


def _query_variants(keyword: str) -> list[str]:
    primary = keyword_to_trends_query(keyword)
    s = str(keyword or "").strip() or "Amazon"
    variants = [primary, s.title()]
    if primary.endswith(" stock"):
        variants.append(primary[:-6].strip())
    return list(dict.fromkeys(v for v in variants if v))


def _patch_pytrends_urllib3() -> None:
    """pytrends <-> urllib3 v2: method_whitelist renamed to allowed_methods."""
    try:
        import urllib3.util.retry as retry_mod

        if getattr(retry_mod.Retry.__init__, "_fast_patched", False):
            return

        _orig = retry_mod.Retry.__init__

        def _patched(self, *args, **kwargs):
            if "method_whitelist" in kwargs:
                kwargs["allowed_methods"] = kwargs.pop("method_whitelist")
            return _orig(self, *args, **kwargs)

        _patched._fast_patched = True  # type: ignore[attr-defined]
        retry_mod.Retry.__init__ = _patched  # type: ignore[method-assign]
    except Exception:
        pass


def _make_trend_req():
    _patch_pytrends_urllib3()
    from pytrends.request import TrendReq

    return TrendReq(
        hl="en-US",
        tz=360,
        retries=2,
        backoff_factor=0.5,
        timeout=(10, 30),
        requests_args={"headers": {"User-Agent": _UA}},
    )


def _dataframe_from_raw(raw: pd.DataFrame, query: str) -> pd.DataFrame:
    if raw is None or raw.empty:
        raise ValueError(f"No Google Trends data for '{query}'.")

    raw = raw.reset_index()
    if "isPartial" in raw.columns:
        raw = raw.drop(columns=["isPartial"])

    date_col = "date" if "date" in raw.columns else raw.columns[0]
    if query in raw.columns:
        value_col = query
    else:
        value_cols = [c for c in raw.columns if c != date_col]
        if not value_cols:
            raise ValueError(f"No value column in Google Trends response for '{query}'.")
        value_col = value_cols[0]

    out = pd.DataFrame(
        {
            "ds": pd.to_datetime(raw[date_col], errors="coerce"),
            "y": pd.to_numeric(raw[value_col], errors="coerce").fillna(0.0).clip(0, 100),
        }
    )
    out = out.dropna(subset=["ds"]).sort_values("ds").reset_index(drop=True)
    if len(out) < 5:
        raise ValueError(f"Not enough Google Trends history for '{query}'.")
    return out


def _fetch_pytrends_interest(
    query: str,
    *,
    timeframe: str = "today 3-m",
    geo: str = "US",
    gprop: str = "news",
) -> pd.DataFrame:
    pytrends = _make_trend_req()
    pytrends.build_payload(
        kw_list=[query],
        timeframe=timeframe,
        geo=geo,
        gprop=gprop,
    )
    raw = pytrends.interest_over_time()
    return _dataframe_from_raw(raw, query)


def fetch_google_trends_news_interest(
    keyword: str,
    timeframe: str = "today 3-m",
    geo: str = "US",
) -> tuple[pd.DataFrame, str, str]:
    """
    Google Trends interest over time for news (then web) searches.
    Returns (df, query_used, source_label).
    """
    errors: list[str] = []
    for query in _query_variants(keyword):
        for gprop, label in (("news", "Google Trends · news"), ("", "Google Trends · web")):
            try:
                df = _fetch_pytrends_interest(
                    query,
                    timeframe=timeframe,
                    geo=geo,
                    gprop=gprop,
                )
                logger.info("Google Trends OK: query=%r gprop=%r rows=%s", query, gprop or "web", len(df))
                return df, query, label
            except Exception as exc:
                errors.append(f"{query} ({label}): {exc}")
                logger.debug("Google Trends attempt failed: %s", errors[-1])

    detail = "; ".join(errors[:3])
    raise RuntimeError(
        f"Google Trends unavailable ({detail}). "
        "Wait a minute if rate-limited, or the app will use Finviz news activity as fallback."
    )


def fetch_google_news_articles(keyword: str, limit: int = 12) -> list[dict[str, Any]]:
    """Recent news headlines from Google News for the stock/search term."""
    query = keyword_to_trends_query(keyword)
    url = (
        "https://news.google.com/rss/search?"
        f"q={urllib.parse.quote_plus(query)}&hl=en-US&gl=US&ceid=US:en"
    )
    req = Request(url, headers={"User-Agent": _UA})
    with urlopen(req, timeout=20) as resp:
        root = ET.fromstring(resp.read())

    articles: list[dict[str, Any]] = []
    for item in root.findall(".//item")[:limit]:
        articles.append(
            {
                "headline": (item.findtext("title") or "").strip(),
                "url": (item.findtext("link") or "").strip(),
                "published": (item.findtext("pubDate") or "").strip(),
            }
        )
    return [a for a in articles if a["headline"]]

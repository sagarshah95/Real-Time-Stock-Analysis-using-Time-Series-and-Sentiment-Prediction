"""Finviz fundamentals scraping."""

from __future__ import annotations

import re
from typing import Any, Optional
from urllib.request import Request, urlopen

from bs4 import BeautifulSoup

_FINVIZ_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)


def _finviz_snapshot_pairs(soup: BeautifulSoup) -> list[tuple[str, str]]:
    pairs = []
    for table in soup.find_all("table", class_=lambda c: c and "snapshot-table2" in c):
        for tr in table.find_all("tr"):
            tds = tr.find_all("td")
            for i in range(0, len(tds) - 1, 2):
                label = tds[i].get_text(strip=True)
                if not label:
                    continue
                val_td = tds[i + 1]
                if label.lower() == "website":
                    a = val_td.find("a", href=True)
                    val = (a.get("href") or "").strip() if a else val_td.get_text(" ", strip=True)
                else:
                    val = val_td.get_text(" ", strip=True)
                pairs.append((label, val))
    return pairs


def _parse_compact_number(text: Optional[str]) -> Optional[float]:
    if text is None:
        return None
    s = str(text).strip().replace(",", "")
    if not s or s in ("-", "N/A", "—"):
        return None
    mult = 1.0
    for suf, m in (("T", 1e12), ("B", 1e9), ("M", 1e6), ("K", 1e3)):
        if s.endswith(suf):
            mult = m
            s = s[: -len(suf)].strip()
            break
    try:
        return float(s) * mult
    except ValueError:
        return None


def _parse_int_volume(text: Optional[str]) -> Optional[int]:
    n = _parse_compact_number(text)
    return int(round(n)) if n is not None else None


def _parse_percent_fraction(text: Optional[str]) -> Optional[float]:
    if not text or text in ("-", "N/A"):
        return None
    s = str(text).strip().replace(",", "")
    if s.endswith("%"):
        try:
            return float(s[:-1].strip()) / 100.0
        except ValueError:
            return None
    return None


def _eps_next_year_estimate(pairs: list[tuple[str, str]]) -> Optional[float]:
    for lab, val in pairs:
        if lab == "EPS next Y" and val and "%" not in val:
            try:
                return float(str(val).strip().replace(",", ""))
            except ValueError:
                continue
    return None


def _forward_dividend_parts(text: Optional[str]) -> tuple[Optional[float], Optional[float]]:
    if not text:
        return None, None
    m = re.search(r"([\d.]+)\s*\(([\d.]+)%\)", str(text))
    if not m:
        return None, _parse_percent_fraction(text)
    try:
        return float(m.group(1)), float(m.group(2)) / 100.0
    except ValueError:
        return None, None


def _build_company_info(ticker: str, soup: BeautifulSoup) -> dict[str, Any]:
    pairs = _finviz_snapshot_pairs(soup)
    if not pairs:
        raise ValueError("Finviz quote page contained no snapshot table data.")

    mp = dict(pairs)

    title_name = None
    if soup.title and soup.title.string and " - " in soup.title.string:
        title_name = soup.title.string.split(" - ", 1)[1].split(" Stock")[0].strip()

    og = soup.find("meta", attrs={"property": "og:description"})
    og_desc = (og.get("content") or "").strip() if og else ""

    price = _parse_compact_number(mp.get("Price"))
    prev_close = _parse_compact_number(mp.get("Prev Close"))
    vol = _parse_int_volume(mp.get("Volume"))
    avg_vol = _parse_compact_number(mp.get("Avg Volume"))
    mcap = _parse_compact_number(mp.get("Market Cap"))
    ev = _parse_compact_number(mp.get("Enterprise Value"))
    income = _parse_compact_number(mp.get("Income"))
    beta = _parse_compact_number(mp.get("Beta"))
    trailing_pe = _parse_compact_number(mp.get("P/E"))
    forward_pe = _parse_compact_number(mp.get("Forward P/E"))
    peg = _parse_compact_number(mp.get("PEG"))
    pb = _parse_compact_number(mp.get("P/B"))
    book_sh = _parse_compact_number(mp.get("Book/sh"))
    eps_fwd = _eps_next_year_estimate(pairs)
    div_rate, div_yield = _forward_dividend_parts(mp.get("Forward Dividend & Yield", ""))
    if div_yield is None:
        div_yield = _parse_percent_fraction(mp.get("Dividend %"))

    profit_margin = _parse_percent_fraction(mp.get("Profit Margin"))
    payout = _parse_percent_fraction(mp.get("Payout"))
    ev_sales = _parse_compact_number(mp.get("EV/Sales"))
    ev_ebitda = _parse_compact_number(mp.get("EV/EBITDA"))
    sh_out = _parse_compact_number(mp.get("Shs Outstand"))
    sh_float = _parse_compact_number(mp.get("Shs Float"))
    short_ratio = _parse_compact_number(mp.get("Short Ratio"))
    short_interest = _parse_compact_number(mp.get("Short Interest"))

    summary = og_desc or (
        f"{title_name or ticker} — snapshot fundamentals from Finviz "
        f"(sector: {mp.get('Sector', 'N/A')}, industry: {mp.get('Industry', 'N/A')})."
    )

    return {
        "longName": title_name or mp.get("Company", ticker),
        "shortName": title_name or ticker,
        "symbol": ticker,
        "regularMarketPrice": price,
        "currentPrice": price,
        "previousClose": prev_close,
        "currency": "USD",
        "volume": vol,
        "averageVolume": int(avg_vol) if avg_vol is not None else None,
        "marketCap": int(mcap) if mcap is not None else None,
        "enterpriseValue": int(ev) if ev is not None else None,
        "trailingPE": trailing_pe,
        "forwardPE": forward_pe,
        "pegRatio": peg,
        "priceToBook": pb,
        "forwardEps": eps_fwd,
        "beta": beta,
        "bookValue": book_sh,
        "dividendRate": div_rate,
        "dividendYield": div_yield,
        "profitMargins": profit_margin,
        "payoutRatio": payout,
        "enterpriseToRevenue": ev_sales,
        "enterpriseToEbitda": ev_ebitda,
        "netIncomeToCommon": int(income) if income is not None else None,
        "sector": mp.get("Sector", "N/A"),
        "industry": mp.get("Industry", "N/A"),
        "country": mp.get("Country", "N/A"),
        "exchange": mp.get("Exchange", "N/A"),
        "longBusinessSummary": summary,
        "sharesOutstanding": int(sh_out) if sh_out is not None else None,
        "floatShares": int(sh_float) if sh_float is not None else None,
        "sharesShort": int(short_interest) if short_interest is not None else None,
        "shortRatio": short_ratio,
        "fullTimeEmployees": _parse_int_volume(mp.get("Employees")),
    }


def get_finviz_company_info(ticker: str) -> dict[str, Any]:
    sym = str(ticker).strip().upper()
    if not sym:
        return {}
    url = f"https://finviz.com/quote.ashx?t={sym}"
    req = Request(url, headers={"User-Agent": _FINVIZ_UA})
    with urlopen(req, timeout=25) as resp:
        soup = BeautifulSoup(resp.read(), "html.parser")
    return _build_company_info(sym, soup)

"""Agent tool definitions and execution."""

from __future__ import annotations

import json
import logging
from datetime import date, timedelta
from typing import Any, Callable

from services.market import download_ohlcv, get_company_info, compute_quick_stats, get_price_column
from services.news import get_news_sentiment, summarize_sentiment
from services.fundamentals import get_finviz_company_info
from services.forecast import forecast_price
from services.technicals import get_technicals_summary
from rag.retriever import get_retriever

logger = logging.getLogger(__name__)


def _safe_json(obj: Any) -> str:
    return json.dumps(obj, default=str, indent=2)


# OpenAI function-calling schema
TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "get_stock_quote",
            "description": "Get current stock price, volume, and key market stats for a ticker.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {"type": "string", "description": "Stock ticker symbol, e.g. AAPL"},
                },
                "required": ["ticker"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_news_sentiment",
            "description": "Get recent news headlines with VADER sentiment scores from Finviz.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {"type": "string", "description": "Stock ticker symbol"},
                },
                "required": ["ticker"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_fundamentals",
            "description": "Get company fundamentals: P/E, market cap, sector, margins, etc.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {"type": "string", "description": "Stock ticker symbol"},
                },
                "required": ["ticker"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_technicals",
            "description": "Get technical analysis: SMA, MACD, bullish/bearish signal.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {"type": "string", "description": "Stock ticker symbol"},
                    "days": {"type": "integer", "description": "Days of history to analyze", "default": 365},
                },
                "required": ["ticker"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "forecast_stock_price",
            "description": "Generate a statistical price forecast (Prophet or polynomial trend).",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {"type": "string", "description": "Stock ticker symbol"},
                    "years": {"type": "integer", "description": "Years to forecast ahead", "default": 1},
                },
                "required": ["ticker"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_earnings_transcripts",
            "description": "RAG search over earnings call transcripts for management commentary and financial details.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Natural language search query"},
                    "ticker": {"type": "string", "description": "Optional ticker filter"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_ticker",
            "description": "Run a full multi-source analysis combining quote, sentiment, technicals, and RAG context.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {"type": "string", "description": "Stock ticker symbol"},
                },
                "required": ["ticker"],
            },
        },
    },
]


def tool_get_stock_quote(ticker: str) -> str:
    sym = ticker.upper()
    info = get_company_info(sym)
    end = date.today()
    start = end - timedelta(days=30)
    prices = download_ohlcv(sym, start=start, end=end)
    stats = compute_quick_stats(info, prices)
    return _safe_json({
        "ticker": sym,
        "name": info.get("longName", sym),
        "sector": info.get("sector"),
        "industry": info.get("industry"),
        "stats": stats,
        "market_cap": info.get("marketCap"),
        "pe_ratio": info.get("trailingPE"),
    })


def tool_get_news_sentiment(ticker: str) -> str:
    sym = ticker.upper()
    news_df = get_news_sentiment(sym)
    summary = summarize_sentiment(news_df)
    return _safe_json({"ticker": sym, **summary})


def tool_get_fundamentals(ticker: str) -> str:
    sym = ticker.upper()
    try:
        info = get_finviz_company_info(sym)
    except Exception:
        info = get_company_info(sym)
    key_fields = {
        k: info.get(k)
        for k in (
            "longName", "symbol", "sector", "industry", "regularMarketPrice",
            "marketCap", "trailingPE", "forwardPE", "pegRatio", "priceToBook",
            "dividendYield", "profitMargins", "beta", "fullTimeEmployees",
        )
        if info.get(k) is not None
    }
    return _safe_json(key_fields)


def tool_get_technicals(ticker: str, days: int = 365) -> str:
    sym = ticker.upper()
    end = date.today()
    start = end - timedelta(days=days)
    prices = download_ohlcv(sym, start=start, end=end)
    col = get_price_column(prices) or "Close"
    tech = get_technicals_summary(prices, col)
    return _safe_json({"ticker": sym, **tech})


def tool_forecast_stock_price(ticker: str, years: int = 1) -> str:
    result = forecast_price(ticker, years=years)
    return _safe_json(result)


def tool_search_earnings_transcripts(query: str, ticker: str = "") -> str:
    retriever = get_retriever()
    t = ticker.upper() if ticker else None
    result = retriever.query_with_context(query, ticker=t)
    return _safe_json(result)


def tool_analyze_ticker(ticker: str) -> str:
    sym = ticker.upper()
    quote = json.loads(tool_get_stock_quote(sym))
    sentiment = json.loads(tool_get_news_sentiment(sym))
    technicals = json.loads(tool_get_technicals(sym))
    rag = json.loads(tool_search_earnings_transcripts(
        f"key financial highlights risks outlook for {sym}",
        sym,
    ))
    return _safe_json({
        "ticker": sym,
        "quote": quote,
        "sentiment": sentiment,
        "technicals": technicals,
        "earnings_context": rag,
        "disclaimer": "This is automated analysis, not financial advice.",
    })


TOOL_HANDLERS: dict[str, Callable[..., str]] = {
    "get_stock_quote": tool_get_stock_quote,
    "get_news_sentiment": tool_get_news_sentiment,
    "get_fundamentals": tool_get_fundamentals,
    "get_technicals": tool_get_technicals,
    "forecast_stock_price": tool_forecast_stock_price,
    "search_earnings_transcripts": tool_search_earnings_transcripts,
    "analyze_ticker": tool_analyze_ticker,
}


def execute_tool(name: str, arguments: dict[str, Any]) -> str:
    handler = TOOL_HANDLERS.get(name)
    if handler is None:
        return _safe_json({"error": f"Unknown tool: {name}"})
    try:
        return handler(**arguments)
    except Exception as exc:
        logger.exception("Tool %s failed", name)
        return _safe_json({"error": str(exc), "tool": name})

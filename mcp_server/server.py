"""MCP server exposing F.A.S.T stock analysis tools.

Run with:
    python -m mcp_server.server

Or configure in Cursor/Claude Desktop MCP settings.
"""

from __future__ import annotations

import json
import logging

from mcp.server.fastmcp import FastMCP

from agents.tools import (
    tool_analyze_ticker,
    tool_forecast_stock_price,
    tool_get_fundamentals,
    tool_get_news_sentiment,
    tool_get_stock_quote,
    tool_get_technicals,
    tool_search_earnings_transcripts,
)
from rag.ingest import ingest_all, ingest_transcripts
from rag.retriever import rag_health

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

mcp = FastMCP(
    "FAST Stock Analysis",
    instructions=(
        "Financial Analysis and Stock Trading (F.A.S.T) MCP server. "
        "Provides live market data, news sentiment, technicals, forecasts, "
        "and RAG search over earnings transcripts."
    ),
)


@mcp.tool()
def get_stock_quote(ticker: str) -> str:
    """Get current stock price, volume, and key market stats."""
    return tool_get_stock_quote(ticker)


@mcp.tool()
def get_news_sentiment(ticker: str) -> str:
    """Get recent news headlines with VADER sentiment scores."""
    return tool_get_news_sentiment(ticker)


@mcp.tool()
def get_fundamentals(ticker: str) -> str:
    """Get company fundamentals: P/E, market cap, sector, margins."""
    return tool_get_fundamentals(ticker)


@mcp.tool()
def get_technicals(ticker: str, days: int = 365) -> str:
    """Get technical analysis: SMA, MACD, bullish/bearish signal."""
    return tool_get_technicals(ticker, days=days)


@mcp.tool()
def forecast_stock_price(ticker: str, years: int = 1) -> str:
    """Generate a statistical price forecast."""
    return tool_forecast_stock_price(ticker, years=years)


@mcp.tool()
def search_earnings_transcripts(query: str, ticker: str = "") -> str:
    """RAG search over earnings call transcripts."""
    return tool_search_earnings_transcripts(query, ticker)


@mcp.tool()
def analyze_ticker(ticker: str) -> str:
    """Full multi-source analysis: quote + sentiment + technicals + RAG."""
    return tool_analyze_ticker(ticker)


@mcp.tool()
def rag_health_check() -> str:
    """Check RAG vector store status and document count."""
    return json.dumps(rag_health(), indent=2)


@mcp.tool()
def ingest_rag_corpus() -> str:
    """Index all earnings transcripts into the RAG vector store."""
    result = ingest_transcripts()
    return json.dumps(result, indent=2)


def main():
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()

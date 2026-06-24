"""Agent / LLM operations — run inside Streamlit (direct mode) or via FastAPI."""

from __future__ import annotations

from typing import Any, Optional


def run_chat(
    message: str,
    ticker: Optional[str] = None,
    history: Optional[list[dict[str, str]]] = None,
) -> dict[str, Any]:
    from agents.orchestrator import get_agent

    agent = get_agent()
    return agent.chat(message, ticker=ticker or None, history=history or [])


def run_analysis(ticker: str) -> dict[str, Any]:
    from agents.orchestrator import get_agent

    return get_agent().analyze(ticker.upper())

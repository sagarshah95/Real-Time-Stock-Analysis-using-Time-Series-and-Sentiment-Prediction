"""Route AI/RAG calls to direct in-process logic or a remote FastAPI URL."""

from __future__ import annotations

import os
from typing import Any, Optional
from urllib.parse import urlparse

import requests

from config.settings import get_settings


def _is_localhost_url(url: str) -> bool:
    host = (urlparse(url).hostname or "").lower()
    return host in {"localhost", "127.0.0.1", "0.0.0.0", ""}


def _is_streamlit_cloud() -> bool:
    return os.environ.get("STREAMLIT_RUNTIME_ENV", "").lower() == "cloud"


def should_use_remote_api() -> bool:
    """True only when USE_API is enabled and a non-empty API_URL is configured."""
    settings = get_settings()
    if not settings.use_api:
        return False
    url = settings.resolved_api_url
    return bool(url)


def runtime_mode_label() -> str:
    if should_use_remote_api():
        return f"Remote API ({get_settings().resolved_api_url})"
    return "Direct (in-process)"


def _api_request(method: str, path: str, **kwargs) -> dict[str, Any]:
    settings = get_settings()
    base = settings.resolved_api_url.rstrip("/")
    if not base:
        return {"error": "API_URL is not set. Set USE_API=true and API_URL in .env or Streamlit secrets."}
    if _is_localhost_url(base) and _is_streamlit_cloud():
        return {
            "error": (
                "API_URL points to localhost, which does not work on Streamlit Cloud. "
                "Set USE_API=false for direct mode, or deploy FastAPI and set a public API_URL."
            )
        }
    try:
        resp = requests.request(method, f"{base}{path}", timeout=120, **kwargs)
        resp.raise_for_status()
        return resp.json()
    except requests.ConnectionError:
        return {"error": f"Cannot reach API at {base}. Check API_URL and that the server is running."}
    except Exception as exc:
        return {"error": str(exc)}


def chat_message(
    message: str,
    ticker: Optional[str] = None,
    history: Optional[list[dict[str, str]]] = None,
) -> dict[str, Any]:
    if should_use_remote_api():
        return _api_request(
            "POST",
            "/chat",
            json={"message": message, "ticker": ticker, "history": history or []},
        )
    from backend.agent_pipeline import run_chat

    return run_chat(message, ticker=ticker, history=history)


def analyze_ticker(ticker: str) -> dict[str, Any]:
    if should_use_remote_api():
        return _api_request("POST", "/analyze", json={"ticker": ticker})
    from backend.agent_pipeline import run_analysis

    return run_analysis(ticker)


def rag_search(
    query: str,
    ticker: Optional[str] = None,
    top_k: Optional[int] = None,
) -> dict[str, Any]:
    if should_use_remote_api():
        return _api_request(
            "POST",
            "/rag/query",
            json={"query": query, "ticker": ticker, "top_k": top_k},
        )
    from backend.rag_pipeline import run_query

    return run_query(query, ticker=ticker, top_k=top_k)


def ingest_rag_corpus() -> dict[str, Any]:
    if should_use_remote_api():
        result = _api_request("POST", "/rag/ingest")
        if "error" in result:
            return result
        return result.get("details", result)
    from backend.rag_pipeline import ingest_corpus

    return ingest_corpus()


def rag_health_status() -> dict[str, Any]:
    if should_use_remote_api():
        health = _api_request("GET", "/health")
        if "error" in health:
            return health
        return health.get("rag", {})
    from backend.rag_pipeline import corpus_health

    return corpus_health()

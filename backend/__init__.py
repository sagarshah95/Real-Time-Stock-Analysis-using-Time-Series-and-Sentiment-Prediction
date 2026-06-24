"""In-process backend logic for Streamlit (direct mode) and shared API handlers."""

from backend.client import (
    analyze_ticker,
    chat_message,
    rag_health_status,
    ingest_rag_corpus,
    rag_search,
    runtime_mode_label,
    should_use_remote_api,
)

__all__ = [
    "analyze_ticker",
    "chat_message",
    "ingest_rag_corpus",
    "rag_health_status",
    "rag_search",
    "runtime_mode_label",
    "should_use_remote_api",
]

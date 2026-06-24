"""RAG operations — run inside Streamlit (direct mode) or via FastAPI."""

from __future__ import annotations

from typing import Any, Optional


def run_query(
    query: str,
    ticker: Optional[str] = None,
    top_k: Optional[int] = None,
) -> dict[str, Any]:
    from rag.retriever import get_retriever

    retriever = get_retriever()
    result = retriever.query_with_context(query, ticker or None)
    result["chunks"] = retriever.retrieve(query, ticker or None, top_k=top_k)
    return result


def ingest_corpus() -> dict[str, Any]:
    from rag.ingest import ingest_transcripts

    return ingest_transcripts()


def corpus_health() -> dict[str, Any]:
    from rag.retriever import rag_health

    return rag_health()

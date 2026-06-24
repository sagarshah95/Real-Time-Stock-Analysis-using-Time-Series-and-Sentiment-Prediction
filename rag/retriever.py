"""Semantic retrieval over financial documents."""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any, Optional

from config.settings import get_settings
from rag.store import collection_stats, get_collection

logger = logging.getLogger(__name__)


class RAGRetriever:
    def __init__(self, top_k: Optional[int] = None):
        settings = get_settings()
        self.top_k = top_k or settings.rag_top_k
        self._collection = get_collection()

    def retrieve(
        self,
        query: str,
        ticker: Optional[str] = None,
        doc_type: Optional[str] = None,
        top_k: Optional[int] = None,
    ) -> list[dict[str, Any]]:
        k = top_k or self.top_k
        where: dict[str, Any] = {}
        if ticker:
            where["ticker"] = ticker.upper()
        if doc_type:
            where["doc_type"] = doc_type

        try:
            if where:
                results = self._collection.query(
                    query_texts=[query],
                    n_results=k,
                    where=where,
                )
            else:
                results = self._collection.query(
                    query_texts=[query],
                    n_results=k,
                )
        except Exception as exc:
            logger.error("RAG query failed: %s", exc)
            return []

        return self._format_results(results)

    def _format_results(self, results: dict) -> list[dict[str, Any]]:
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        formatted = []
        for i, doc in enumerate(docs):
            meta = metas[i] if i < len(metas) else {}
            dist = distances[i] if i < len(distances) else None
            formatted.append({
                "content": doc,
                "ticker": meta.get("ticker", ""),
                "source": meta.get("source", ""),
                "source_file": meta.get("source_file", ""),
                "doc_type": meta.get("doc_type", ""),
                "headline": meta.get("headline", ""),
                "sentiment": meta.get("sentiment", ""),
                "relevance_score": round(1.0 - dist, 4) if dist is not None else None,
            })
        return formatted

    def query_with_context(
        self,
        query: str,
        ticker: Optional[str] = None,
    ) -> dict[str, Any]:
        chunks = self.retrieve(query, ticker=ticker)
        if not chunks:
            return {
                "query": query,
                "ticker": ticker,
                "context": "",
                "sources": [],
                "chunk_count": 0,
            }

        context_parts = []
        sources = []
        for c in chunks:
            context_parts.append(c["content"])
            sources.append({
                "ticker": c["ticker"],
                "source": c["source"],
                "source_file": c.get("source_file", ""),
                "relevance_score": c.get("relevance_score"),
            })

        return {
            "query": query,
            "ticker": ticker,
            "context": "\n\n---\n\n".join(context_parts),
            "sources": sources,
            "chunk_count": len(chunks),
        }


@lru_cache
def get_retriever() -> RAGRetriever:
    return RAGRetriever()


def rag_health() -> dict[str, Any]:
    return collection_stats()

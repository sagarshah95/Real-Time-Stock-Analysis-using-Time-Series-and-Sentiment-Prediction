"""Document ingestion for RAG knowledge base."""

from __future__ import annotations

import hashlib
import logging
import re
from pathlib import Path
from typing import Any, Optional

from config.settings import get_settings
from rag.store import get_collection

logger = logging.getLogger(__name__)


def _chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = end - overlap
    return chunks


def _doc_id(source: str, chunk_idx: int) -> str:
    raw = f"{source}::{chunk_idx}"
    return hashlib.sha256(raw.encode()).hexdigest()[:24]


def ingest_transcripts(
    data_dir: Optional[Path] = None,
    tickers: Optional[list[str]] = None,
) -> dict[str, Any]:
    """Index earnings call transcripts from data/transcripts/ (configurable via INFERENCE_DATA_DIR)."""
    settings = get_settings()
    root = data_dir or settings.inference_data_path
    if not root.exists():
        raise FileNotFoundError(f"Inference data directory not found: {root}")

    col = get_collection()
    chunk_size = settings.rag_chunk_size
    overlap = settings.rag_chunk_overlap

    files = [f for f in root.iterdir() if f.is_file()]
    if tickers:
        tickers_upper = {t.upper() for t in tickers}
        files = [f for f in files if f.name.upper() in tickers_upper]

    total_chunks = 0
    indexed_files = 0

    for fp in files:
        ticker = fp.name.upper()
        try:
            text = fp.read_text(encoding="utf-8", errors="replace")
        except Exception as exc:
            logger.warning("Could not read %s: %s", fp, exc)
            continue

        chunks = _chunk_text(text, chunk_size, overlap)
        if not chunks:
            continue

        ids = []
        documents = []
        metadatas = []

        for i, chunk in enumerate(chunks):
            ids.append(_doc_id(f"transcript:{ticker}", i))
            documents.append(chunk)
            metadatas.append({
                "ticker": ticker,
                "source": "earnings_transcript",
                "source_file": str(fp.name),
                "chunk_index": i,
                "doc_type": "transcript",
            })

        col.upsert(ids=ids, documents=documents, metadatas=metadatas)
        total_chunks += len(chunks)
        indexed_files += 1
        logger.info("Indexed %s: %d chunks", ticker, len(chunks))

    return {
        "indexed_files": indexed_files,
        "total_chunks": total_chunks,
        "data_dir": str(root),
    }


def ingest_news_headlines(ticker: str) -> dict[str, Any]:
    """Index recent news headlines for a ticker into RAG."""
    from services.news import get_news_sentiment

    sym = ticker.upper()
    news_df = get_news_sentiment(sym)
    if news_df.empty:
        return {"ticker": sym, "indexed": 0}

    col = get_collection()
    ids = []
    documents = []
    metadatas = []

    for i, row in news_df.iterrows():
        headline = str(row.get("headline", ""))
        if not headline or headline == "No headline available":
            continue
        doc_text = f"[{sym}] {headline} (Sentiment: {row.get('Sentiment', 'Neutral')})"
        ids.append(_doc_id(f"news:{sym}:{i}", 0))
        documents.append(doc_text)
        metadatas.append({
            "ticker": sym,
            "source": "finviz_news",
            "headline": headline,
            "sentiment": str(row.get("Sentiment", "Neutral")),
            "date": str(row.get("date", "")),
            "doc_type": "news",
            "chunk_index": 0,
        })

    if documents:
        col.upsert(ids=ids, documents=documents, metadatas=metadatas)

    return {"ticker": sym, "indexed": len(documents)}


def ingest_all(include_news_tickers: Optional[list[str]] = None) -> dict[str, Any]:
    """Full ingestion: all transcripts + optional news for tickers."""
    transcript_result = ingest_transcripts()
    news_results = []
    for t in include_news_tickers or []:
        try:
            news_results.append(ingest_news_headlines(t))
        except Exception as exc:
            news_results.append({"ticker": t, "error": str(exc)})

    return {
        "transcripts": transcript_result,
        "news": news_results,
    }

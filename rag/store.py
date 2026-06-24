"""ChromaDB vector store with lightweight ONNX embeddings (no torch/transformers)."""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any

from config.bootstrap import apply_runtime_compat

apply_runtime_compat()

import chromadb
from chromadb.config import Settings as ChromaSettings

from config.settings import get_settings

logger = logging.getLogger(__name__)

COLLECTION_NAME = "fast_financial_docs"
_embedding_fn = None


def _get_embedding_function():
    """Use Chroma's bundled ONNX MiniLM — avoids sentence-transformers/torchvision."""
    global _embedding_fn
    if _embedding_fn is not None:
        return _embedding_fn

    settings = get_settings()
    if settings.hf_token_configured:
        logger.info("Hugging Face Hub token configured for embedding model access")

    try:
        from chromadb.utils.embedding_functions.onnx_mini_lm_l6_v2 import ONNXMiniLM_L6_V2

        _embedding_fn = ONNXMiniLM_L6_V2()
        logger.info("Loaded ONNX MiniLM-L6-v2 embeddings")
    except Exception as exc:
        logger.warning("ONNX embeddings unavailable (%s); using Chroma default", exc)
        try:
            from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

            _embedding_fn = DefaultEmbeddingFunction()
        except Exception:
            _embedding_fn = None
    return _embedding_fn


@lru_cache
def get_chroma_client() -> chromadb.PersistentClient:
    settings = get_settings()
    return chromadb.PersistentClient(
        path=str(settings.chroma_path),
        settings=ChromaSettings(anonymized_telemetry=False),
    )


def get_collection():
    client = get_chroma_client()
    ef = _get_embedding_function()
    kwargs: dict[str, Any] = {"name": COLLECTION_NAME}
    if ef is not None:
        kwargs["embedding_function"] = ef
    try:
        return client.get_or_create_collection(**kwargs)
    except Exception:
        return client.get_or_create_collection(name=COLLECTION_NAME)


def collection_stats() -> dict[str, Any]:
    try:
        col = get_collection()
        return {"collection": COLLECTION_NAME, "document_count": col.count()}
    except Exception as exc:
        return {"collection": COLLECTION_NAME, "document_count": 0, "error": str(exc)}

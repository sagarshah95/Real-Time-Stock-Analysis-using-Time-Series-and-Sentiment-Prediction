"""FastAPI backend for agentic AI, RAG, and tool endpoints.

Run with:
    uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from api.models import (
    AnalyzeRequest,
    AnalyzeResponse,
    ChatRequest,
    ChatResponse,
    HealthResponse,
    IngestResponse,
    RAGQueryRequest,
    RAGQueryResponse,
)
from agents.orchestrator import get_agent
from config.settings import get_settings
from rag.ingest import ingest_all
from rag.retriever import get_retriever, rag_health

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    logger.info("F.A.S.T API starting (LLM configured: %s)", settings.llm_configured)

    try:
        stats = rag_health()
        if stats.get("document_count", 0) == 0:
            logger.info("RAG corpus empty — auto-indexing transcripts...")
            result = ingest_all()
            logger.info("RAG indexed: %s", result.get("transcripts", {}))
    except Exception as exc:
        logger.warning("RAG auto-ingest skipped: %s", exc)

    yield
    logger.info("F.A.S.T API shutting down")


app = FastAPI(
    title="F.A.S.T Stock Analysis API",
    description="Agentic AI, RAG, and market data tools for financial analysis.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
async def health():
    settings = get_settings()
    return HealthResponse(
        status="ok",
        llm_configured=settings.llm_configured,
        llm_provider=settings.active_provider,
        llm_model=settings.resolved_model,
        rag=rag_health(),
    )


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        agent = get_agent()
        result = agent.chat(
            message=request.message,
            ticker=request.ticker,
            history=request.history,
        )
        return ChatResponse(**result)
    except Exception as exc:
        logger.exception("Chat failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(request: AnalyzeRequest):
    try:
        agent = get_agent()
        result = agent.analyze(request.ticker.upper())
        return AnalyzeResponse(
            ticker=result.get("ticker", request.ticker.upper()),
            mode=result.get("mode", "unknown"),
            narrative=result.get("narrative", ""),
            quote=result.get("quote", {}),
            sentiment=result.get("sentiment", {}),
            technicals=result.get("technicals", {}),
            earnings_context=result.get("earnings_context", {}),
        )
    except Exception as exc:
        logger.exception("Analyze failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/rag/query", response_model=RAGQueryResponse)
async def rag_query(request: RAGQueryRequest):
    try:
        retriever = get_retriever()
        result = retriever.query_with_context(
            request.query,
            ticker=request.ticker.upper() if request.ticker else None,
        )
        chunks = retriever.retrieve(
            request.query,
            ticker=request.ticker.upper() if request.ticker else None,
            top_k=request.top_k,
        )
        return RAGQueryResponse(
            query=request.query,
            ticker=request.ticker,
            chunks=chunks,
            context=result.get("context", ""),
            sources=result.get("sources", []),
        )
    except Exception as exc:
        logger.exception("RAG query failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/rag/ingest", response_model=IngestResponse)
async def rag_ingest():
    try:
        result = ingest_all()
        return IngestResponse(status="ok", details=result)
    except Exception as exc:
        logger.exception("RAG ingest failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/tools")
async def list_tools():
    from agents.tools import TOOL_DEFINITIONS

    return {"tools": [t["function"]["name"] for t in TOOL_DEFINITIONS]}

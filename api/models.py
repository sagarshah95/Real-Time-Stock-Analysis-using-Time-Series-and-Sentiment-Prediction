"""Pydantic models for the REST API."""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=4000)
    ticker: Optional[str] = Field(default=None, max_length=10)
    history: Optional[list[dict[str, str]]] = None


class ChatResponse(BaseModel):
    response: str
    mode: str
    tools_used: list[str] = []
    ticker: Optional[str] = None
    model: Optional[str] = None
    note: Optional[str] = None


class AnalyzeRequest(BaseModel):
    ticker: str = Field(..., min_length=1, max_length=10)


class AnalyzeResponse(BaseModel):
    ticker: str
    mode: str
    narrative: str
    quote: dict[str, Any] = {}
    sentiment: dict[str, Any] = {}
    technicals: dict[str, Any] = {}
    earnings_context: dict[str, Any] = {}
    disclaimer: str = "Not financial advice."


class RAGQueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    ticker: Optional[str] = None
    top_k: Optional[int] = Field(default=None, ge=1, le=20)


class RAGQueryResponse(BaseModel):
    query: str
    ticker: Optional[str]
    chunks: list[dict[str, Any]]
    context: str
    sources: list[dict[str, Any]]


class IngestResponse(BaseModel):
    status: str
    details: dict[str, Any]


class HealthResponse(BaseModel):
    status: str
    llm_configured: bool
    llm_provider: str = ""
    llm_model: str = ""
    rag: dict[str, Any]
    version: str = "1.0.0"

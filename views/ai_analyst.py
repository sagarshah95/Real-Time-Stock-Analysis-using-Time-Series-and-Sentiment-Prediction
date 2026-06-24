"""AI Analyst UI module (not a Streamlit multipage file — see views/ not pages/)."""

from __future__ import annotations

import sys
import traceback
from pathlib import Path
from typing import Any

# Ensure project root is on sys.path when Streamlit loads this module
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import requests
import streamlit as st

from config.settings import get_settings


def _api_request(method: str, path: str, **kwargs) -> dict[str, Any]:
    settings = get_settings()
    url = f"{settings.api_url.rstrip('/')}{path}"
    try:
        resp = requests.request(method, url, timeout=120, **kwargs)
        resp.raise_for_status()
        return resp.json()
    except requests.ConnectionError:
        return {"error": "API unavailable. Start the backend: uvicorn api.main:app --port 8000"}
    except Exception as exc:
        return {"error": str(exc)}


def _local_chat(message: str, ticker: str) -> dict[str, Any]:
    from agents.orchestrator import get_agent

    agent = get_agent()
    history = st.session_state.get("ai_chat_history", [])
    return agent.chat(message, ticker=ticker or None, history=history[-10:])


def render_ai_analyst_page(st_module, page_title_fn, render_section_card_start, render_section_card_end):
    """Render the AI Analyst page. Called from app.py main()."""
    st = st_module
    page_title_fn(st, "AI Analyst", "Agentic AI with RAG over earnings transcripts")

    settings = get_settings()

    cfg_left, cfg_right = st.columns([1, 1])
    with cfg_left:
        use_api = st.toggle("Use FastAPI backend", value=False, help="Route requests through the REST API")
    with cfg_right:
        ticker = st.text_input(
            "Focus ticker",
            value=st.session_state.get("global_ticker_search", "AAPL"),
        ).upper().strip()
        st.session_state["global_ticker_search"] = ticker

    if use_api:
        health = _api_request("GET", "/health")
        if "error" in health:
            st.warning(health["error"])

    tab_chat, tab_analyze, tab_rag, tab_setup = st.tabs(
        ["Chat", "Full Analysis", "RAG Search", "Setup"]
    )

    if "ai_chat_history" not in st.session_state:
        st.session_state["ai_chat_history"] = []
    if "ai_messages" not in st.session_state:
        st.session_state["ai_messages"] = []

    with tab_chat:
        render_section_card_start(st, "Agentic Chat", "Ask questions about stocks, sentiment, earnings, and forecasts")

        for msg in st.session_state["ai_messages"]:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg.get("tools_used"):
                    st.caption(f"Tools: {', '.join(msg['tools_used'])}")

        prompt = st.chat_input(f"Ask about {ticker} or any stock...")
        if prompt:
            try:
                if use_api:
                    result = _api_request(
                        "POST", "/chat",
                        json={"message": prompt, "ticker": ticker},
                    )
                else:
                    result = _local_chat(prompt, ticker)

                if "error" in result:
                    response_text = f"**Error:** {result['error']}"
                    tools_used: list[str] = []
                else:
                    response_text = result.get("response") or "No response generated."
                    tools_used = result.get("tools_used", [])

                st.session_state["ai_messages"].append({"role": "user", "content": prompt})
                st.session_state["ai_messages"].append({
                    "role": "assistant",
                    "content": response_text,
                    "tools_used": tools_used,
                })
                st.session_state["ai_chat_history"].append({"role": "user", "content": prompt})
                st.session_state["ai_chat_history"].append({"role": "assistant", "content": response_text})
                st.rerun()
            except Exception as exc:
                st.error(f"AI Analyst failed: {exc}")
                with st.expander("Details"):
                    st.code(traceback.format_exc())

        if st.button("Clear chat", key="ai_clear_chat"):
            st.session_state["ai_messages"] = []
            st.session_state["ai_chat_history"] = []
            st.rerun()

        render_section_card_end(st)

    with tab_analyze:
        render_section_card_start(st, "Full Ticker Analysis", "Multi-source agent analysis")
        if st.button(f"Analyze {ticker}", type="primary", key="ai_analyze_btn"):
            try:
                with st.spinner(f"Running full analysis on {ticker}..."):
                    if use_api:
                        result = _api_request("POST", "/analyze", json={"ticker": ticker})
                    else:
                        from agents.orchestrator import get_agent
                        result = get_agent().analyze(ticker)
                st.session_state["ai_last_analysis"] = result
            except Exception as exc:
                st.error(f"Analysis failed: {exc}")
                with st.expander("Details"):
                    st.code(traceback.format_exc())

        result = st.session_state.get("ai_last_analysis")
        if result and not result.get("error"):
            st.markdown(result.get("narrative", ""))
            with st.expander("Raw data"):
                st.json({k: v for k, v in result.items() if k != "narrative"})
        elif result and result.get("error"):
            st.error(result["error"])

        render_section_card_end(st)

    with tab_rag:
        render_section_card_start(st, "RAG Search", "Semantic search over earnings transcripts")
        rag_query = st.text_input("Search query", placeholder="What did management say about supply chain?")
        rag_ticker = st.text_input("Filter by ticker (optional)", value=ticker)

        if st.button("Search transcripts", key="ai_rag_search"):
            try:
                with st.spinner("Searching..."):
                    if use_api:
                        result = _api_request(
                            "POST", "/rag/query",
                            json={"query": rag_query, "ticker": rag_ticker or None},
                        )
                    else:
                        from rag.retriever import get_retriever
                        retriever = get_retriever()
                        result = retriever.query_with_context(rag_query, rag_ticker or None)
                        result["chunks"] = retriever.retrieve(rag_query, rag_ticker or None)
                st.session_state["ai_rag_results"] = result
            except Exception as exc:
                st.error(f"RAG search failed: {exc}")

        rag_result = st.session_state.get("ai_rag_results")
        if rag_result:
            if rag_result.get("error"):
                st.error(rag_result["error"])
            elif not rag_result.get("chunks"):
                st.info("No results. Run RAG ingestion from the Setup tab.")
            else:
                for i, chunk in enumerate(rag_result.get("chunks", []), 1):
                    score = chunk.get("relevance_score", "—")
                    src = chunk.get("source_file") or chunk.get("source", "")
                    st.markdown(f"**[{i}] {chunk.get('ticker', '')}** (score: {score}) — {src}")
                    st.text(chunk.get("content", "")[:500])

        render_section_card_end(st)

    with tab_setup:
        render_section_card_start(st, "Setup & Configuration", "RAG indexing, API, and MCP")
        st.markdown("""
        ### Quick start
        1. Use **Chat** tab (leave "Use FastAPI backend" off for direct mode)
        2. If RAG docs = 0, click **Index RAG corpus** below
        3. Set `LLM_PROVIDER` and `HF_TOKEN` in `.env`
        """)

        hf_status = "configured" if settings.hf_token_configured else "not set"
        st.caption(f"Hugging Face Hub token: {hf_status}")

        try:
            from rag.retriever import rag_health
            rag_count = rag_health().get("document_count", 0)
            st.caption(f"RAG corpus: {rag_count} document chunks indexed")
        except Exception as exc:
            with st.expander("RAG setup issue", expanded=False):
                st.error(str(exc))
                st.caption(
                    "If this mentions protobuf, restart the app after `pip install -r requirements.txt`."
                )

        if st.button("Index RAG corpus (all transcripts)", key="ai_rag_ingest"):
            try:
                with st.spinner("Indexing transcripts..."):
                    if use_api:
                        result = _api_request("POST", "/rag/ingest")
                        details = result.get("details", result)
                    else:
                        from rag.ingest import ingest_transcripts
                        details = ingest_transcripts()
                st.success(
                    f"Indexed {details.get('indexed_files', 0)} files, "
                    f"{details.get('total_chunks', 0)} chunks"
                )
            except Exception as exc:
                st.error(str(exc))

        st.code("""# .env
LLM_PROVIDER=groq
GROQ_API_KEY=gsk_...
HF_TOKEN=hf_...
OPENAI_API_KEY=sk-...""", language="bash")

        render_section_card_end(st)

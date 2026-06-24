"""AI Analyst UI — uses backend.client (direct mode by default)."""

from __future__ import annotations

import sys
import traceback
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import streamlit as st

from backend.client import (
    analyze_ticker,
    chat_message,
    ingest_rag_corpus,
    rag_health_status,
    rag_search,
    runtime_mode_label,
    should_use_remote_api,
)
from config.settings import get_settings


def render_ai_analyst_page(st_module, page_title_fn, render_section_card_start, render_section_card_end):
    """Render the AI Analyst page. Called from app.py main()."""
    st = st_module
    page_title_fn(st, "AI Analyst", "Agentic AI with RAG over earnings transcripts")

    settings = get_settings()

    ticker = st.text_input(
        "Focus ticker",
        value=st.session_state.get("global_ticker_search", "AAPL"),
    ).upper().strip()
    st.session_state["global_ticker_search"] = ticker

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
                history = st.session_state.get("ai_chat_history", [])
                result = chat_message(prompt, ticker=ticker, history=history[-10:])

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
                    result = analyze_ticker(ticker)
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
                    result = rag_search(rag_query, ticker=rag_ticker or None)
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
        render_section_card_start(st, "Setup & Configuration", "RAG indexing and runtime mode")
        st.caption(f"Runtime: **{runtime_mode_label()}**")

        if should_use_remote_api():
            st.info(
                "Remote API mode is enabled (`USE_API=true`). "
                "Ensure `API_URL` points to your deployed FastAPI service."
            )
        else:
            st.markdown("""
            ### Quick start (direct mode — recommended for Streamlit Cloud)
            1. Use **Chat** or **Full Analysis** tabs
            2. If RAG corpus is empty, click **Index RAG corpus** below
            3. Set secrets in Streamlit Cloud or `.env` locally (`LLM_PROVIDER`, API keys, `HF_TOKEN`)
            """)

        hf_status = "configured" if settings.hf_token_configured else "not set"
        st.caption(f"Hugging Face Hub token: {hf_status}")

        try:
            health = rag_health_status()
            if health.get("error"):
                with st.expander("RAG setup issue", expanded=False):
                    st.error(str(health["error"]))
            else:
                rag_count = health.get("document_count", 0)
                st.caption(f"RAG corpus: {rag_count} document chunks indexed")
        except Exception as exc:
            with st.expander("RAG setup issue", expanded=False):
                st.error(str(exc))

        if st.button("Index RAG corpus (all transcripts)", key="ai_rag_ingest"):
            try:
                with st.spinner("Indexing transcripts..."):
                    details = ingest_rag_corpus()
                if isinstance(details, dict) and details.get("error"):
                    st.error(details["error"])
                else:
                    st.success(
                        f"Indexed {details.get('indexed_files', 0)} files, "
                        f"{details.get('total_chunks', 0)} chunks"
                    )
            except Exception as exc:
                st.error(str(exc))

        st.code("""# Streamlit Cloud secrets (.streamlit/secrets.toml) — direct mode
USE_API = false
LLM_PROVIDER = "groq"
GROQ_API_KEY = "gsk_..."
HF_TOKEN = "hf_..."

# Optional: remote FastAPI (Render / Railway / etc.)
# USE_API = true
# API_URL = "https://your-fastapi.onrender.com"
""", language="toml")

        render_section_card_end(st)

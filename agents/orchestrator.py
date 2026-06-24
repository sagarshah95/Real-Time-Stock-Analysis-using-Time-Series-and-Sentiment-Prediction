"""LLM agent orchestrator with tool-calling loop."""

from __future__ import annotations

import json
import logging
from typing import Any, Optional

from config.settings import get_settings
from agents.prompts import SYSTEM_PROMPT, ANALYSIS_PROMPT
from agents.tools import TOOL_DEFINITIONS, execute_tool, tool_analyze_ticker

logger = logging.getLogger(__name__)

# Fields Groq/Gemini reject when replaying OpenAI SDK assistant messages
_UNSUPPORTED_MSG_FIELDS = frozenset({
    "annotations", "refusal", "audio", "function_call",
})


def _sanitize_assistant_message(assistant_msg) -> dict[str, Any]:
    """Build a provider-safe assistant message for multi-turn tool calling."""
    msg: dict[str, Any] = {"role": "assistant"}
    if assistant_msg.content:
        msg["content"] = assistant_msg.content

    if assistant_msg.tool_calls:
        msg["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                },
            }
            for tc in assistant_msg.tool_calls
        ]
    elif not assistant_msg.content:
        msg["content"] = ""

    return msg


class StockAnalysisAgent:
    """Agentic AI orchestrator using OpenAI-compatible tool calling."""

    def __init__(self):
        self.settings = get_settings()
        self._client = None

    def _get_client(self):
        if self._client is not None:
            return self._client
        if not self.settings.llm_configured:
            return None
        try:
            from openai import OpenAI

            self._client = OpenAI(
                api_key=self.settings.resolved_api_key,
                base_url=self.settings.resolved_base_url,
            )
        except Exception as exc:
            logger.error("Failed to init OpenAI client: %s", exc)
        return self._client

    def chat(
        self,
        message: str,
        ticker: Optional[str] = None,
        history: Optional[list[dict[str, str]]] = None,
    ) -> dict[str, Any]:
        """Process a user message through the agent loop."""
        if self.settings.llm_configured and self.settings.enable_llm_agent:
            return self._llm_chat(message, ticker, history)

        return self._fallback_chat(message, ticker)

    def _llm_chat(
        self,
        message: str,
        ticker: Optional[str],
        history: Optional[list[dict[str, str]]],
    ) -> dict[str, Any]:
        client = self._get_client()
        if client is None:
            return self._fallback_chat(message, ticker)

        messages: list[dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT}]
        if history:
            messages.extend(history)
        if ticker:
            messages.append({"role": "system", "content": f"User is analyzing ticker: {ticker.upper()}"})
        messages.append({"role": "user", "content": message})

        tools_used: list[str] = []
        max_iter = self.settings.agent_max_iterations

        for _ in range(max_iter):
            response = client.chat.completions.create(
                model=self.settings.resolved_model,
                messages=messages,
                tools=TOOL_DEFINITIONS,
                tool_choice="auto",
                temperature=self.settings.llm_temperature,
            )
            choice = response.choices[0]
            assistant_msg = choice.message

            if assistant_msg.tool_calls:
                messages.append(_sanitize_assistant_message(assistant_msg))
                for tc in assistant_msg.tool_calls:
                    fn_name = tc.function.name
                    try:
                        args = json.loads(tc.function.arguments)
                    except json.JSONDecodeError:
                        args = {}
                    tools_used.append(fn_name)
                    result = execute_tool(fn_name, args)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result,
                    })
            else:
                return {
                    "response": assistant_msg.content or "",
                    "mode": "llm_agent",
                    "model": self.settings.resolved_model,
                    "provider": self.settings.active_provider,
                    "tools_used": tools_used,
                    "ticker": ticker,
                }

        return {
            "response": "I reached the maximum analysis steps. Please try a more specific question.",
            "mode": "llm_agent",
            "tools_used": tools_used,
            "ticker": ticker,
        }

    def _fallback_chat(self, message: str, ticker: Optional[str]) -> dict[str, Any]:
        """Deterministic pipeline when no LLM API key is configured."""
        msg_lower = message.lower()
        sym = (ticker or "").upper()

        if not sym:
            for word in message.upper().split():
                if word.isalpha() and 1 <= len(word) <= 5:
                    sym = word
                    break
            if not sym:
                sym = "AAPL"

        tools_used = ["analyze_ticker"]
        if any(kw in msg_lower for kw in ("earnings", "transcript", "management", "ceo", "cfo")):
            from agents.tools import tool_search_earnings_transcripts
            rag_result = tool_search_earnings_transcripts(message, sym)
            analysis = json.loads(tool_analyze_ticker(sym))
            response = self._format_fallback_response(sym, analysis, rag_result)
            tools_used.append("search_earnings_transcripts")
        elif any(kw in msg_lower for kw in ("news", "sentiment", "headline")):
            from agents.tools import tool_get_news_sentiment
            result = tool_get_news_sentiment(sym)
            analysis = json.loads(result)
            response = self._format_sentiment_response(sym, analysis)
            tools_used = ["get_news_sentiment"]
        elif any(kw in msg_lower for kw in ("forecast", "predict", "future", "price target")):
            from agents.tools import tool_forecast_stock_price
            result = tool_forecast_stock_price(sym, years=1)
            analysis = json.loads(result)
            response = self._format_forecast_response(sym, analysis)
            tools_used = ["forecast_stock_price"]
        else:
            analysis = json.loads(tool_analyze_ticker(sym))
            response = self._format_fallback_response(sym, analysis)

        return {
            "response": response,
            "mode": "deterministic",
            "tools_used": tools_used,
            "ticker": sym,
            "note": "Set OPENAI_API_KEY for full agentic AI with multi-step reasoning.",
        }

    def _format_fallback_response(self, sym: str, analysis: dict, rag: str = "") -> str:
        quote = analysis.get("quote", {})
        stats = quote.get("stats", {})
        sentiment = analysis.get("sentiment", {})
        tech = analysis.get("technicals", {})

        lines = [
            f"## Analysis: {sym}",
            "",
            f"**Price:** {stats.get('price', 'N/A')} ({stats.get('change_pct', 'N/A')})",
            f"**Volume:** {stats.get('volume', 'N/A')}",
            f"**Technical Signal:** {tech.get('signal', 'N/A')} (SMA5: {tech.get('sma_5')}, SMA20: {tech.get('sma_20')})",
            f"**MACD Histogram:** {tech.get('macd_histogram', 'N/A')}",
            "",
            f"**News Sentiment:** {sentiment.get('overall', 'N/A')} "
            f"({sentiment.get('positive', 0)}+ / {sentiment.get('negative', 0)}- / {sentiment.get('neutral', 0)} neutral)",
        ]

        headlines = sentiment.get("headlines", [])[:3]
        if headlines:
            lines.append("\n**Recent Headlines:**")
            for h in headlines:
                lines.append(f"- [{h.get('sentiment')}] {h.get('headline')}")

        if rag:
            try:
                rag_data = json.loads(rag)
                if rag_data.get("context"):
                    lines.append("\n**Earnings Transcript Context:**")
                    lines.append(rag_data["context"][:1500])
                    if len(rag_data["context"]) > 1500:
                        lines.append("...(truncated)")
            except json.JSONDecodeError:
                pass

        lines.append("\n*This is automated analysis, not financial advice.*")
        return "\n".join(lines)

    def _format_sentiment_response(self, sym: str, data: dict) -> str:
        lines = [
            f"## News Sentiment: {sym}",
            f"**Overall:** {data.get('overall', 'N/A')} (avg compound: {data.get('avg_compound', 0)})",
            f"**Breakdown:** {data.get('positive', 0)} positive, {data.get('negative', 0)} negative, {data.get('neutral', 0)} neutral",
        ]
        for h in data.get("headlines", [])[:5]:
            lines.append(f"- [{h.get('sentiment')}] {h.get('headline')}")
        lines.append("\n*Not financial advice.*")
        return "\n".join(lines)

    def _format_forecast_response(self, sym: str, data: dict) -> str:
        return (
            f"## Price Forecast: {sym}\n\n"
            f"**Method:** {data.get('method', 'N/A')}\n"
            f"**Last Close:** ${data.get('last_close', 'N/A')}\n"
            f"**Projected ({data.get('years', 1)}yr):** ${data.get('projected_end', 'N/A'):,.2f}\n\n"
            f"{data.get('disclaimer', '')}"
        )

    def analyze(self, ticker: str) -> dict[str, Any]:
        """Structured full analysis for a ticker."""
        result = json.loads(tool_analyze_ticker(ticker))
        if self.settings.llm_configured and self.settings.enable_llm_agent:
            client = self._get_client()
            if client:
                prompt = ANALYSIS_PROMPT.format(ticker=ticker.upper())
                llm_result = self._llm_chat(prompt, ticker.upper(), None)
                result["narrative"] = llm_result.get("response", "")
                result["mode"] = "llm_agent"
                return result

        result["narrative"] = self._format_fallback_response(ticker.upper(), result)
        result["mode"] = "deterministic"
        return result


_agent: Optional[StockAnalysisAgent] = None


def get_agent() -> StockAnalysisAgent:
    global _agent
    if _agent is None:
        _agent = StockAnalysisAgent()
    return _agent

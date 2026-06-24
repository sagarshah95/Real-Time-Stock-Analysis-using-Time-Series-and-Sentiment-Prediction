"""Social trends UI — real-time X tweets via Kafka pipeline."""

from __future__ import annotations

from html import escape
from typing import Any, Callable

import pandas as pd

from config.settings import get_settings
from services.x_trends_store import (
    get_bucket_stats,
    get_recent_tweets,
    tweets_to_activity_series,
)
from services.x_twitter import XAuthError, XCreditsError, build_x_search_query, keyword_to_ticker, verify_x_auth
from streaming.kafka_config import invalidate_kafka_cache, kafka_available, kafka_status
from streaming.x_consumer import is_consumer_running, start_x_consumer
from streaming.x_producer import backfill_keyword, is_producer_running, start_x_producer


def _sentiment_color(label: str) -> str:
    if label == "Positive":
        return "#16a34a"
    if label == "Negative":
        return "#dc2626"
    return "#64748b"


def ensure_x_pipeline(keyword: str) -> dict[str, Any]:
    """Start Kafka consumer + X producer threads for the keyword."""
    settings = get_settings()
    status = {
        "x_configured": settings.x_configured,
        "kafka_enabled": settings.kafka_enabled,
        "kafka_connected": kafka_available(),
        "consumer_running": False,
        "producer_running": False,
    }
    if not settings.x_configured:
        return status

    if settings.kafka_enabled and kafka_available():
        status["consumer_running"] = start_x_consumer()
    status["producer_running"] = start_x_producer(keyword)
    return status


def render_x_social_trends_page(
    st,
    *,
    page_title: Callable,
    render_page_hero: Callable,
    render_section_card_start: Callable,
    render_section_card_end: Callable,
    render_sentiment_badge: Callable,
    make_pred: Callable,
) -> None:
    settings = get_settings()

    page_title(st, "Social trends", "Real-time X posts via Kafka stream")
    render_page_hero(
        st,
        icon="𝕏",
        kicker="Live social signal",
        title="Real-time X trend analysis",
        subtitle=(
            "Tweets about your stock are pulled from the **X API**, published to a **Kafka** topic, "
            "aggregated for volume and VADER sentiment, then forecast forward."
        ),
        pills=["X API", "Kafka stream", "VADER sentiment", "Live poll"],
    )

    st.sidebar.write("## Stock ticker or company")
    keyword = st.sidebar.text_input("Keyword or ticker", "Amazon", key="x_social_keyword")
    periods = st.sidebar.slider("Forecast horizon (days)", 7, 365, 90, key="x_social_periods")
    auto_refresh = st.sidebar.checkbox("Auto-refresh every 60s", value=True, key="x_social_autorefresh")

    kafka_state = kafka_status()
    kafka_ok = kafka_state["ok"]
    if st.sidebar.button("Reconnect Kafka", use_container_width=True):
        invalidate_kafka_cache()
        kafka_state = kafka_status(force=True)
        kafka_ok = kafka_state["ok"]
        st.rerun()
    st.sidebar.caption(
        f"Kafka: **{'connected' if kafka_ok else 'offline — direct ingest mode'}** "
        f"(`{settings.kafka_bootstrap_servers}`)"
    )
    if not kafka_ok and kafka_state.get("detail"):
        st.sidebar.caption(kafka_state["detail"])

    if not settings.x_configured:
        st.error(
            "X API credentials are missing. Set `X_API_KEY`, `X_API_SECRET`, "
            "`X_ACCESS_TOKEN`, and `X_ACCESS_TOKEN_SECRET` in `.env`, then restart the app."
        )
        with st.expander("Credential checklist"):
            for label, ok in get_settings().x_credential_status().items():
                if label == "auth_mode":
                    st.write(f"**{label}:** {ok}")
                else:
                    st.write(f"{'✅' if ok else '❌'} {label}")
        st.stop()

    creds = get_settings().x_credential_status()
    if settings.x_auth_mode.strip().lower() == "oauth1" and not creds.get("oauth1_complete"):
        st.error(
            "OAuth1 mode requires all four keys: `X_API_KEY`, `X_API_SECRET`, "
            "`X_ACCESS_TOKEN`, `X_ACCESS_TOKEN_SECRET`."
        )
        st.stop()

    ticker = keyword_to_ticker(keyword)
    query = build_x_search_query(keyword)
    st.info(f"X search: `{query}` · ticker **{ticker}**")

    auth = verify_x_auth()
    if auth["ok"]:
        st.caption(f"X connected as **@{auth['username']}** via {auth['method']} auth")
        if not auth.get("search_ok"):
            st.warning(
                auth.get("search_error")
                or "Tweet search is not available on your current X API plan."
            )
            st.info(
                "Credentials are valid, but **tweet search requires X API credits** (pay-per-use). "
                "Add credits in the [X Developer Portal](https://developer.x.com/) to enable the live stream."
            )
    else:
        st.error(
            f"X API authentication failed: {auth['error']}. "
            "Regenerate your keys at [developer.x.com](https://developer.x.com/), update `.env`, "
            "set `X_AUTH_MODE=oauth1`, and restart Streamlit."
        )
        st.stop()

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        if st.button("Backfill recent tweets", use_container_width=True, disabled=not auth.get("search_ok")):
            with st.spinner("Fetching from X…"):
                try:
                    n = backfill_keyword(keyword)
                    st.success(f"Loaded {n} tweets from X.")
                except XCreditsError as exc:
                    st.error(str(exc))
                except XAuthError as exc:
                    st.error(str(exc))
                except Exception as exc:
                    st.error(f"Backfill failed: {exc}")
    with col_b:
        if st.button("Refresh now", use_container_width=True):
            st.rerun()
    with col_c:
        stream_on = auth.get("search_ok") and is_producer_running(keyword)
        st.metric("Stream", "Live" if stream_on else ("Paused" if not auth.get("search_ok") else "Starting…"))

    if not auth.get("search_ok"):
        st.caption("Live stream paused until X API search credits are available.")
        pipe = {"kafka_connected": kafka_ok, "producer_running": False}
    else:
        pipe = ensure_x_pipeline(keyword)
    if pipe["kafka_connected"]:
        st.success(
            f"Kafka pipeline active · consumer: "
            f"{'running' if is_consumer_running() else 'starting'} · "
            f"producer: {'running' if pipe['producer_running'] else 'off'}"
        )
    else:
        detail = kafka_status().get("detail", "")
        st.info(
            "**Direct ingest mode** — tweets are saved without Kafka. "
            + (detail if detail else "Start Kafka to enable the full broker pipeline.")
        )

    stats = get_bucket_stats(keyword)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Tweets indexed", stats.get("tweet_count", 0))
    m2.metric("Ticker", stats.get("ticker", ticker))
    m3.metric("Avg sentiment", f"{stats.get('avg_sentiment', 0):.2f}")
    m4.metric("Last update", (stats.get("updated_at") or "—")[:19])

    df = tweets_to_activity_series(keyword)
    if df.empty or len(df) < 3:
        st.info(
            "Not enough tweet volume yet. Click **Backfill recent tweets** or wait for the live stream "
            f"(polls every {settings.x_stream_poll_seconds}s)."
        )
    else:
        try:
            forecast, fig1, fig2 = make_pred(df, periods, show_fallback_caption=False)
            st.pyplot(fig1)
            st.write("Tweet volume patterns (monthly / weekday)")
            st.pyplot(fig2)
        except Exception as exc:
            st.warning(f"Could not build forecast: {exc}")
            st.line_chart(df.set_index("ds")["y"])

    render_section_card_start(st, "Live tweet feed", "Most recent posts from the Kafka/X pipeline")
    tweets = get_recent_tweets(keyword, limit=25)
    if not tweets:
        st.info("No tweets in the stream yet for this keyword.")
    else:
        for tw in tweets:
            label = tw.get("sentiment", "Neutral")
            color = _sentiment_color(label)
            text = escape(tw.get("text", ""))
            created = escape((tw.get("created_at") or "")[:19].replace("T", " "))
            likes = tw.get("like_count", 0)
            rts = tw.get("retweet_count", 0)
            st.markdown(
                f"""
                <div style="padding:0.75rem 0;border-bottom:1px solid rgba(128,128,128,0.2);">
                  <div style="font-size:0.8rem;opacity:0.75;">{created} · ♥ {likes} · ↻ {rts}</div>
                  <div style="margin:0.35rem 0;">{text}</div>
                  <span style="color:{color};font-weight:600;font-size:0.85rem;">{label}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
    render_section_card_end(st)

    if auto_refresh:
        import time
        time.sleep(60)
        st.rerun()

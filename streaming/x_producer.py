"""Kafka producer — polls X API and publishes tweets to the stream topic."""

from __future__ import annotations

import logging
import threading
import time
from typing import Callable

from config.settings import get_settings
from services.x_trends_store import get_since_id, ingest_tweet
from services.x_twitter import XAuthError, XCreditsError, fetch_recent_tweets
from streaming.kafka_config import create_producer, kafka_available, publish_tweet

logger = logging.getLogger(__name__)

_ACTIVE: dict[str, threading.Event] = {}
_THREADS: dict[str, threading.Thread] = {}


def _producer_loop(keyword: str, stop_event: threading.Event) -> None:
    settings = get_settings()
    use_kafka = kafka_available()
    producer = create_producer() if use_kafka else None
    kw = str(keyword or "").strip() or "Amazon"

    logger.info("X producer started for %s (kafka=%s)", kw, use_kafka)

    while not stop_event.is_set():
        try:
            since_id = get_since_id(kw)
            tweets = fetch_recent_tweets(kw, since_id=since_id, max_results=100)
            for tw in tweets:
                if use_kafka and producer:
                    publish_tweet(producer, tw)
                else:
                    ingest_tweet(tw)
        except XCreditsError as exc:
            logger.error("X API credits required for %s: %s — stopping producer", kw, exc)
            stop_event.set()
            return
        except XAuthError as exc:
            logger.error("X auth failed for %s: %s — stopping producer until restart", kw, exc)
            stop_event.set()
            return
        except Exception as exc:
            logger.warning("X producer error for %s: %s", kw, exc)

        stop_event.wait(settings.x_stream_poll_seconds)

    if producer:
        producer.close()
    logger.info("X producer stopped for %s", kw)


def start_x_producer(keyword: str) -> bool:
    """Start background producer thread for keyword (idempotent)."""
    kw = str(keyword or "").strip() or "Amazon"
    if kw in _THREADS and _THREADS[kw].is_alive():
        return True

    stop_event = threading.Event()
    thread = threading.Thread(
        target=_producer_loop,
        args=(kw, stop_event),
        name=f"x-producer-{kw}",
        daemon=True,
    )
    _ACTIVE[kw] = stop_event
    _THREADS[kw] = thread
    thread.start()
    return True


def stop_x_producer(keyword: str) -> None:
    kw = str(keyword or "").strip() or "Amazon"
    ev = _ACTIVE.pop(kw, None)
    if ev:
        ev.set()


def is_producer_running(keyword: str) -> bool:
    kw = str(keyword or "").strip() or "Amazon"
    t = _THREADS.get(kw)
    return bool(t and t.is_alive())


def backfill_keyword(keyword: str, on_progress: Callable[[str], None] | None = None) -> int:
    """One-shot backfill from X search API into store/Kafka."""
    settings = get_settings()
    use_kafka = kafka_available()
    producer = create_producer() if use_kafka else None
    kw = str(keyword or "").strip() or "Amazon"

    if on_progress:
        on_progress(f"Fetching last {settings.x_tweet_lookback_days} days of X posts for {kw}…")

    tweets = fetch_recent_tweets(kw, since_id=None, max_results=100)
    for tw in tweets:
        ingest_tweet(tw)
        if use_kafka and producer:
            publish_tweet(producer, tw)

    if producer:
        producer.close()

    if on_progress:
        on_progress(f"Indexed {len(tweets)} tweets.")
    return len(tweets)

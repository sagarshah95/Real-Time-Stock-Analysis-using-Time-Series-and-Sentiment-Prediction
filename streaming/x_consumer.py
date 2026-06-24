"""Kafka consumer — ingests tweets from the stream into the trend store."""

from __future__ import annotations

import logging
import threading

from services.x_trends_store import ingest_tweet
from streaming.kafka_config import create_consumer, kafka_available

logger = logging.getLogger(__name__)

_CONSUMER_THREAD: threading.Thread | None = None
_STOP = threading.Event()


def _consumer_loop(stop_event: threading.Event) -> None:
    consumer = create_consumer()
    if consumer is None:
        logger.warning("Kafka consumer could not connect.")
        return

    logger.info("Kafka X consumer listening…")
    try:
        while not stop_event.is_set():
            polled = consumer.poll(timeout_ms=1000)
            for _tp, records in polled.items():
                for record in records:
                    tweet = record.value
                    if isinstance(tweet, dict):
                        ingest_tweet(tweet)
    finally:
        consumer.close()
        logger.info("Kafka X consumer stopped.")


def start_x_consumer() -> bool:
    """Start background Kafka consumer (idempotent)."""
    global _CONSUMER_THREAD

    if not kafka_available():
        return False

    if _CONSUMER_THREAD and _CONSUMER_THREAD.is_alive():
        return True

    _STOP.clear()
    _CONSUMER_THREAD = threading.Thread(
        target=_consumer_loop,
        args=(_STOP,),
        name="x-kafka-consumer",
        daemon=True,
    )
    _CONSUMER_THREAD.start()
    return True


def stop_x_consumer() -> None:
    _STOP.set()


def is_consumer_running() -> bool:
    return bool(_CONSUMER_THREAD and _CONSUMER_THREAD.is_alive())

"""Kafka helpers for the X tweet stream."""

from __future__ import annotations

import json
import socket
import time
from typing import Any

from config.settings import get_settings, reload_settings

_CACHE: dict[str, Any] = {"ts": 0.0, "ok": False, "reason": "", "detail": ""}


def _bootstrap_hosts() -> list[str]:
    reload_settings()
    raw = get_settings().kafka_bootstrap_servers.strip()
    return [h.strip() for h in raw.split(",") if h.strip()]


def _socket_reachable(host: str, port: int, timeout: float = 2.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _broker_reachable() -> bool:
    for hostport in _bootstrap_hosts():
        if ":" in hostport:
            host, port_s = hostport.rsplit(":", 1)
            try:
                port = int(port_s)
            except ValueError:
                continue
        else:
            host, port = hostport, 9092
        if _socket_reachable(host, port):
            return True
    return False


def invalidate_kafka_cache() -> None:
    _CACHE.update({"ts": 0.0, "ok": False, "reason": "", "detail": ""})


def kafka_status(*, force: bool = False, cache_seconds: float = 20.0) -> dict[str, Any]:
    """Detailed Kafka readiness: {ok, reason, detail}."""
    settings = reload_settings()
    if not settings.kafka_enabled:
        return {"ok": False, "reason": "disabled", "detail": "KAFKA_ENABLED=false in .env"}

    try:
        import kafka  # noqa: F401
    except ImportError:
        return {
            "ok": False,
            "reason": "missing_package",
            "detail": "Install kafka-python: pip install kafka-python",
        }

    now = time.time()
    if not force and now - float(_CACHE.get("ts", 0)) < cache_seconds:
        return {
            "ok": bool(_CACHE.get("ok")),
            "reason": _CACHE.get("reason", ""),
            "detail": _CACHE.get("detail", ""),
        }

    hosts = _bootstrap_hosts()
    if not _broker_reachable():
        result = {
            "ok": False,
            "reason": "broker_unreachable",
            "detail": (
                f"No broker at {', '.join(hosts)}. "
                "Run scripts/start_kafka.bat or: docker compose -f infra/docker-compose.kafka.yml up -d"
            ),
        }
        _CACHE.update({"ts": now, **result})
        return result

    producer = create_producer()
    if producer is None:
        result = {
            "ok": False,
            "reason": "producer_failed",
            "detail": "Could not create Kafka producer.",
        }
        _CACHE.update({"ts": now, **result})
        return result

    connected = False
    try:
        connected = bool(producer.bootstrap_connected())
    except Exception as exc:
        result = {
            "ok": False,
            "reason": "connect_failed",
            "detail": str(exc),
        }
        _CACHE.update({"ts": now, **result})
        return result
    finally:
        try:
            producer.close()
        except Exception:
            pass

    if not connected:
        result = {
            "ok": False,
            "reason": "not_connected",
            "detail": f"Broker port open but client could not connect to {', '.join(hosts)}",
        }
    else:
        result = {
            "ok": True,
            "reason": "connected",
            "detail": f"Broker {', '.join(hosts)}",
        }
    _CACHE.update({"ts": now, **result})
    return result


def kafka_available(*, force: bool = False, cache_seconds: float = 20.0) -> bool:
    return kafka_status(force=force, cache_seconds=cache_seconds)["ok"]


def create_producer():
    settings = get_settings()
    if not settings.kafka_enabled:
        return None
    try:
        from kafka import KafkaProducer

        return KafkaProducer(
            bootstrap_servers=_bootstrap_hosts(),
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
            key_serializer=lambda k: k.encode("utf-8") if k else None,
            acks="all",
            retries=3,
            request_timeout_ms=8000,
            api_version_auto_timeout_ms=8000,
            reconnect_backoff_max_ms=2000,
        )
    except Exception:
        return None


def create_consumer(group_id: str = "fast-x-trends-consumer"):
    settings = get_settings()
    if not settings.kafka_enabled:
        return None
    try:
        from kafka import KafkaConsumer

        return KafkaConsumer(
            settings.kafka_x_tweets_topic,
            bootstrap_servers=_bootstrap_hosts(),
            group_id=group_id,
            value_deserializer=lambda v: json.loads(v.decode("utf-8")),
            auto_offset_reset="latest",
            enable_auto_commit=True,
            request_timeout_ms=8000,
            api_version_auto_timeout_ms=8000,
        )
    except Exception:
        return None


def publish_tweet(producer, tweet: dict[str, Any]) -> bool:
    if producer is None:
        return False
    settings = get_settings()
    key = str(tweet.get("ticker") or tweet.get("keyword") or "unknown")
    try:
        producer.send(settings.kafka_x_tweets_topic, key=key, value=tweet)
        producer.flush(timeout=8)
        return True
    except Exception:
        return False

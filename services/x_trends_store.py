"""Persistent store for X tweet aggregates (Kafka consumer output)."""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from config.settings import get_settings
from services.x_twitter import keyword_to_ticker

_LOCK = threading.Lock()
_STATE_FILE = "state.json"
_MAX_TWEETS_PER_KEY = 300


def _state_path() -> Path:
    return get_settings().x_trends_data_dir / _STATE_FILE


def _empty_bucket() -> dict[str, Any]:
    return {
        "ticker": "",
        "keyword": "",
        "query": "",
        "last_since_id": None,
        "hourly": {},
        "tweets": [],
        "updated_at": None,
    }


def _load_state() -> dict[str, Any]:
    path = _state_path()
    if not path.exists():
        return {}
    try:
        with path.open(encoding="utf-8") as fh:
            return json.load(fh)
    except (json.JSONDecodeError, OSError):
        return {}


def _save_state(state: dict[str, Any]) -> None:
    path = _state_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(state, fh, indent=2)
    tmp.replace(path)


def _bucket_key(keyword: str) -> str:
    return keyword_to_ticker(keyword).upper()


def _hour_key(iso_ts: str) -> str:
    ts = pd.to_datetime(iso_ts, utc=True)
    return ts.floor("h").isoformat()


def ingest_tweet(tweet: dict[str, Any]) -> None:
    """Append one tweet to the rolling store (idempotent by tweet id)."""
    keyword = str(tweet.get("keyword") or tweet.get("ticker") or "AMZN")
    key = _bucket_key(keyword)
    with _LOCK:
        state = _load_state()
        bucket = state.get(key, _empty_bucket())
        bucket["ticker"] = tweet.get("ticker") or keyword_to_ticker(keyword)
        bucket["keyword"] = keyword

        existing_ids = {str(t.get("id")) for t in bucket.get("tweets", [])}
        tid = str(tweet.get("id", ""))
        if tid and tid in existing_ids:
            _save_state(state)
            return

        tweets: list[dict[str, Any]] = bucket.get("tweets", [])
        tweets.append(tweet)
        tweets.sort(key=lambda t: t.get("created_at", ""))
        bucket["tweets"] = tweets[-_MAX_TWEETS_PER_KEY:]

        hour = _hour_key(str(tweet.get("created_at", datetime.now(timezone.utc).isoformat())))
        hourly: dict[str, int] = bucket.get("hourly", {})
        hourly[hour] = int(hourly.get(hour, 0)) + 1
        bucket["hourly"] = hourly

        if tid:
            prev = bucket.get("last_since_id")
            if not prev or int(tid) > int(prev):
                bucket["last_since_id"] = tid

        bucket["updated_at"] = datetime.now(timezone.utc).isoformat()
        state[key] = bucket
        _save_state(state)


def ingest_tweets(tweets: list[dict[str, Any]]) -> int:
    added = 0
    for tw in tweets:
        before = get_bucket_stats(tw.get("keyword") or tw.get("ticker") or "AMZN")
        ingest_tweet(tw)
        after = get_bucket_stats(tw.get("keyword") or tw.get("ticker") or "AMZN")
        if after.get("tweet_count", 0) > before.get("tweet_count", 0):
            added += 1
    return added


def get_since_id(keyword: str) -> str | None:
    key = _bucket_key(keyword)
    with _LOCK:
        bucket = _load_state().get(key, {})
    return bucket.get("last_since_id")


def get_recent_tweets(keyword: str, limit: int = 20) -> list[dict[str, Any]]:
    key = _bucket_key(keyword)
    with _LOCK:
        bucket = _load_state().get(key, {})
    tweets = list(bucket.get("tweets", []))
    tweets.sort(key=lambda t: t.get("created_at", ""), reverse=True)
    return tweets[:limit]


def get_bucket_stats(keyword: str) -> dict[str, Any]:
    key = _bucket_key(keyword)
    with _LOCK:
        bucket = _load_state().get(key, _empty_bucket())
    tweets = bucket.get("tweets", [])
    hourly = bucket.get("hourly", {})
    compounds = [float(t.get("sentiment_compound", 0)) for t in tweets]
    avg_sent = float(np.mean(compounds)) if compounds else 0.0
    return {
        "ticker": bucket.get("ticker") or keyword_to_ticker(keyword),
        "keyword": bucket.get("keyword") or keyword,
        "tweet_count": len(tweets),
        "hourly_buckets": len(hourly),
        "avg_sentiment": round(avg_sent, 4),
        "updated_at": bucket.get("updated_at"),
        "last_since_id": bucket.get("last_since_id"),
    }


def tweets_to_activity_series(keyword: str) -> pd.DataFrame:
    """
    Daily tweet volume scaled 0–100 for forecasting charts.
    Built from Kafka-ingested X tweets in the local store.
    """
    key = _bucket_key(keyword)
    with _LOCK:
        bucket = _load_state().get(key, {})
    hourly: dict[str, int] = bucket.get("hourly", {})

    if not hourly:
        tweets = bucket.get("tweets", [])
        if not tweets:
            return pd.DataFrame(columns=["ds", "y"])
        hourly = {}
        for tw in tweets:
            hour = _hour_key(str(tw.get("created_at", "")))
            hourly[hour] = int(hourly.get(hour, 0)) + 1

    idx = pd.to_datetime(list(hourly.keys()), utc=True)
    counts = pd.Series(list(hourly.values()), index=idx).sort_index()
    daily = counts.resample("D").sum()
    if daily.empty:
        return pd.DataFrame(columns=["ds", "y"])

    daily.index = daily.index.normalize()
    end_d = daily.index.max()
    start_d = max(daily.index.min(), end_d - pd.Timedelta(days=150))
    full_idx = pd.date_range(start_d, end_d, freq="D")
    aligned = daily.reindex(full_idx.normalize(), fill_value=0.0).astype(float)
    y = aligned.values
    ymin, ymax = float(y.min()), float(y.max())
    if ymax > ymin:
        y = 100.0 * (y - ymin) / (ymax - ymin)
    else:
        y = np.full_like(y, 50.0, dtype=float)
    return pd.DataFrame({"ds": full_idx, "y": y})

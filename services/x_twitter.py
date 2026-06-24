"""X (Twitter) API client — fetch tweets for stock/cashtag keywords."""

from __future__ import annotations

import logging
import re
from datetime import datetime, timedelta, timezone
from typing import Any, Iterator

import tweepy
from tweepy.errors import Forbidden, TooManyRequests, TweepyException, Unauthorized
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from config.settings import get_settings, reload_settings

logger = logging.getLogger(__name__)

_TICKER_ALIAS = {
    "amazon": "AMZN",
    "google": "GOOGL",
    "alphabet": "GOOGL",
    "apple": "AAPL",
    "microsoft": "MSFT",
    "meta": "META",
    "facebook": "META",
    "netflix": "NFLX",
    "tesla": "TSLA",
    "nvidia": "NVDA",
    "amd": "AMD",
    "intel": "INTC",
    "disney": "DIS",
    "walmart": "WMT",
    "nike": "NKE",
    "coca cola": "KO",
    "coca-cola": "KO",
    "jpmorgan": "JPM",
    "jp morgan": "JPM",
    "bank of america": "BAC",
    "goldman": "GS",
    "berkshire": "BRK-B",
    "berkshire hathaway": "BRK-B",
}

_vader = SentimentIntensityAnalyzer()


class XAuthError(RuntimeError):
    """Raised when X API credentials are rejected."""


class XCreditsError(RuntimeError):
    """Raised when the X API account has no credits for search/stream endpoints."""


def keyword_to_ticker(keyword: str) -> str:
    s = str(keyword or "").strip() or "AMZN"
    low = s.lower()
    if low in _TICKER_ALIAS:
        return _TICKER_ALIAS[low]
    compact = re.sub(r"\s+", "", s).upper()
    if re.fullmatch(r"[A-Z.\-]{1,6}", compact):
        return compact.replace(".", "-")
    return compact[:6] if compact else "AMZN"


def build_x_search_query(keyword: str) -> str:
    """X search query for cashtag + ticker mentions (no retweets, English)."""
    ticker = keyword_to_ticker(keyword)
    cashtag = ticker.replace("-", "")
    return f"(${cashtag} OR #{cashtag} OR {ticker}) -is:retweet lang:en"


def _oauth1_credentials() -> dict[str, str] | None:
    s = get_settings()
    creds = {
        "consumer_key": s.x_api_key.strip(),
        "consumer_secret": s.x_api_secret.strip(),
        "access_token": s.x_access_token.strip(),
        "access_token_secret": s.x_access_token_secret.strip(),
    }
    if all(creds.values()):
        return creds
    return None


def _oauth1_ready() -> bool:
    return _oauth1_credentials() is not None


def _client_oauth1() -> tweepy.Client:
    creds = _oauth1_credentials()
    if not creds:
        raise XAuthError(
            "OAuth1 credentials incomplete. Set X_API_KEY, X_API_SECRET, "
            "X_ACCESS_TOKEN, and X_ACCESS_TOKEN_SECRET in .env"
        )
    return tweepy.Client(
        consumer_key=creds["consumer_key"],
        consumer_secret=creds["consumer_secret"],
        access_token=creds["access_token"],
        access_token_secret=creds["access_token_secret"],
        wait_on_rate_limit=True,
    )


def _client_bearer() -> tweepy.Client:
    bearer = get_settings().resolved_x_bearer_token
    if not bearer:
        raise XAuthError("X_BEARER_TOKEN is not set in .env")
    return tweepy.Client(bearer_token=bearer, wait_on_rate_limit=True)


def iter_x_clients() -> Iterator[tuple[str, tweepy.Client]]:
    """
    Clients to try, in order.

    Tweepy uses bearer auth when both bearer + OAuth1 are passed — a stale bearer
    causes 401 even when OAuth1 tokens are valid. Default order: OAuth1 first.
    """
    reload_settings()
    s = get_settings()
    mode = (getattr(s, "x_auth_mode", None) or "oauth1").strip().lower()

    if mode not in {"oauth1", "bearer", "auto"}:
        mode = "oauth1"

    order: list[str]
    if mode == "oauth1":
        order = ["oauth1"]
    elif mode == "bearer":
        order = ["bearer"]
    else:
        order = ["oauth1", "bearer"]

    seen: set[str] = set()
    for label in order:
        if label in seen:
            continue
        seen.add(label)
        if label == "oauth1" and _oauth1_ready():
            yield label, _client_oauth1()
        elif label == "bearer" and s.resolved_x_bearer_token:
            yield label, _client_bearer()


def get_x_client() -> tweepy.Client:
    clients = list(iter_x_clients())
    if not clients:
        raise XAuthError(
            "X API credentials missing. Set OAuth1 keys (X_API_KEY, X_API_SECRET, "
            "X_ACCESS_TOKEN, X_ACCESS_TOKEN_SECRET) and/or X_BEARER_TOKEN in .env"
        )
    return clients[0][1]


def _is_credits_error(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return "402" in msg or "payment required" in msg or "does not have any credits" in msg


def verify_x_auth() -> dict[str, Any]:
    """Validate credentials; returns {ok, method, user, search_ok, error}."""
    reload_settings()
    last_error = ""
    auth_ok = False
    result: dict[str, Any] = {
        "ok": False,
        "method": None,
        "username": None,
        "name": None,
        "search_ok": False,
        "search_error": None,
        "error": None,
    }

    for label, client in iter_x_clients():
        try:
            if label == "oauth1":
                me = client.get_me(user_fields=["username", "name"], user_auth=True)
                user = me.data if me and me.data else None
                result.update(
                    {
                        "ok": True,
                        "method": label,
                        "username": getattr(user, "username", None),
                        "name": getattr(user, "name", None),
                    }
                )
                auth_ok = True
            else:
                client.search_recent_tweets(
                    query="AAPL -is:retweet lang:en",
                    max_results=10,
                )
                result.update(
                    {
                        "ok": True,
                        "method": label,
                        "username": "app-only",
                        "name": "Bearer token",
                    }
                )
                auth_ok = True

            try:
                _search_with_client(
                    client,
                    "AMZN -is:retweet lang:en",
                    since_id=None,
                    max_results=10,
                    lookback_days=1,
                    user_auth=(label == "oauth1"),
                )
                result["search_ok"] = True
            except TweepyException as search_exc:
                if _is_credits_error(search_exc):
                    result["search_error"] = (
                        "X API credits required: your developer account has no credits "
                        "for tweet search. Add credits or upgrade at developer.x.com."
                    )
                else:
                    result["search_error"] = str(search_exc)
            return result

        except TypeError as exc:
            last_error = f"{label}: invalid credentials ({exc})"
            logger.warning(last_error)
        except Unauthorized as exc:
            last_error = f"{label}: 401 Unauthorized — {exc}"
            logger.warning(last_error)
        except Forbidden as exc:
            last_error = f"{label}: 403 Forbidden — {exc}"
            logger.warning(last_error)
        except TweepyException as exc:
            if _is_credits_error(exc) and auth_ok:
                result["search_error"] = str(exc)
                return result
            last_error = f"{label}: {exc}"
            logger.warning(last_error)

    result["error"] = last_error
    return result


def _sentiment_label(compound: float) -> str:
    if compound >= 0.05:
        return "Positive"
    if compound <= -0.05:
        return "Negative"
    return "Neutral"


def normalize_tweet(
    tweet_id: str,
    text: str,
    created_at: datetime | None,
    author_id: str | None,
    keyword: str,
    ticker: str,
) -> dict[str, Any]:
    scores = _vader.polarity_scores(text or "")
    compound = float(scores.get("compound", 0.0))
    ts = created_at or datetime.now(timezone.utc)
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return {
        "id": str(tweet_id),
        "text": text,
        "created_at": ts.isoformat(),
        "author_id": str(author_id or ""),
        "keyword": keyword,
        "ticker": ticker,
        "sentiment_compound": round(compound, 4),
        "sentiment": _sentiment_label(compound),
        "like_count": 0,
        "retweet_count": 0,
    }


def _search_with_client(
    client: tweepy.Client,
    query: str,
    *,
    since_id: str | None,
    max_results: int,
    lookback_days: int,
    user_auth: bool = False,
):
    kwargs: dict[str, Any] = {
        "query": query,
        "max_results": max(10, min(max_results, 100)),
        "tweet_fields": ["created_at", "author_id", "public_metrics", "lang"],
        "user_auth": user_auth,
    }
    if since_id:
        kwargs["since_id"] = since_id
    else:
        start_time = datetime.now(timezone.utc) - timedelta(days=max(1, min(lookback_days, 7)))
        kwargs["start_time"] = start_time

    return client.search_recent_tweets(**kwargs)


def fetch_recent_tweets(
    keyword: str,
    *,
    since_id: str | None = None,
    max_results: int = 100,
    lookback_days: int | None = None,
) -> list[dict[str, Any]]:
    """Pull recent tweets from X search API for the stock keyword."""
    reload_settings()
    settings = get_settings()
    ticker = keyword_to_ticker(keyword)
    query = build_x_search_query(keyword)
    days = lookback_days if lookback_days is not None else settings.x_tweet_lookback_days

    clients = list(iter_x_clients())
    if not clients:
        raise XAuthError("X API credentials are not configured in .env")

    errors: list[str] = []
    for label, client in clients:
        try:
            response = _search_with_client(
                client,
                query,
                since_id=since_id,
                max_results=max_results,
                lookback_days=days,
                user_auth=(label == "oauth1"),
            )
            tweets_out: list[dict[str, Any]] = []
            if not response or not response.data:
                logger.debug("X search (%s) returned no tweets for %s", label, query)
                return tweets_out

            for tw in response.data:
                metrics = getattr(tw, "public_metrics", None) or {}
                row = normalize_tweet(
                    tweet_id=str(tw.id),
                    text=tw.text or "",
                    created_at=getattr(tw, "created_at", None),
                    author_id=str(getattr(tw, "author_id", "") or ""),
                    keyword=keyword,
                    ticker=ticker,
                )
                row["like_count"] = int(metrics.get("like_count", 0))
                row["retweet_count"] = int(metrics.get("retweet_count", 0))
                tweets_out.append(row)

            tweets_out.sort(key=lambda t: t["created_at"])
            logger.info("X search OK via %s (%s tweets)", label, len(tweets_out))
            return tweets_out

        except Unauthorized as exc:
            errors.append(f"{label}: 401 Unauthorized")
            logger.warning("X auth failed (%s): %s", label, exc)
        except Forbidden as exc:
            errors.append(
                f"{label}: 403 — your X API plan may not include tweet search. "
                "Upgrade to Basic or higher in the X Developer Portal."
            )
            logger.warning("X forbidden (%s): %s", label, exc)
        except TooManyRequests as exc:
            raise RuntimeError("X API rate limit reached. Wait a few minutes.") from exc
        except TweepyException as exc:
            if _is_credits_error(exc):
                raise XCreditsError(
                    "Your X developer account has no API credits for tweet search. "
                    "Purchase credits or upgrade your plan at https://developer.x.com/ "
                    "to enable Social trends."
                ) from exc
            errors.append(f"{label}: {exc}")
            logger.warning("X API error (%s): %s", label, exc)

    detail = "; ".join(errors) or "Unknown authentication error"
    raise XAuthError(
        f"X API rejected all credential methods ({detail}). "
        "Regenerate keys in the X Developer Portal, set X_AUTH_MODE=oauth1 in .env, "
        "and ensure your app has Read access with tweet search enabled."
    )

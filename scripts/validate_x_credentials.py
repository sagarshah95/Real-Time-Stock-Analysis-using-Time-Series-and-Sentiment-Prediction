#!/usr/bin/env python3
"""Validate X API credentials from .env (run from project root)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.settings import reload_settings
from services.x_twitter import XCreditsError, fetch_recent_tweets, verify_x_auth


def main() -> int:
    settings = reload_settings()
    status = settings.x_credential_status()

    print("=== .env credential check ===")
    for key, val in status.items():
        print(f"  {key}: {val}")

    if not status["oauth1_complete"] and not status["bearer_set"]:
        print("\nFAIL: No complete OAuth1 set and no bearer token.")
        return 1

    print("\n=== Live X API check ===")
    auth = verify_x_auth()
    print(json.dumps({k: v for k, v in auth.items() if k != "error"}, indent=2))
    if not auth["ok"]:
        print(f"  error: {auth['error']}")
        return 1

    print("\n=== Tweet search probe (Amazon) ===")
    if not auth.get("search_ok"):
        print(f"  search_blocked: {auth.get('search_error')}")
        print("\nOK: X OAuth credentials valid.")
        print("WARN: Tweet search requires X API credits (402).")
        return 2

    try:
        tweets = fetch_recent_tweets("Amazon", max_results=10)
        print(f"  tweets_returned: {len(tweets)}")
        if tweets:
            print(f"  sample: {tweets[-1]['text'][:80]}…")
    except XCreditsError as exc:
        print(f"  search_error: {exc}")
        print("\nOK: X OAuth credentials valid.")
        return 2
    except Exception as exc:
        print(f"  search_error: {exc}")
        return 1

    print("\nOK: X credentials validated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

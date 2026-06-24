"""Finviz news scraping and VADER sentiment analysis."""

from __future__ import annotations

import logging
from typing import Any
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

logger = logging.getLogger(__name__)

_FINVIZ_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)


def _fetch_finviz_news_table(ticker: str) -> Any:
    url = f"https://finviz.com/quote.ashx?t={ticker}"
    req = Request(url, headers={"User-Agent": _FINVIZ_UA})
    with urlopen(req, timeout=25) as resp:
        soup = BeautifulSoup(resp.read(), "html.parser")
    return soup.find(id="news-table")


def get_news_sentiment(ticker: str) -> pd.DataFrame:
    """Scrape Finviz headlines and score with VADER."""
    sym = str(ticker).strip().upper()
    news_table = _fetch_finviz_news_table(sym)
    if news_table is None:
        return pd.DataFrame(columns=["ticker", "date", "time", "headline", "compound", "Sentiment"])

    parsed_news = []
    for row in news_table.findAll("tr"):
        text = row.a.get_text() if row.a else "No headline available"
        date_scrape = row.td.text.split()
        date_val, time_val = None, None
        if len(date_scrape) == 1:
            time_val = date_scrape[0]
        else:
            date_val = date_scrape[0]
            time_val = date_scrape[1]
        parsed_news.append([sym, date_val, time_val, text])

    columns = ["ticker", "date", "time", "headline"]
    df = pd.DataFrame(parsed_news, columns=columns)
    if df.empty:
        return df

    vader = SentimentIntensityAnalyzer()
    scores = df["headline"].apply(vader.polarity_scores).tolist()
    scores_df = pd.DataFrame(scores)
    df = df.join(scores_df)
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
    df["Sentiment"] = np.where(
        df["compound"] > 0,
        "Positive",
        np.where(df["compound"] == 0, "Neutral", "Negative"),
    )
    return df


def summarize_sentiment(news_df: pd.DataFrame) -> dict[str, Any]:
    """Aggregate sentiment counts and average compound score."""
    if news_df is None or news_df.empty:
        return {
            "total": 0,
            "positive": 0,
            "neutral": 0,
            "negative": 0,
            "avg_compound": 0.0,
            "overall": "Neutral",
            "headlines": [],
        }

    counts = news_df["Sentiment"].value_counts().to_dict()
    avg = float(news_df["compound"].mean()) if "compound" in news_df.columns else 0.0
    if avg > 0.05:
        overall = "Positive"
    elif avg < -0.05:
        overall = "Negative"
    else:
        overall = "Neutral"

    headlines = []
    for _, row in news_df.head(10).iterrows():
        headlines.append({
            "headline": str(row.get("headline", "")),
            "sentiment": str(row.get("Sentiment", "Neutral")),
            "date": str(row.get("date", "")),
            "compound": float(row.get("compound", 0)),
        })

    return {
        "total": len(news_df),
        "positive": int(counts.get("Positive", 0)),
        "neutral": int(counts.get("Neutral", 0)),
        "negative": int(counts.get("Negative", 0)),
        "avg_compound": round(avg, 4),
        "overall": overall,
        "headlines": headlines,
    }

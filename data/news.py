"""
data.news — News headlines and sentiment scores from Finnhub.

Requires FINNHUB_API_KEY in .env (or passed explicitly).
Returns empty DataFrames gracefully when the key is absent.
"""

from __future__ import annotations

import os
import time
import pandas as pd
from datetime import date, timedelta


def get_news(
    tickers: list[str],
    api_key: str | None = None,
    days_back: int = 7,
    max_per_ticker: int = 5,
) -> pd.DataFrame:
    """
    Fetch recent news from Finnhub for each ticker.

    Returns DataFrame: [ticker, datetime, headline, source, url, sentiment_score].
    Sorted newest-first. Empty DataFrame if key is absent or all requests fail.
    """
    key = api_key or os.getenv("FINNHUB_API_KEY", "")
    empty = pd.DataFrame(columns=["ticker", "datetime", "headline", "source", "url", "sentiment_score"])
    if not key or not tickers:
        return empty

    import finnhub
    client = finnhub.Client(api_key=key)

    end = date.today()
    start = end - timedelta(days=days_back)
    rows: list[dict] = []

    for ticker in tickers:
        try:
            articles = client.company_news(
                ticker,
                _from=start.strftime("%Y-%m-%d"),
                to=end.strftime("%Y-%m-%d"),
            )
            for a in (articles or [])[:max_per_ticker]:
                rows.append({
                    "ticker": ticker,
                    "datetime": pd.Timestamp(a["datetime"], unit="s"),
                    "headline": a.get("headline", "")[:120],
                    "source": a.get("source", ""),
                    "url": a.get("url", ""),
                    "sentiment_score": float(a.get("sentiment", 0.0)),
                })
            time.sleep(0.2)  # 60 req/min free tier guard
        except Exception:
            continue

    if not rows:
        return empty
    return pd.DataFrame(rows).sort_values("datetime", ascending=False).reset_index(drop=True)


def sentiment_summary(news_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate per-ticker sentiment from get_news() output.

    Returns DataFrame: [ticker, avg_sentiment, article_count, signal]
    signal: "BULLISH" | "NEUTRAL" | "BEARISH"
    """
    empty = pd.DataFrame(columns=["ticker", "avg_sentiment", "article_count", "signal"])
    if news_df.empty:
        return empty

    agg = (
        news_df.groupby("ticker")
        .agg(avg_sentiment=("sentiment_score", "mean"), article_count=("headline", "count"))
        .reset_index()
    )

    def _signal(score: float) -> str:
        if score > 0.1:
            return "BULLISH"
        if score < -0.1:
            return "BEARISH"
        return "NEUTRAL"

    agg["signal"] = agg["avg_sentiment"].apply(_signal)
    agg["avg_sentiment"] = agg["avg_sentiment"].round(3)
    return agg

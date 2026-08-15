"""
data.fetch — pull OHLCV price data from yfinance, with a local CSV cache layer.

Public API:
    get_prices(tickers)         → checks cache first, fetches missing ones
    fetch_ohlcv(tickers)        → always hits yfinance (use for a forced refresh)
"""

from __future__ import annotations

import yfinance as yf
import pandas as pd
from data import cache


def fetch_ohlcv(tickers: list[str], period: str = "1y", interval: str = "1d") -> dict[str, pd.DataFrame]:
    """
    Download OHLCV from yfinance for each ticker individually.

    Returns {ticker: DataFrame} with columns [Open, High, Low, Close, Volume].
    Tickers that return empty data are silently skipped.
    """
    result: dict[str, pd.DataFrame] = {}
    for ticker in tickers:
        df = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=True)
        if df.empty:
            continue
        # yfinance can return MultiIndex columns even for single tickers in newer versions
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.index = pd.to_datetime(df.index)
        result[ticker] = df
    return result


def get_prices(
    tickers: list[str],
    period: str = "1y",
    interval: str = "1d",
    refresh: bool = False,
) -> dict[str, pd.DataFrame]:
    """
    Return OHLCV data for each ticker, using the local cache where possible.

    Args:
        tickers:  e.g. ["AAPL", "SPY", "AGG"]
        period:   yfinance period string — "1y", "6mo", "2y", etc.
        interval: "1d" for daily (all Week 1-4 work uses daily)
        refresh:  set True to bypass cache and re-download everything
    """
    result: dict[str, pd.DataFrame] = {}
    to_fetch: list[str] = []

    for ticker in tickers:
        if not refresh:
            cached = cache.load(ticker)
            if cached is not None:
                result[ticker] = cached
                continue
        to_fetch.append(ticker)

    if to_fetch:
        fetched = fetch_ohlcv(to_fetch, period=period, interval=interval)
        for ticker, df in fetched.items():
            cache.save(ticker, df)
            result[ticker] = df

    return result

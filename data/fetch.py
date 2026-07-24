"""
data.fetch — pull OHLCV price data from yfinance for a watchlist of tickers.

Week 1 BUILD target.
"""

import yfinance as yf
import pandas as pd


def fetch_ohlcv(tickers: list[str], period: str = "1y", interval: str = "1d") -> dict[str, pd.DataFrame]:
    """
    Download OHLCV data for each ticker via yfinance.

    Returns a dict of {ticker: DataFrame} with columns [Open, High, Low, Close, Volume].
    """
    # TODO: implement
    raise NotImplementedError

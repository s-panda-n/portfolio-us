"""
metrics.returns — daily and cumulative return calculations.
"""

from __future__ import annotations

import pandas as pd


def daily_returns(prices: pd.Series) -> pd.Series:
    """Percentage change day-over-day from a Close price series. First row is dropped (NaN)."""
    return prices.pct_change().dropna()


def cumulative_returns(daily: pd.Series) -> pd.Series:
    """
    Compound cumulative return from a daily returns series.

    Value at each point = total return since the first observation.
    e.g. [+10%, -5%] → [0.10, 0.045]  (10% up, then net +4.5%)
    """
    return (1 + daily).cumprod() - 1

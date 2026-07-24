"""
metrics.returns — daily and cumulative return calculations.

Week 1 BUILD target.
"""

import pandas as pd


def daily_returns(prices: pd.Series) -> pd.Series:
    """Percentage change day-over-day from a Close price series."""
    # TODO: implement
    raise NotImplementedError


def cumulative_returns(daily: pd.Series) -> pd.Series:
    """Compound cumulative return from a daily returns series."""
    # TODO: implement
    raise NotImplementedError

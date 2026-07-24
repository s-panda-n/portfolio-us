"""
metrics.risk — Sharpe, Sortino, Calmar, max drawdown, and correlation matrix.

Week 1 BUILD target.
"""

import pandas as pd
import numpy as np

TRADING_DAYS = 252


def sharpe(daily_returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    """Annualised Sharpe ratio."""
    # TODO: implement
    raise NotImplementedError


def sortino(daily_returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    """Annualised Sortino ratio (penalises only downside volatility)."""
    # TODO: implement
    raise NotImplementedError


def calmar(daily_returns: pd.Series) -> float:
    """Calmar ratio: annualised return divided by max drawdown magnitude."""
    # TODO: implement
    raise NotImplementedError


def max_drawdown(daily_returns: pd.Series) -> float:
    """Max peak-to-trough drawdown as a negative fraction, e.g. -0.23 = -23%."""
    # TODO: implement
    raise NotImplementedError


def correlation_matrix(returns_df: pd.DataFrame) -> pd.DataFrame:
    """Pairwise Pearson correlation matrix for a DataFrame of daily returns (columns = tickers)."""
    # TODO: implement
    raise NotImplementedError

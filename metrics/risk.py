"""
metrics.risk — Sharpe, Sortino, Calmar, max drawdown, and correlation matrix.

All ratio functions accept a daily returns Series and annualise using 252 trading days.
"""

from __future__ import annotations

import pandas as pd
import numpy as np

TRADING_DAYS = 252


def sharpe(daily_returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    """
    Annualised Sharpe ratio.

    Excess return per unit of total volatility.
    risk_free_rate is the annual rate (e.g. 0.05 for 5%) — divided by 252 internally.
    """
    excess = daily_returns - risk_free_rate / TRADING_DAYS
    return float(excess.mean() / excess.std() * np.sqrt(TRADING_DAYS))


def sortino(daily_returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    """
    Annualised Sortino ratio.

    Like Sharpe but penalises only downside volatility — a fairer measure for
    assets with positively skewed return distributions.
    """
    excess = daily_returns - risk_free_rate / TRADING_DAYS
    downside = excess[excess < 0]
    downside_std = np.sqrt((downside ** 2).mean())
    return float(excess.mean() / downside_std * np.sqrt(TRADING_DAYS))


def max_drawdown(daily_returns: pd.Series) -> float:
    """
    Maximum peak-to-trough drawdown as a negative fraction.

    e.g. -0.23 means the portfolio fell 23% from its peak at worst.
    Starts from an implicit value of 1.0 before the first return, so the very
    first day's loss is captured correctly.
    """
    # Prepend 1.0 so we measure from before the first return
    cumulative = pd.concat([pd.Series([1.0]), (1 + daily_returns).cumprod()])
    peak = cumulative.cummax()
    drawdown = (cumulative - peak) / peak
    return float(drawdown.min())


def calmar(daily_returns: pd.Series) -> float:
    """
    Calmar ratio: annualised return divided by the absolute max drawdown.

    Higher is better. Returns inf if there is no drawdown.
    """
    n = len(daily_returns)
    annualised_return = float((1 + daily_returns).prod() ** (TRADING_DAYS / n) - 1)
    mdd = max_drawdown(daily_returns)
    if mdd == 0.0:
        return float("inf")
    return annualised_return / abs(mdd)


def correlation_matrix(returns_df: pd.DataFrame) -> pd.DataFrame:
    """
    Pairwise Pearson correlation matrix for a DataFrame of daily returns.

    Columns should be ticker symbols; each column is a daily returns Series.
    Values in [-1, 1] — diagonal is always 1.0.
    """
    return returns_df.corr()

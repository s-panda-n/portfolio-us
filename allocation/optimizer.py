"""
allocation.optimizer — mean-variance optimiser via scipy.optimize.

Public API:
    efficient_frontier(returns_df)              Monte-Carlo scatter data
    max_sharpe(returns_df, risk_free_rate)      long-only max-Sharpe weights
    min_variance(returns_df)                    long-only min-variance weights
    allocate(returns_df, capital, risk_level)   top-level entry point → allocation DataFrame
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

TRADING_DAYS = 252


def _prep(returns_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Annualised mean returns, covariance matrix, ticker list."""
    tickers = list(returns_df.columns)
    mu = returns_df.mean().values * TRADING_DAYS
    cov = returns_df.cov().values * TRADING_DAYS
    return mu, cov, tickers


def _portfolio_stats(w: np.ndarray, mu: np.ndarray, cov: np.ndarray) -> tuple[float, float]:
    ret = float(w @ mu)
    vol = float(np.sqrt(w @ cov @ w))
    return ret, vol


def _neg_sharpe(w: np.ndarray, mu: np.ndarray, cov: np.ndarray, rf: float) -> float:
    ret, vol = _portfolio_stats(w, mu, cov)
    return -(ret - rf) / vol if vol > 1e-9 else 0.0


def _port_variance(w: np.ndarray, cov: np.ndarray) -> float:
    return float(w @ cov @ w)


def _solve(objective, args: tuple, n: int) -> np.ndarray:
    res = minimize(
        objective,
        x0=np.ones(n) / n,
        args=args,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * n,
        constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0}],
        options={"maxiter": 1000, "ftol": 1e-9},
    )
    return res.x


def efficient_frontier(returns_df: pd.DataFrame, n_portfolios: int = 1500) -> pd.DataFrame:
    """
    Monte-Carlo sample of the efficient frontier.
    Returns DataFrame with columns: [return, volatility, sharpe].
    """
    mu, cov, _ = _prep(returns_df)
    n = len(returns_df.columns)
    rows = []
    for _ in range(n_portfolios):
        w = np.random.dirichlet(np.ones(n))
        ret, vol = _portfolio_stats(w, mu, cov)
        rows.append({
            "return": ret,
            "volatility": vol,
            "sharpe": ret / vol if vol > 1e-9 else 0.0,
        })
    return pd.DataFrame(rows)


def max_sharpe(returns_df: pd.DataFrame, risk_free_rate: float = 0.0) -> dict[str, float]:
    """Long-only max-Sharpe weights. Returns {ticker: weight}."""
    mu, cov, tickers = _prep(returns_df)
    w = _solve(_neg_sharpe, (mu, cov, risk_free_rate), len(tickers))
    return dict(zip(tickers, w))


def min_variance(returns_df: pd.DataFrame) -> dict[str, float]:
    """Long-only min-variance weights. Returns {ticker: weight}."""
    _, cov, tickers = _prep(returns_df)
    w = _solve(_port_variance, (cov,), len(tickers))
    return dict(zip(tickers, w))


def allocate(
    returns_df: pd.DataFrame,
    capital: float,
    risk_level: str = "MEDIUM",
) -> pd.DataFrame:
    """
    Top-level allocation entry point.

    LOW    → min_variance
    HIGH   → max_sharpe
    MEDIUM → 50/50 blend of both

    Returns DataFrame: [ticker, weight_%, dollars]
    """
    tickers = list(returns_df.columns)

    if risk_level == "LOW":
        weights = min_variance(returns_df)
    elif risk_level == "HIGH":
        weights = max_sharpe(returns_df)
    else:
        low_w = min_variance(returns_df)
        high_w = max_sharpe(returns_df)
        weights = {t: 0.5 * low_w[t] + 0.5 * high_w[t] for t in tickers}

    rows = [
        {
            "ticker": t,
            "weight_%": round(w * 100, 1),
            "dollars": int(round(w * capital)),
        }
        for t, w in sorted(weights.items(), key=lambda x: -x[1])
        if w > 0.001
    ]
    return pd.DataFrame(rows)

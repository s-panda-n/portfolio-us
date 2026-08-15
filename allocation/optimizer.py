"""
allocation.optimizer — mean-variance optimiser via scipy.optimize.

Macro signal overlay: sentiment, VIX regime, and sector momentum tilt
expected returns (mu) before solving. Min-variance (LOW risk) is unaffected
since it optimises covariance only.

Public API:
    efficient_frontier(returns_df)
    max_sharpe(returns_df, risk_free_rate)
    min_variance(returns_df)
    allocate(returns_df, capital, risk_level,
             sentiment_map, vix_regime, sector_returns, ticker_to_sector)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

TRADING_DAYS = 252


def _prep(returns_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Annualised mean returns, covariance matrix, ticker list."""
    tickers = list(returns_df.columns)
    mu  = returns_df.mean().values * TRADING_DAYS
    cov = returns_df.cov().values  * TRADING_DAYS
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


def _apply_signal_overlay(
    mu: np.ndarray,
    tickers: list[str],
    sentiment_map: dict[str, str],
    vix_regime: str,
    sector_returns: dict[str, float],
    ticker_to_sector: dict[str, str],
) -> np.ndarray:
    """
    Tilt annualised expected returns before optimisation.

    Adjustments (both scaled 50% in high-VIX regimes where signals are noisier):
      Sentiment:        BULLISH +2% ann. | BEARISH -2% ann.
      Sector momentum:  top-third sector +1% ann. | bottom-third -1% ann.

    Only applied when risk_level != LOW (min-variance uses cov only).
    """
    scale = 0.5 if vix_regime == "high" else 1.0
    adj   = np.zeros(len(tickers))

    for i, t in enumerate(tickers):
        sig = sentiment_map.get(t, "")
        if sig == "BULLISH":
            adj[i] += 0.02 * scale
        elif sig == "BEARISH":
            adj[i] -= 0.02 * scale

    if sector_returns:
        vals = sorted(sector_returns.values())
        n = len(vals)
        if n >= 3:
            top_cut    = vals[int(n * 2 / 3)]
            bottom_cut = vals[int(n * 1 / 3)]
            for i, t in enumerate(tickers):
                etf = ticker_to_sector.get(t)
                if etf and etf in sector_returns:
                    r = sector_returns[etf]
                    if r >= top_cut:
                        adj[i] += 0.01 * scale
                    elif r <= bottom_cut:
                        adj[i] -= 0.01 * scale

    return mu + adj


def efficient_frontier(returns_df: pd.DataFrame, n_portfolios: int = 1500) -> pd.DataFrame:
    """Monte-Carlo sample of the efficient frontier."""
    mu, cov, _ = _prep(returns_df)
    n = len(returns_df.columns)
    rows = []
    for _ in range(n_portfolios):
        w = np.random.dirichlet(np.ones(n))
        ret, vol = _portfolio_stats(w, mu, cov)
        rows.append({"return": ret, "volatility": vol,
                     "sharpe": ret / vol if vol > 1e-9 else 0.0})
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
    sentiment_map: dict[str, str] | None = None,
    vix_regime: str = "low",
    sector_returns: dict[str, float] | None = None,
    ticker_to_sector: dict[str, str] | None = None,
) -> pd.DataFrame:
    """
    Top-level allocation entry point.

    LOW    → min_variance  (covariance-only; signal overlay not applied)
    HIGH   → max_sharpe    (signal-adjusted expected returns)
    MEDIUM → 50/50 blend of both

    Signal overlays applied to mu before solving (HIGH and MEDIUM only):
      sentiment_map:   {ticker: 'BULLISH' | 'BEARISH'} from 30-day Finnhub news
      vix_regime:      'low' | 'elevated' | 'high' — scales signal confidence
      sector_returns:  {sector_etf: 1m_return_%} for sector momentum tilt
      ticker_to_sector:{ticker: sector_etf} mapping for sector tilt
    """
    mu, cov, tickers = _prep(returns_df)
    n  = len(tickers)
    rf = 0.0

    if risk_level != "LOW" and (sentiment_map or sector_returns):
        mu = _apply_signal_overlay(
            mu, tickers,
            sentiment_map or {},
            vix_regime,
            sector_returns or {},
            ticker_to_sector or {},
        )

    if risk_level == "LOW":
        w = _solve(_port_variance, (cov,), n)
    elif risk_level == "HIGH":
        w = _solve(_neg_sharpe, (mu, cov, rf), n)
    else:
        w_low  = _solve(_port_variance, (cov,), n)
        w_high = _solve(_neg_sharpe, (mu, cov, rf), n)
        w = 0.5 * w_low + 0.5 * w_high

    weights = dict(zip(tickers, w))
    rows = [
        {
            "ticker":   t,
            "weight_%": round(wt * 100, 1),
            "dollars":  int(round(wt * capital)),
        }
        for t, wt in sorted(weights.items(), key=lambda x: -x[1])
        if wt > 0.001
    ]
    return pd.DataFrame(rows)

"""
agents.orchestrator — full pipeline: fetch → metrics → allocate → news → reason.

Entry point: run_pipeline(tickers, capital, risk_level, ...)
Returns a single dict consumed by app.py — one call per dashboard refresh.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from data.fetch import get_prices
from data.bonds import get_bond_etf_prices, get_yield_curve, BOND_ETFS
from data.news import get_news, sentiment_summary
from metrics.returns import daily_returns, cumulative_returns
from metrics.risk import sharpe, sortino, calmar, max_drawdown, correlation_matrix
from allocation.optimizer import allocate, efficient_frontier

TRADING_DAYS = 252
DEFAULT_EQUITIES = ["AAPL", "MSFT", "GOOGL", "NVDA", "SPY", "QQQ"]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _close(prices: dict[str, pd.DataFrame]) -> dict[str, pd.Series]:
    result = {}
    for ticker, df in prices.items():
        col = "Close" if "Close" in df.columns else df.columns[0]
        result[ticker] = df[col].dropna()
    return result


def _ticker_metrics(r: pd.Series) -> dict:
    n = len(r)
    return {
        "ann_return_%": round(float((1 + r).prod() ** (TRADING_DAYS / n) - 1) * 100, 2),
        "ann_vol_%":    round(float(r.std() * np.sqrt(TRADING_DAYS)) * 100, 2),
        "sharpe":       round(sharpe(r), 2),
        "sortino":      round(sortino(r), 2),
        "max_dd_%":     round(max_drawdown(r) * 100, 2),
    }


def _weighted_port_returns(returns_df: pd.DataFrame, weights: dict[str, float]) -> pd.Series:
    """Daily portfolio return stream given a weight dict."""
    cols = [t for t in weights if t in returns_df.columns]
    w = np.array([weights[t] for t in cols])
    w = w / w.sum()
    return returns_df[cols].dot(w)


# ── Pipeline ──────────────────────────────────────────────────────────────────

def run_pipeline(
    tickers: list[str],
    capital: float,
    risk_level: str,
    finnhub_key: str | None = None,
    fred_key: str | None = None,
    refresh: bool = False,
) -> dict:
    """
    Full data → metrics → allocate → news → reasoning pipeline.

    Returns dict with keys:
        allocation      DataFrame [ticker, weight_%, dollars]
        metrics         DataFrame per-ticker risk/return
        frontier        DataFrame [return, volatility, sharpe]
        corr_matrix     DataFrame pairwise Pearson
        cum_returns     dict {ticker: pd.Series of cumulative returns}
        optimal         dict {return, volatility, sharpe, sortino, max_drawdown}
        yield_curve     DataFrame [maturity, yield_pct] or empty
        news            DataFrame headlines or empty
        sentiment       DataFrame per-ticker sentiment summary or empty
        last_prices     dict {ticker: float}
        pct_changes     dict {ticker: float}
        reasoning       list[str]
        errors          list[str] non-fatal warnings to surface in UI
    """
    errors: list[str] = []
    reasoning: list[str] = []

    # ── 1. Fetch prices (equities + bond ETFs) ────────────────────────────────
    all_tickers = list(dict.fromkeys(tickers + BOND_ETFS))
    try:
        prices = get_prices(all_tickers, refresh=refresh)
    except Exception as e:
        errors.append(f"Price fetch failed: {e}")
        prices = {}

    close_map = _close(prices)
    if not close_map:
        raise ValueError("No price data available. Check ticker symbols and network.")

    # ── 2. Build aligned daily returns DataFrame ──────────────────────────────
    returns_raw = {t: daily_returns(s) for t, s in close_map.items() if len(s) > 5}
    if not returns_raw:
        raise ValueError("Insufficient price history to compute returns.")

    returns_df = pd.DataFrame(returns_raw).dropna()
    available = list(returns_df.columns)
    reasoning.append(f"> LOADED: {len(available)} assets — {', '.join(available)}")

    # ── 3. Allocation ─────────────────────────────────────────────────────────
    try:
        alloc_df = allocate(returns_df, capital, risk_level)
        opt_weights = dict(zip(alloc_df["ticker"], alloc_df["weight_%"] / 100))
    except Exception as e:
        errors.append(f"Optimizer error (falling back to equal weight): {e}")
        n = len(available)
        alloc_df = pd.DataFrame({
            "ticker": available,
            "weight_%": [round(100 / n, 1)] * n,
            "dollars": [int(round(capital / n))] * n,
        })
        opt_weights = {t: 1 / n for t in available}

    optimizer_label = {
        "LOW": "MIN-VARIANCE", "HIGH": "MAX-SHARPE", "MEDIUM": "BLENDED (50/50)",
    }.get(risk_level, risk_level)
    reasoning.append(f"> RISK: {risk_level}  OPTIMIZER: {optimizer_label}")
    for _, row in alloc_df.iterrows():
        reasoning.append(
            f">   {row['ticker']:<8}  {row['weight_%']:>5.1f}%   ${row['dollars']:>10,.0f}"
        )

    # ── 4. Efficient frontier ─────────────────────────────────────────────────
    try:
        frontier_df = efficient_frontier(returns_df)
    except Exception as e:
        errors.append(f"Frontier error: {e}")
        frontier_df = pd.DataFrame(columns=["return", "volatility", "sharpe"])

    # ── 5. Per-asset metrics ──────────────────────────────────────────────────
    metrics_df = pd.DataFrame(
        {t: _ticker_metrics(returns_df[t]) for t in available}
    ).T

    # ── 6. Portfolio-level metrics ────────────────────────────────────────────
    port_r = _weighted_port_returns(returns_df, opt_weights)
    n_days = len(port_r)
    optimal = {
        "return":       float((1 + port_r).prod() ** (TRADING_DAYS / n_days) - 1),
        "volatility":   float(port_r.std() * np.sqrt(TRADING_DAYS)),
        "sharpe":       sharpe(port_r),
        "sortino":      sortino(port_r),
        "max_drawdown": max_drawdown(port_r),
    }
    reasoning.append(
        f"> PORTFOLIO:  return={optimal['return']:.1%}  "
        f"vol={optimal['volatility']:.1%}  "
        f"Sharpe={optimal['sharpe']:.2f}  "
        f"MDD={optimal['max_drawdown']:.1%}"
    )

    # ── 7. Correlation matrix ─────────────────────────────────────────────────
    corr = correlation_matrix(returns_df)

    # ── 8. Cumulative returns ─────────────────────────────────────────────────
    cum_returns: dict[str, pd.Series] = {t: cumulative_returns(returns_df[t]) for t in available}
    cum_returns["PORTFOLIO"] = cumulative_returns(port_r)

    # ── 9. Last prices + 1-day pct change for ticker tape ────────────────────
    last_prices = {t: float(s.iloc[-1]) for t, s in close_map.items() if t in available}
    pct_changes = {
        t: round((float(s.iloc[-1]) - float(s.iloc[-2])) / float(s.iloc[-2]) * 100, 2)
        for t, s in close_map.items()
        if t in available and len(s) >= 2
    }

    # ── 10. Yield curve ───────────────────────────────────────────────────────
    try:
        yield_df = get_yield_curve(api_key=fred_key)
    except Exception as e:
        errors.append(f"FRED yield curve error: {e}")
        yield_df = pd.DataFrame(columns=["maturity", "yield_pct"])

    # ── 11. News & sentiment ──────────────────────────────────────────────────
    equity_tickers = [t for t in alloc_df["ticker"] if t not in BOND_ETFS][:4]
    try:
        news_df = get_news(equity_tickers, api_key=finnhub_key)
        sentiment_df = sentiment_summary(news_df)
    except Exception as e:
        errors.append(f"News fetch error: {e}")
        news_df = pd.DataFrame()
        sentiment_df = pd.DataFrame()

    return {
        "allocation":   alloc_df,
        "metrics":      metrics_df,
        "frontier":     frontier_df,
        "corr_matrix":  corr,
        "cum_returns":  cum_returns,
        "optimal":      optimal,
        "yield_curve":  yield_df,
        "news":         news_df,
        "sentiment":    sentiment_df,
        "last_prices":  last_prices,
        "pct_changes":  pct_changes,
        "reasoning":    reasoning,
        "errors":       errors,
    }

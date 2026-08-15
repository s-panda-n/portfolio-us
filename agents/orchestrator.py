"""
agents.orchestrator — full pipeline: prices → signals → sentiment → allocate → metrics.

Entry point: run_pipeline(tickers, capital, risk_level, ...)
Returns a single dict consumed by app.py — one call per dashboard refresh.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from data.fetch import get_prices
from data.bonds import get_yield_curve, BOND_ETFS
from data.news import get_news, sentiment_summary
from data.macro import (
    get_vix, get_sector_momentum, get_earnings_calendar, get_macro_news,
    TICKER_TO_SECTOR_ETF,
)
from data.screener import run_screener
from metrics.returns import daily_returns, cumulative_returns
from metrics.risk import sharpe, sortino, max_drawdown, correlation_matrix
from allocation.optimizer import allocate, efficient_frontier

TRADING_DAYS = 252

# Default universe: six asset classes so the optimizer has real diversification.
DEFAULT_EQUITIES = [
    # US mega-cap tech / growth
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META",
    # Broad US market ETFs
    "SPY", "QQQ",
    # International developed markets
    "EFA",
    # Emerging markets
    "EEM",
    # Real estate (REITs)
    "VNQ",
    # Gold
    "GLD",
    # Commodities basket
    "GSG",
    # Financials / value
    "JPM",
]


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
    Full pipeline. Signal order matters — news/macro fetched before allocation
    so sentiment and sector momentum can tilt expected returns.

    Returns dict with keys:
        allocation, metrics, frontier, corr_matrix, cum_returns, optimal,
        yield_curve, news, sentiment, macro_news, vix, sector_momentum,
        earnings_calendar, last_prices, pct_changes, reasoning, errors
    """
    errors: list[str] = []
    reasoning: list[str] = []

    # ── 1. Prices — 3-year window ─────────────────────────────────────────────
    all_tickers = list(dict.fromkeys(tickers + BOND_ETFS))
    try:
        prices = get_prices(all_tickers, period="3y", refresh=refresh)
    except Exception as e:
        errors.append(f"Price fetch failed: {e}")
        prices = {}

    close_map = _close(prices)
    if not close_map:
        raise ValueError("No price data available. Check ticker symbols and network.")

    # ── 2. Daily returns ──────────────────────────────────────────────────────
    returns_raw = {t: daily_returns(s) for t, s in close_map.items() if len(s) > 5}
    if not returns_raw:
        raise ValueError("Insufficient price history to compute returns.")
    returns_df = pd.DataFrame(returns_raw).dropna()
    available  = list(returns_df.columns)
    reasoning.append(f"> LOADED: {len(available)} assets — {', '.join(available)}")
    reasoning.append(f"> LOOKBACK: 3 years  ({len(returns_df)} trading days)")

    # ── 3. VIX & sector momentum ──────────────────────────────────────────────
    try:
        vix_info = get_vix()
    except Exception as e:
        errors.append(f"VIX fetch error: {e}")
        vix_info = {}

    try:
        sector_df = get_sector_momentum()
    except Exception as e:
        errors.append(f"Sector momentum error: {e}")
        sector_df = pd.DataFrame(columns=["ticker", "sector", "return_1m_%"])

    sector_returns: dict[str, float] = (
        dict(zip(sector_df["ticker"], sector_df["return_1m_%"]))
        if not sector_df.empty else {}
    )
    vix_regime = vix_info.get("regime", "low")

    if vix_info:
        reasoning.append(
            f"> VIX: {vix_info['level']:.1f} ({vix_info['regime'].upper()}) — "
            f"trend {vix_info['trend']}, was {vix_info['month_ago']:.1f} a month ago"
        )
    if sector_returns:
        top_sector = max(sector_returns, key=sector_returns.get)
        bot_sector = min(sector_returns, key=sector_returns.get)
        reasoning.append(
            f"> SECTOR: leading={top_sector} ({sector_returns[top_sector]:+.1f}%)  "
            f"lagging={bot_sector} ({sector_returns[bot_sector]:+.1f}%)"
        )

    # ── 4. News & sentiment — 30-day window, all equity tickers ──────────────
    equity_tickers = [t for t in tickers if t not in BOND_ETFS]
    try:
        news_df      = get_news(equity_tickers, api_key=finnhub_key, days_back=30)
        sentiment_df = sentiment_summary(news_df)
    except Exception as e:
        errors.append(f"News fetch error: {e}")
        news_df      = pd.DataFrame()
        sentiment_df = pd.DataFrame()

    sentiment_map: dict[str, str] = {}
    if sentiment_df is not None and not sentiment_df.empty and "signal" in sentiment_df.columns:
        sentiment_map = dict(zip(sentiment_df["ticker"], sentiment_df["signal"]))
        for t, sig in sentiment_map.items():
            if sig in ("BULLISH", "BEARISH"):
                reasoning.append(f"> SENTIMENT: {t} → {sig}")

    # ── 5. Macro news ─────────────────────────────────────────────────────────
    try:
        macro_news_df = get_macro_news(finnhub_key or "", days_back=14)
    except Exception as e:
        errors.append(f"Macro news error: {e}")
        macro_news_df = pd.DataFrame()

    # ── 6. Allocation (signal-adjusted expected returns) ─────────────────────
    signals_active = bool(sentiment_map or sector_returns) and risk_level != "LOW"
    try:
        alloc_df = allocate(
            returns_df, capital, risk_level,
            sentiment_map=sentiment_map,
            vix_regime=vix_regime,
            sector_returns=sector_returns,
            ticker_to_sector=TICKER_TO_SECTOR_ETF,
        )
        opt_weights = dict(zip(alloc_df["ticker"], alloc_df["weight_%"] / 100))
    except Exception as e:
        errors.append(f"Optimizer error (falling back to equal weight): {e}")
        n = len(available)
        alloc_df = pd.DataFrame({
            "ticker":   available,
            "weight_%": [round(100 / n, 1)] * n,
            "dollars":  [int(round(capital / n))] * n,
        })
        opt_weights = {t: 1 / n for t in available}

    optimizer_label = {
        "LOW": "MIN-VARIANCE", "HIGH": "MAX-SHARPE", "MEDIUM": "BLENDED (50/50)",
    }.get(risk_level, risk_level)
    reasoning.append(
        f"> RISK: {risk_level}  OPTIMIZER: {optimizer_label}"
        + ("  + SIGNAL OVERLAY (sentiment + sector momentum)" if signals_active else "")
    )
    for _, row in alloc_df.iterrows():
        reasoning.append(
            f">   {row['ticker']:<8}  {row['weight_%']:>5.1f}%   ${row['dollars']:>10,.0f}"
        )

    # ── 7. Efficient frontier ─────────────────────────────────────────────────
    try:
        frontier_df = efficient_frontier(returns_df)
    except Exception as e:
        errors.append(f"Frontier error: {e}")
        frontier_df = pd.DataFrame(columns=["return", "volatility", "sharpe"])

    # ── 8. Per-asset metrics ──────────────────────────────────────────────────
    metrics_df = pd.DataFrame(
        {t: _ticker_metrics(returns_df[t]) for t in available}
    ).T

    # ── 9. Portfolio-level metrics ────────────────────────────────────────────
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

    # ── 10. Correlation matrix ────────────────────────────────────────────────
    corr = correlation_matrix(returns_df)

    # ── 11. Cumulative returns ────────────────────────────────────────────────
    cum_returns: dict[str, pd.Series] = {
        t: cumulative_returns(returns_df[t]) for t in available
    }
    cum_returns["PORTFOLIO"] = cumulative_returns(port_r)

    # ── 12. Last prices + 1-day pct change ───────────────────────────────────
    last_prices = {t: float(s.iloc[-1]) for t, s in close_map.items() if t in available}
    pct_changes = {
        t: round((float(s.iloc[-1]) - float(s.iloc[-2])) / float(s.iloc[-2]) * 100, 2)
        for t, s in close_map.items()
        if t in available and len(s) >= 2
    }

    # ── 13. Yield curve ───────────────────────────────────────────────────────
    try:
        yield_df = get_yield_curve(api_key=fred_key)
    except Exception as e:
        errors.append(f"FRED yield curve error: {e}")
        yield_df = pd.DataFrame(columns=["maturity", "yield_pct"])

    # ── 14. Earnings calendar ─────────────────────────────────────────────────
    try:
        earnings_df = get_earnings_calendar(equity_tickers)
    except Exception as e:
        errors.append(f"Earnings calendar error: {e}")
        earnings_df = pd.DataFrame(columns=["ticker", "earnings_date", "days_until"])

    # ── 15. Broad stock screener (55+ tickers, discovery layer) ──────────────
    try:
        screener_df = run_screener(
            period="1y",
            top_n=20,
            sector_returns=sector_returns,
            sentiment_map=sentiment_map,
        )
    except Exception as e:
        errors.append(f"Screener error: {e}")
        screener_df = pd.DataFrame()

    return {
        "allocation":        alloc_df,
        "metrics":           metrics_df,
        "frontier":          frontier_df,
        "corr_matrix":       corr,
        "cum_returns":       cum_returns,
        "optimal":           optimal,
        "yield_curve":       yield_df,
        "news":              news_df,
        "sentiment":         sentiment_df,
        "macro_news":        macro_news_df,
        "vix":               vix_info,
        "sector_momentum":   sector_df,
        "earnings_calendar": earnings_df,
        "screener":          screener_df,
        "last_prices":       last_prices,
        "pct_changes":       pct_changes,
        "reasoning":         reasoning,
        "errors":            errors,
    }

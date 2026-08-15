"""
data.portfolio — Manual holdings tracker.

Computes current value, P&L, and historical portfolio value from a list of
manually entered holdings. No database — caller stores holdings in session_state.
"""

from __future__ import annotations

import pandas as pd
from data.fetch import get_prices


_EMPTY_POSITIONS = pd.DataFrame(columns=[
    "ticker", "shares", "avg_cost", "last_price",
    "current_value", "cost_basis", "pnl_$", "pnl_%",
])


def compute_portfolio_value(holdings: list[dict]) -> dict:
    """
    From a list of holdings dicts compute current value, P&L, and history.

    Each holding: {ticker, date (YYYY-MM-DD), shares, cost_basis (per share)}

    Returns dict with keys:
        total_value     float
        total_cost      float
        total_pnl       float
        total_pnl_pct   float
        positions       DataFrame
        value_history   pd.Series (date-indexed, daily total portfolio value)
        weights         dict {ticker: float}  current portfolio weights
    """
    if not holdings:
        return {
            "total_value": 0.0, "total_cost": 0.0,
            "total_pnl": 0.0, "total_pnl_pct": 0.0,
            "positions": _EMPTY_POSITIONS.copy(),
            "value_history": pd.Series(dtype=float),
            "weights": {},
        }

    tickers = list({h["ticker"] for h in holdings})
    prices = get_prices(tickers, period="2y")

    # Per-holding breakdown
    rows: list[dict] = []
    for h in holdings:
        ticker      = h["ticker"]
        shares      = float(h["shares"])
        cost_ps     = float(h["cost_basis"])
        df          = prices.get(ticker)
        last_price  = float(df["Close"].iloc[-1]) if df is not None and not df.empty else cost_ps

        current_val = shares * last_price
        cost_total  = shares * cost_ps
        pnl         = current_val - cost_total
        pnl_pct     = (pnl / cost_total * 100) if cost_total > 0 else 0.0

        rows.append({
            "ticker":        ticker,
            "shares":        shares,
            "avg_cost":      round(cost_ps, 2),
            "last_price":    round(last_price, 2),
            "current_value": round(current_val, 2),
            "cost_basis":    round(cost_total, 2),
            "pnl_$":         round(pnl, 2),
            "pnl_%":         round(pnl_pct, 2),
        })

    positions_df   = pd.DataFrame(rows)
    total_value    = positions_df["current_value"].sum()
    total_cost     = positions_df["cost_basis"].sum()
    total_pnl      = total_value - total_cost
    total_pnl_pct  = (total_pnl / total_cost * 100) if total_cost > 0 else 0.0

    # Daily portfolio value history: sum of (shares × daily close) per holding,
    # starting from each holding's purchase date
    daily_cols: dict[str, pd.Series] = {}
    for i, h in enumerate(holdings):
        ticker   = h["ticker"]
        shares   = float(h["shares"])
        buy_date = pd.Timestamp(h["date"])
        df       = prices.get(ticker)
        if df is None or df.empty or "Close" not in df.columns:
            continue
        close = df["Close"]
        close = close[close.index >= buy_date]
        daily_cols[f"{ticker}_{i}"] = close * shares

    if daily_cols:
        value_history = pd.DataFrame(daily_cols).ffill().fillna(0).sum(axis=1)
        earliest = min(pd.Timestamp(h["date"]) for h in holdings)
        value_history = value_history[value_history.index >= earliest]
    else:
        value_history = pd.Series(dtype=float)

    weights = {
        row["ticker"]: row["current_value"] / total_value
        for row in rows if total_value > 0
    }

    return {
        "total_value":   round(total_value, 2),
        "total_cost":    round(total_cost, 2),
        "total_pnl":     round(total_pnl, 2),
        "total_pnl_pct": round(total_pnl_pct, 2),
        "positions":     positions_df,
        "value_history": value_history,
        "weights":       weights,
    }

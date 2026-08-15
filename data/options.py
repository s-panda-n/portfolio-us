"""
data.options — Fetch live options chain data from Alpaca Market Data API.

Requires ALPACA_API_KEY + ALPACA_SECRET_KEY in .env.
Returns empty DataFrames gracefully when credentials are absent,
data is unavailable, or the subscription doesn't include OPRA.
"""

from __future__ import annotations

import os
import pandas as pd
from datetime import datetime, timedelta


def _client():
    from alpaca.data.historical import OptionHistoricalDataClient
    key    = os.getenv("ALPACA_API_KEY", "")
    secret = os.getenv("ALPACA_SECRET_KEY", "")
    if not key or not secret:
        return None
    return OptionHistoricalDataClient(api_key=key, secret_key=secret)


_EMPTY_COLS = [
    "contract", "type", "strike", "expiry",
    "bid", "ask", "mid", "iv", "delta", "gamma", "theta", "vega",
]


def get_options_chain(
    symbol: str,
    max_expiry_days: int = 45,
) -> pd.DataFrame:
    """
    Fetch ATM options for the nearest expiry within max_expiry_days.

    Returns DataFrame with columns in _EMPTY_COLS.
    Returns an empty DataFrame on any error (missing keys, no data, OPRA restriction).
    """
    empty = pd.DataFrame(columns=_EMPTY_COLS)
    client = _client()
    if client is None:
        return empty

    try:
        from alpaca.data.requests import OptionSnapshotRequest

        today    = datetime.today().date()
        exp_end  = (today + timedelta(days=max_expiry_days)).isoformat()
        exp_start = today.isoformat()

        req = OptionSnapshotRequest(
            symbol_or_symbols=symbol,
            expiration_date_gte=exp_start,
            expiration_date_lte=exp_end,
        )
        snapshots = client.get_option_snapshot(req)

    except Exception:
        return empty

    if not snapshots:
        return empty

    rows: list[dict] = []
    for contract_sym, snap in snapshots.items():
        try:
            details = getattr(snap, "details", None) or getattr(snap, "greeks", None)
            quote   = getattr(snap, "latest_quote", None)
            gk      = getattr(snap, "greeks", None)
            undl    = getattr(snap, "latest_trade", None) or getattr(snap, "underlying_asset", None)

            strike  = getattr(getattr(snap, "details", None), "strike_price", None)
            expiry  = getattr(getattr(snap, "details", None), "expiration_date", None)
            opt_type = "call" if contract_sym[-9] == "C" else "put"

            bid = getattr(quote, "bid_price", None)
            ask = getattr(quote, "ask_price", None)

            rows.append({
                "contract": contract_sym,
                "type":     opt_type,
                "strike":   strike,
                "expiry":   expiry,
                "bid":      bid,
                "ask":      ask,
                "mid":      round((bid + ask) / 2, 2) if bid is not None and ask is not None else None,
                "iv":       round(getattr(gk, "implied_volatility", None) or 0, 4) or None,
                "delta":    round(getattr(gk, "delta", None) or 0, 4) or None,
                "gamma":    round(getattr(gk, "gamma", None) or 0, 6) or None,
                "theta":    round(getattr(gk, "theta", None) or 0, 4) or None,
                "vega":     round(getattr(gk, "vega", None) or 0, 4) or None,
            })
        except Exception:
            continue

    if not rows:
        return empty

    df = pd.DataFrame(rows).dropna(subset=["strike", "expiry"])
    return df.sort_values(["expiry", "strike"]).reset_index(drop=True)


def get_atm_options(
    symbol: str,
    current_price: float,
    max_expiry_days: int = 45,
    n_strikes: int = 3,
) -> pd.DataFrame:
    """
    Filter a full options chain down to the N strikes nearest to current_price
    for calls and puts separately — the most useful view for a position holder.
    """
    df = get_options_chain(symbol, max_expiry_days)
    if df.empty or "strike" not in df.columns:
        return df

    df["distance"] = (df["strike"] - current_price).abs()
    nearest = df.nsmallest(n_strikes * 2, "distance")["strike"].unique()[:n_strikes]

    return df[df["strike"].isin(nearest)].reset_index(drop=True)

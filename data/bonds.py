"""
data.bonds — Bond data via yfinance ETF proxies and FRED Treasury yields.

Bond ETF proxies (pulled through the existing cache layer):
    AGG  iShares Core US Aggregate Bond ETF  (broad market)
    TLT  iShares 20+ Year Treasury Bond ETF  (long duration)
    SHY  iShares 1-3 Year Treasury Bond ETF  (short duration)

FRED Treasury yield curve (requires FRED_API_KEY in .env):
    Tenors: 3M, 2Y, 5Y, 10Y, 30Y  — units are percent (4.32 means 4.32%).
"""

from __future__ import annotations

import os
import requests
import pandas as pd
from data.fetch import get_prices

BOND_ETFS = ["AGG", "TLT", "SHY"]

_FRED_SERIES = {
    "3M":  "DGS3MO",
    "2Y":  "DGS2",
    "5Y":  "DGS5",
    "10Y": "DGS10",
    "30Y": "DGS30",
}
_FRED_URL = "https://api.stlouisfed.org/fred/series/observations"
_TENOR_ORDER = list(_FRED_SERIES.keys())


def get_bond_etf_prices(period: str = "1y", refresh: bool = False) -> dict[str, pd.DataFrame]:
    """Fetch OHLCV for AGG, TLT, SHY via the existing yfinance cache layer."""
    return get_prices(BOND_ETFS, period=period, refresh=refresh)


def get_yield_curve(api_key: str | None = None) -> pd.DataFrame:
    """
    Fetch current Treasury yields from FRED.

    Returns DataFrame columns [maturity, yield_pct], ordered from short to long tenor.
    Returns an empty DataFrame if FRED_API_KEY is not set or FRED is unreachable.
    """
    key = api_key or os.getenv("FRED_API_KEY")
    if not key:
        return pd.DataFrame(columns=["maturity", "yield_pct"])

    rows: list[dict] = []
    for tenor, series_id in _FRED_SERIES.items():
        try:
            resp = requests.get(
                _FRED_URL,
                params={
                    "series_id": series_id,
                    "api_key": key,
                    "file_type": "json",
                    "sort_order": "desc",
                    "limit": 5,  # a few in case the latest observation is "."
                },
                timeout=10,
            )
            resp.raise_for_status()
            obs = [o for o in resp.json().get("observations", []) if o["value"] != "."]
            if obs:
                rows.append({"maturity": tenor, "yield_pct": float(obs[0]["value"])})
        except Exception:
            continue

    if not rows:
        return pd.DataFrame(columns=["maturity", "yield_pct"])

    df = pd.DataFrame(rows)
    df["_ord"] = df["maturity"].map({m: i for i, m in enumerate(_TENOR_ORDER)})
    return df.sort_values("_ord").drop(columns="_ord").reset_index(drop=True)

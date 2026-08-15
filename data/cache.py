"""
data.cache — local CSV cache so we don't hammer yfinance on every run.

Stores one file per ticker under data/.cache/<TICKER>.csv.
"""

from __future__ import annotations

import pandas as pd
from pathlib import Path

CACHE_DIR = Path(__file__).parent / ".cache"


def load(ticker: str) -> pd.DataFrame | None:
    """Return cached DataFrame for ticker, or None if not cached yet."""
    path = CACHE_DIR / f"{ticker}.csv"
    if not path.exists():
        return None
    return pd.read_csv(path, index_col=0, parse_dates=True)


def save(ticker: str, df: pd.DataFrame) -> None:
    """Write DataFrame to cache. Silent no-op on read-only filesystems (e.g. Streamlit Cloud)."""
    try:
        CACHE_DIR.mkdir(exist_ok=True)
        df.to_csv(CACHE_DIR / f"{ticker}.csv")
    except OSError:
        pass

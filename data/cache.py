"""
data.cache — local CSV cache so we don't hammer yfinance on every run.

Stores files under data/.cache/<ticker>.csv.
Week 1 BUILD target.
"""

import pandas as pd
from pathlib import Path

CACHE_DIR = Path(__file__).parent / ".cache"


def load(ticker: str) -> pd.DataFrame | None:
    """Return cached DataFrame for ticker, or None if not cached."""
    # TODO: implement
    raise NotImplementedError


def save(ticker: str, df: pd.DataFrame) -> None:
    """Write DataFrame to cache."""
    # TODO: implement
    raise NotImplementedError

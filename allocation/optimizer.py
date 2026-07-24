"""
allocation.optimizer — mean-variance optimizer via scipy.

Week 2 BUILD target.
"""

import numpy as np
import pandas as pd


def efficient_frontier(returns_df: pd.DataFrame, n_portfolios: int = 500) -> pd.DataFrame:
    """
    Monte-Carlo sample of the efficient frontier.

    Returns DataFrame with columns [weights, return, volatility, sharpe].
    Week 2 — not yet implemented.
    """
    # TODO: implement
    raise NotImplementedError

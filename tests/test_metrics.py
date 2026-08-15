"""
Tests for metrics.returns and metrics.risk.

All tests use synthetic data with known closed-form answers so failures are obvious.
"""

import pandas as pd
import numpy as np

from metrics.returns import daily_returns, cumulative_returns
from metrics.risk import sharpe, sortino, calmar, max_drawdown, correlation_matrix


# ---------- returns ----------

def test_daily_returns_values():
    prices = pd.Series([100.0, 110.0, 99.0])
    r = daily_returns(prices)
    assert len(r) == 2
    assert abs(r.iloc[0] - 0.10) < 1e-9          # +10%
    assert abs(r.iloc[1] - (-11 / 110)) < 1e-9   # -10% of 110


def test_daily_returns_no_leading_nan():
    prices = pd.Series([50.0, 55.0, 52.0])
    r = daily_returns(prices)
    assert not r.isna().any()


def test_cumulative_returns_compounding():
    # +10% then -10%: 1.1 * 0.9 - 1 = -0.01
    r = cumulative_returns(pd.Series([0.10, -0.10]))
    assert abs(r.iloc[-1] - (-0.01)) < 1e-9


def test_cumulative_returns_monotone_up():
    r = cumulative_returns(pd.Series([0.01] * 10))
    assert (r.diff().dropna() > 0).all()


# ---------- max_drawdown ----------

def test_max_drawdown_declining_series():
    # Prices 100 → 90 → 80: peak-to-trough = -20%
    prices = pd.Series([100.0, 90.0, 80.0])
    dr = daily_returns(prices)
    assert abs(max_drawdown(dr) - (-0.20)) < 1e-9


def test_max_drawdown_always_rising():
    dr = pd.Series([0.01] * 100)
    assert max_drawdown(dr) == 0.0


def test_max_drawdown_is_negative():
    dr = pd.Series([0.05, -0.15, 0.02, -0.08])
    assert max_drawdown(dr) < 0


# ---------- sharpe ----------

def test_sharpe_positive_drift():
    # Consistent small gains → positive Sharpe
    dr = pd.Series([0.001] * 252)
    assert sharpe(dr) > 0


def test_sharpe_zero_std_is_nan():
    # All returns identical → std = 0 → NaN
    dr = pd.Series([0.0] * 252)
    assert np.isnan(sharpe(dr))


# ---------- sortino ----------

def test_sortino_greater_than_sharpe_when_positive_skew():
    # Positive skew: mostly small gains, a few small losses, a few large gains.
    # Sortino's denominator (downside-only std) < Sharpe's (total std), so Sortino > Sharpe.
    dr = pd.Series([0.003] * 200 + [-0.001] * 40 + [0.05] * 12)
    assert sortino(dr) > sharpe(dr)


# ---------- calmar ----------

def test_calmar_positive():
    # 240 days of +1%, 12 days of -5% → clearly net positive → calmar > 0
    dr = pd.Series([0.01] * 240 + [-0.05] * 12)
    assert calmar(dr) > 0


def test_calmar_inf_when_no_drawdown():
    dr = pd.Series([0.001] * 100)
    assert calmar(dr) == float("inf")


# ---------- correlation_matrix ----------

def test_correlation_matrix_shape():
    df = pd.DataFrame({"A": [0.01, -0.02, 0.03], "B": [0.02, -0.01, 0.04]})
    corr = correlation_matrix(df)
    assert corr.shape == (2, 2)


def test_correlation_matrix_diagonal_ones():
    df = pd.DataFrame({"X": [0.1, -0.1, 0.2], "Y": [-0.1, 0.1, -0.2]})
    corr = correlation_matrix(df)
    assert abs(corr.loc["X", "X"] - 1.0) < 1e-9
    assert abs(corr.loc["Y", "Y"] - 1.0) < 1e-9


def test_correlation_matrix_perfectly_inverse():
    s = pd.Series([0.01, -0.02, 0.03, -0.01])
    df = pd.DataFrame({"A": s, "B": -s})
    corr = correlation_matrix(df)
    assert abs(corr.loc["A", "B"] - (-1.0)) < 1e-9

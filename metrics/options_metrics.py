"""
metrics.options_metrics — Black-Scholes pricing and Greeks.

All functions are pure math — no external API calls.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm


def black_scholes(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str = "call",
) -> float:
    """
    Black-Scholes option price.

    S:           current stock price
    K:           strike price
    T:           time to expiry in years (e.g. 30 days = 30/365)
    r:           annual risk-free rate as a decimal (e.g. 0.05 for 5%)
    sigma:       annual implied volatility as a decimal (e.g. 0.30 for 30%)
    option_type: "call" or "put"
    """
    if T <= 0 or sigma <= 0:
        return max(0.0, S - K) if option_type == "call" else max(0.0, K - S)

    d1 = (np.log(S / K) + (r + sigma ** 2 / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == "call":
        return float(S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2))
    return float(K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1))


def greeks(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str = "call",
) -> dict[str, float]:
    """
    Black-Scholes Greeks.

    Returns:
        delta:  change in option price per $1 move in the stock
        gamma:  change in delta per $1 move (convexity)
        theta:  daily time decay in dollars (negative for long options)
        vega:   change in option price per 1% move in implied vol
        rho:    change in option price per 1% move in risk-free rate
    """
    if T <= 0 or sigma <= 0:
        return {"delta": 0.0, "gamma": 0.0, "theta": 0.0, "vega": 0.0, "rho": 0.0}

    d1 = (np.log(S / K) + (r + sigma ** 2 / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    delta = float(norm.cdf(d1) if option_type == "call" else norm.cdf(d1) - 1)
    gamma = float(norm.pdf(d1) / (S * sigma * np.sqrt(T)))

    # Theta: annualised, then divided by 365 for daily decay
    theta_annual = -(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
    if option_type == "call":
        theta_annual -= r * K * np.exp(-r * T) * norm.cdf(d2)
    else:
        theta_annual += r * K * np.exp(-r * T) * norm.cdf(-d2)
    theta = float(theta_annual / 365)

    vega = float(S * norm.pdf(d1) * np.sqrt(T) / 100)   # per 1% vol move
    rho_sign = 1 if option_type == "call" else -1
    rho = float(rho_sign * K * T * np.exp(-r * T) * norm.cdf(rho_sign * d2) / 100)

    return {
        "delta": round(delta, 4),
        "gamma": round(gamma, 6),
        "theta": round(theta, 4),
        "vega":  round(vega, 4),
        "rho":   round(rho, 4),
    }


def implied_volatility(
    market_price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    option_type: str = "call",
    tol: float = 1e-5,
    max_iter: int = 200,
) -> float | None:
    """
    Newton-Raphson implied volatility solver.

    Returns IV as a decimal (e.g. 0.28 for 28%), or None if it fails to converge.
    """
    sigma = 0.30  # initial guess: 30%
    for _ in range(max_iter):
        price = black_scholes(S, K, T, r, sigma, option_type)
        d1 = (np.log(S / K) + (r + sigma ** 2 / 2) * T) / (sigma * np.sqrt(T))
        vega_raw = S * norm.pdf(d1) * np.sqrt(T)
        if vega_raw < 1e-10:
            return None
        diff = price - market_price
        if abs(diff) < tol:
            return round(sigma, 4)
        sigma -= diff / vega_raw
        if sigma <= 0:
            return None
    return None


def atm_summary(
    S: float,
    T: float,
    r: float,
    sigma: float,
) -> dict[str, dict]:
    """
    Price and Greeks for the at-the-money call and put. Strike = S (ATM).

    S:     stock price
    T:     time to expiry in years
    r:     annual risk-free rate
    sigma: annual volatility (historical vol used as IV proxy when market data unavailable)
    """
    return {
        "call": {
            "price": round(black_scholes(S, S, T, r, sigma, "call"), 2),
            **greeks(S, S, T, r, sigma, "call"),
        },
        "put": {
            "price": round(black_scholes(S, S, T, r, sigma, "put"), 2),
            **greeks(S, S, T, r, sigma, "put"),
        },
    }

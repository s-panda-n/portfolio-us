"""
data.screener — Broad stock screener across 55+ tickers.

Scores each ticker on Sharpe ratio, 1-year price momentum, and sector
momentum. Returns a ranked DataFrame for the discovery panel in the UI.
Separate from the optimizer — this is a discovery layer, not allocation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import yfinance as yf


# Full screening universe: {ticker: (sector, short description)}
SCREENER_UNIVERSE: dict[str, tuple[str, str]] = {
    # ── US Mega-cap tech ──────────────────────────────────────────────────────
    "AAPL":  ("Technology",    "Apple — iPhone, Mac, Services"),
    "MSFT":  ("Technology",    "Microsoft — Azure cloud + Copilot AI"),
    "NVDA":  ("Technology",    "Nvidia — dominant AI chip maker"),
    "GOOGL": ("Technology",    "Alphabet — Google search + cloud"),
    "META":  ("Technology",    "Meta — Facebook/Instagram ads + AI"),
    "AMZN":  ("Technology",    "Amazon — AWS cloud + e-commerce"),
    # ── AI / Data ─────────────────────────────────────────────────────────────
    "PLTR":  ("AI/Data",       "Palantir — AI analytics for gov & enterprise"),
    "AMD":   ("Technology",    "AMD — CPUs and AI accelerator chips"),
    "CRWD":  ("Cybersecurity", "CrowdStrike — endpoint security"),
    "SNOW":  ("Technology",    "Snowflake — cloud data warehouse"),
    "SMCI":  ("Technology",    "Super Micro — AI server hardware"),
    "ARM":   ("Technology",    "ARM Holdings — chip architecture IP"),
    # ── Defense / Aerospace ───────────────────────────────────────────────────
    "LMT":   ("Defense",       "Lockheed Martin — F-35 jets, defense contracts"),
    "RTX":   ("Defense",       "RTX (Raytheon) — missiles, jet engines"),
    "NOC":   ("Defense",       "Northrop Grumman — stealth bombers, space"),
    "GD":    ("Defense",       "General Dynamics — tanks, submarines, Gulfstream"),
    "BA":    ("Defense",       "Boeing — commercial jets + defense"),
    # ── EV / Auto ─────────────────────────────────────────────────────────────
    "TSLA":  ("Auto/EV",       "Tesla — EVs, energy storage, full self-driving"),
    "F":     ("Auto",          "Ford — F-150 Lightning, legacy auto transition"),
    "GM":    ("Auto",          "General Motors — Ultium EV platform"),
    "RIVN":  ("Auto/EV",       "Rivian — EV trucks and Amazon delivery vans"),
    "LCID":  ("Auto/EV",       "Lucid Motors — luxury EVs, low sales volume"),
    # ── Streaming / Media ────────────────────────────────────────────────────
    "NFLX":  ("Media",         "Netflix — streaming with 260M+ subscribers"),
    "DIS":   ("Media",         "Disney — parks, streaming, IP franchise"),
    "ROKU":  ("Media",         "Roku — streaming OS and ad platform"),
    "SPOT":  ("Media",         "Spotify — audio streaming"),
    # ── Retail / Consumer ────────────────────────────────────────────────────
    "WMT":   ("Retail",        "Walmart — largest US retailer, growing e-commerce"),
    "COST":  ("Retail",        "Costco — membership warehouse, loyal customer base"),
    "TGT":   ("Retail",        "Target — mass merchandise retail"),
    "AMZN":  ("Retail",        "Amazon — e-commerce + AWS (already above)"),
    # ── Energy ───────────────────────────────────────────────────────────────
    "XOM":   ("Energy",        "ExxonMobil — oil & gas supermajor"),
    "CVX":   ("Energy",        "Chevron — oil & gas, global operations"),
    "COP":   ("Energy",        "ConocoPhillips — pure-play upstream oil"),
    "SLB":   ("Energy",        "SLB (Schlumberger) — oilfield services"),
    "OXY":   ("Energy",        "Occidental Petroleum — Buffett-backed energy"),
    "EOG":   ("Energy",        "EOG Resources — US shale oil"),
    # ── Healthcare / Biotech ──────────────────────────────────────────────────
    "JNJ":   ("Healthcare",    "Johnson & Johnson — pharma + MedTech"),
    "UNH":   ("Healthcare",    "UnitedHealth — largest US health insurer"),
    "PFE":   ("Healthcare",    "Pfizer — vaccines, oncology pipeline"),
    "ABBV":  ("Healthcare",    "AbbVie — Humira + cancer drugs"),
    "MRK":   ("Healthcare",    "Merck — Keytruda cancer blockbuster"),
    "LLY":   ("Healthcare",    "Eli Lilly — Ozempic/Mounjaro weight-loss drugs"),
    # ── Financials ───────────────────────────────────────────────────────────
    "JPM":   ("Financials",    "JPMorgan Chase — largest US bank"),
    "BAC":   ("Financials",    "Bank of America — consumer banking giant"),
    "GS":    ("Financials",    "Goldman Sachs — investment banking"),
    "V":     ("Financials",    "Visa — global payment network"),
    "MA":    ("Financials",    "Mastercard — global payment network"),
    "BRK-B": ("Financials",    "Berkshire Hathaway — Buffett's holding company"),
    # ── ETFs — market ────────────────────────────────────────────────────────
    "SPY":   ("ETF",           "S&P 500 ETF — whole US market"),
    "QQQ":   ("ETF",           "Nasdaq-100 ETF — top 100 US non-financial"),
    "IWM":   ("ETF",           "Russell 2000 — US small-cap"),
    # ── ETFs — international & alternatives ──────────────────────────────────
    "EFA":   ("Intl ETF",      "Developed markets ETF (Europe, Japan, Australia)"),
    "EEM":   ("Intl ETF",      "Emerging markets ETF (China, India, Brazil)"),
    "GLD":   ("Gold",          "Gold ETF — inflation hedge"),
    "VNQ":   ("Real Estate",   "US REIT ETF — commercial real estate"),
    "GSG":   ("Commodities",   "Commodities basket ETF"),
}

# Maps screener tickers to their SPDR sector ETF for momentum scoring
_TICKER_TO_SECTOR_ETF: dict[str, str] = {
    # Tech
    "AAPL": "XLK", "MSFT": "XLK", "NVDA": "XLK", "GOOGL": "XLK",
    "META": "XLK", "PLTR": "XLK", "AMD": "XLK", "CRWD": "XLK",
    "SNOW": "XLK", "SMCI": "XLK", "ARM": "XLK",
    # Consumer discretionary
    "AMZN": "XLY", "TSLA": "XLY", "F": "XLY", "GM": "XLY",
    "RIVN": "XLY", "LCID": "XLY", "NFLX": "XLC", "DIS": "XLC",
    "ROKU": "XLC", "SPOT": "XLC",
    # Consumer staples
    "WMT": "XLP", "COST": "XLP", "TGT": "XLP",
    # Energy
    "XOM": "XLE", "CVX": "XLE", "COP": "XLE",
    "SLB": "XLE", "OXY": "XLE", "EOG": "XLE",
    # Healthcare
    "JNJ": "XLV", "UNH": "XLV", "PFE": "XLV",
    "ABBV": "XLV", "MRK": "XLV", "LLY": "XLV",
    # Financials
    "JPM": "XLF", "BAC": "XLF", "GS": "XLF",
    "V": "XLF", "MA": "XLF", "BRK-B": "XLF",
    # Industrials (Defense)
    "LMT": "XLI", "RTX": "XLI", "NOC": "XLI", "GD": "XLI", "BA": "XLI",
    # Real estate
    "VNQ": "XLRE",
}

TRADING_DAYS = 252


def run_screener(
    period: str = "1y",
    top_n: int = 20,
    sector_returns: dict[str, float] | None = None,
    sentiment_map: dict[str, str] | None = None,
) -> pd.DataFrame:
    """
    Score all SCREENER_UNIVERSE tickers and return top_n ranked by composite score.

    Composite = 0.45 * Sharpe + 0.35 * 1yr momentum + 0.20 * sector momentum

    Returns DataFrame [rank, ticker, sector, sharpe, return_1y_%,
                       sector_1m_%, sentiment, composite_score, description]
    """
    empty = pd.DataFrame(columns=[
        "rank", "ticker", "sector", "sharpe", "return_1y_%",
        "sector_1m_%", "sentiment", "composite_score", "description",
    ])

    tickers = list(dict.fromkeys(SCREENER_UNIVERSE.keys()))  # dedup, preserve order

    try:
        raw = yf.download(
            tickers, period=period, interval="1d",
            progress=False, auto_adjust=True, group_by="ticker",
        )
        # Normalise MultiIndex → Close DataFrame
        if isinstance(raw.columns, pd.MultiIndex):
            close = raw.xs("Close", axis=1, level=0) if "Close" in raw.columns.get_level_values(0) else raw.xs("Close", axis=1, level=1)
        elif "Close" in raw.columns:
            close = raw["Close"].to_frame()
        else:
            close = raw
    except Exception:
        return empty

    rows: list[dict] = []
    for ticker, (sector, description) in SCREENER_UNIVERSE.items():
        if ticker not in close.columns:
            continue
        s = close[ticker].dropna()
        if len(s) < 60:
            continue

        daily = s.pct_change().dropna()
        n = len(daily)
        ann_ret  = float((1 + daily).prod() ** (TRADING_DAYS / n) - 1)
        ann_vol  = float(daily.std() * np.sqrt(TRADING_DAYS))
        sharpe   = ann_ret / ann_vol if ann_vol > 1e-6 else 0.0
        momentum = float(s.iloc[-1] / s.iloc[0] - 1)

        etf = _TICKER_TO_SECTOR_ETF.get(ticker)
        sec_ret_pct = sector_returns.get(etf, 0.0) if sector_returns and etf else 0.0

        sentiment = (sentiment_map or {}).get(ticker, "")

        rows.append({
            "ticker":      ticker,
            "sector":      sector,
            "description": description,
            "sharpe":      round(sharpe, 2),
            "return_1y_%": round(ann_ret * 100, 1),
            "sector_1m_%": round(sec_ret_pct, 1),
            "sentiment":   sentiment,
            "_sh":         sharpe,
            "_mom":        momentum,
            "_sec":        sec_ret_pct / 100,
        })

    if not rows:
        return empty

    df = pd.DataFrame(rows)

    # Min-max normalise each component to [0, 1]
    for col in ["_sh", "_mom", "_sec"]:
        mn, mx = df[col].min(), df[col].max()
        df[f"{col}_n"] = (df[col] - mn) / (mx - mn + 1e-9)

    df["composite_score"] = (
        0.45 * df["_sh_n"] +
        0.35 * df["_mom_n"] +
        0.20 * df["_sec_n"]
    ).round(3)

    df = (
        df.sort_values("composite_score", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )
    df["rank"] = range(1, len(df) + 1)

    return df[[
        "rank", "ticker", "sector", "sharpe", "return_1y_%",
        "sector_1m_%", "sentiment", "composite_score", "description",
    ]]

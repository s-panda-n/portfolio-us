"""
agents.recommendations — plain-language investment recommendation cards.

Generates one card per position explaining what the asset is, why the
optimizer chose it, and how it fits the user's risk level.
"""

from __future__ import annotations

import pandas as pd

# What each ticker is in human terms
ASSET_INFO: dict[str, tuple[str, str]] = {
    # US equities
    "AAPL":  ("US Equity",        "Apple — iPhones, Macs, and a fast-growing services business"),
    "MSFT":  ("US Equity",        "Microsoft — Azure cloud, Office 365, and AI tools like Copilot"),
    "GOOGL": ("US Equity",        "Alphabet (Google) — search dominance, YouTube, and Google Cloud"),
    "NVDA":  ("US Equity",        "Nvidia — the picks-and-shovels play on AI; makes the chips that run it all"),
    "TSLA":  ("US Equity",        "Tesla — electric vehicles, energy storage, and autonomous driving bets"),
    "AMZN":  ("US Equity",        "Amazon — e-commerce plus AWS, the most profitable cloud platform"),
    "META":  ("US Equity",        "Meta — Facebook, Instagram, WhatsApp; advertising machine with AI upside"),
    "JPM":   ("US Equity",        "JPMorgan Chase — largest US bank; profits from higher interest rates"),
    "BRK-B": ("US Equity",        "Berkshire Hathaway — Warren Buffett's diversified holding company"),
    # US ETFs
    "SPY":   ("US Market ETF",    "S&P 500 ETF — owns a slice of every major US company in one trade"),
    "QQQ":   ("US Tech ETF",      "Nasdaq-100 ETF — the 100 largest non-financial US companies, tech-heavy"),
    "VTI":   ("US Market ETF",    "Total US market ETF — every publicly traded American company"),
    "IWM":   ("Small-Cap ETF",    "Russell 2000 ETF — 2,000 smaller US companies; higher growth potential"),
    # International
    "EFA":   ("Intl Developed",   "Developed markets ETF — Europe, Japan, Australia; non-US stocks at lower valuations"),
    "EEM":   ("Emerging Markets", "Emerging markets ETF — China, India, Brazil, and 20+ other growing economies"),
    "VWO":   ("Emerging Markets", "Vanguard emerging markets ETF — similar to EEM, slightly cheaper"),
    # Real estate
    "VNQ":   ("Real Estate",      "US REIT ETF — commercial real estate (offices, warehouses, apartments) without buying property"),
    "IYR":   ("Real Estate",      "iShares US real estate ETF — another way to own US property returns"),
    # Commodities / inflation hedges
    "GLD":   ("Gold",             "Gold ETF — tracks physical gold price; classic hedge against inflation and crises"),
    "SLV":   ("Silver",           "Silver ETF — gold's more volatile sibling; also used in industrial applications"),
    "GSG":   ("Commodities",      "Broad commodities ETF — energy, metals, and agriculture in one ticker"),
    "USO":   ("Oil",              "US Oil ETF — tracks crude oil futures; rises with energy prices"),
    "DJP":   ("Commodities",      "iPath Bloomberg Commodity Index — broad commodities basket"),
    # Bonds
    "AGG":   ("Bonds",            "US Aggregate Bond ETF — a mix of government and investment-grade corporate bonds"),
    "TLT":   ("Long Bonds",       "20+ Year Treasury ETF — long-dated government bonds; rises when rates fall"),
    "SHY":   ("Short Bonds",      "1-3 Year Treasury ETF — near-cash safety; barely moves with interest rates"),
    "TIP":   ("Inflation Bonds",  "Treasury Inflation-Protected ETF — bonds that automatically adjust for inflation"),
    "BND":   ("Bonds",            "Vanguard Total Bond Market ETF — broad bond exposure at low cost"),
    "HYG":   ("High-Yield Bonds", "High-yield (junk) bond ETF — higher interest but higher default risk"),
}

_BOND_ETFS = {"AGG", "TLT", "SHY", "TIP", "BND", "HYG"}
_INTL_ETFS = {"EFA", "EEM", "VWO"}
_REAL_ASSET = {"GLD", "SLV", "GSG", "USO", "VNQ", "IYR", "DJP"}

RISK_CONTEXT = {
    "LOW":    "You chose LOW risk, so the optimizer minimised portfolio swings above all else — you'll see more bonds and defensive assets.",
    "MEDIUM": "You chose MEDIUM risk, so the optimizer blended safety and return — a mix of growth assets and stabilisers.",
    "HIGH":   "You chose HIGH risk, so the optimizer maximised Sharpe ratio — it chased the best return per unit of risk, not the lowest volatility.",
}


def _explain(
    ticker: str,
    weight: float,
    metrics: dict,
    corr_matrix: pd.DataFrame,
    all_tickers: list[str],
    risk_level: str,
    sentiment_signal: str,
) -> str:
    """Build a 3–4 sentence plain-English explanation for one position."""
    asset_class, description = ASSET_INFO.get(ticker, ("Asset", ticker))
    sharpe = metrics.get("sharpe", 0.0)
    ann_ret = metrics.get("ann_return_%", 0.0)
    max_dd = metrics.get("max_dd_%", 0.0)
    ann_vol = metrics.get("ann_vol_%", 0.0)

    parts: list[str] = []

    # Sentence 1 — what it is
    parts.append(description + ".")

    # Sentence 2 — why the optimizer picked it
    if ticker in _BOND_ETFS:
        parts.append(
            f"It's in your portfolio as a stabiliser: with only {ann_vol:.1f}% annual volatility "
            f"it dampens the swings of your equity holdings, especially useful at {risk_level} risk."
        )
    elif ticker in _REAL_ASSET:
        # Check correlation to equities
        equity_tickers = [t for t in all_tickers if t not in _BOND_ETFS and t != ticker and t in corr_matrix.columns]
        if equity_tickers and ticker in corr_matrix.columns:
            avg_corr = corr_matrix.loc[ticker, equity_tickers].mean()
            parts.append(
                f"It acts as a portfolio diversifier — its average correlation to your equities "
                f"is just {avg_corr:.2f}, so it tends to move independently and cushion drawdowns."
            )
        else:
            parts.append("It acts as a real-asset diversifier, reducing the portfolio's reliance on equities.")
    elif ticker in _INTL_ETFS:
        parts.append(
            f"It adds international exposure, which lowers the risk of being too concentrated in US markets "
            f"and captures growth in economies growing faster than the US."
        )
    else:
        # Equity — explain via Sharpe and return
        if sharpe > 1.0:
            parts.append(
                f"The optimizer gave it a large weight because its Sharpe ratio of {sharpe:.2f} is one of "
                f"the highest in your universe — it's generating strong returns without excessive risk."
            )
        elif sharpe > 0.5:
            parts.append(
                f"With a Sharpe ratio of {sharpe:.2f} and an annualised return of {ann_ret:.1f}%, "
                f"it offers a solid risk-adjusted return that earned it a place in the portfolio."
            )
        else:
            parts.append(
                f"It contributes {ann_ret:.1f}% annual return; the optimizer included it primarily "
                f"for diversification across the asset universe."
            )

    # Sentence 3 — a key risk or upside note
    if max_dd < -30:
        parts.append(
            f"One thing to watch: its worst historical drawdown was {max_dd:.1f}%, so expect significant "
            f"short-term drops during market stress."
        )
    elif ann_ret > 20:
        parts.append(f"It's one of the stronger performers in the universe, up {ann_ret:.1f}% annually.")

    # Sentence 4 — sentiment (if meaningful)
    if sentiment_signal == "BULLISH":
        parts.append("Recent news sentiment is positive — analysts and headlines are leaning optimistic.")
    elif sentiment_signal == "BEARISH":
        parts.append("Recent news sentiment is cautious — worth monitoring headlines before adding more.")

    return " ".join(parts)


def generate(
    allocation_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    corr_matrix: pd.DataFrame,
    capital: float,
    risk_level: str,
    sentiment_df: pd.DataFrame | None = None,
) -> list[dict]:
    """
    Generate one recommendation card per position, sorted by weight descending.

    Returns list of dicts with keys:
        rank, ticker, weight_pct, dollars, asset_class, description,
        explanation, sharpe, ann_return_pct, max_dd_pct, sentiment
    """
    all_tickers = list(allocation_df["ticker"])

    sentiment_map: dict[str, str] = {}
    if sentiment_df is not None and not sentiment_df.empty and "signal" in sentiment_df.columns:
        sentiment_map = dict(zip(sentiment_df["ticker"], sentiment_df["signal"]))

    cards: list[dict] = []
    for rank, (_, row) in enumerate(allocation_df.iterrows(), start=1):
        ticker = row["ticker"]
        weight = float(row["weight_%"])
        dollars = int(row["dollars"])
        asset_class, description = ASSET_INFO.get(ticker, ("Asset", ticker))
        m = metrics_df.loc[ticker].to_dict() if ticker in metrics_df.index else {}

        cards.append({
            "rank": rank,
            "ticker": ticker,
            "weight_%": weight,
            "dollars": dollars,
            "asset_class": asset_class,
            "description": description,
            "explanation": _explain(
                ticker, weight, m, corr_matrix,
                all_tickers, risk_level,
                sentiment_map.get(ticker, ""),
            ),
            "sharpe": round(m.get("sharpe", 0.0), 2),
            "ann_return_%": round(m.get("ann_return_%", 0.0), 1),
            "max_dd_%": round(m.get("max_dd_%", 0.0), 1),
            "sentiment": sentiment_map.get(ticker, ""),
        })

    return cards

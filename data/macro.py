"""
data.macro — Macro & market-structure signals.

Sources (all free-tier):
  - yfinance: VIX (^VIX), SPDR sector ETFs (XLK, XLE, ...)
  - yfinance: per-ticker earnings calendar
  - Finnhub:  general_news() for macro/geopolitical headlines
"""

from __future__ import annotations

import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta


SECTOR_ETFS: dict[str, str] = {
    "XLK":  "Technology",
    "XLE":  "Energy",
    "XLF":  "Financials",
    "XLV":  "Health Care",
    "XLI":  "Industrials",
    "XLU":  "Utilities",
    "XLY":  "Consumer Disc.",
    "XLP":  "Consumer Staples",
    "XLRE": "Real Estate",
    "XLB":  "Materials",
    "XLC":  "Comm. Services",
}

# Maps individual tickers to their nearest SPDR sector ETF (for momentum tilt)
TICKER_TO_SECTOR_ETF: dict[str, str] = {
    "AAPL": "XLK", "MSFT": "XLK", "NVDA": "XLK",
    "GOOGL": "XLK", "META": "XLK",
    "AMZN": "XLY", "TSLA": "XLY",
    "JPM": "XLF", "BRK-B": "XLF",
    "VNQ": "XLRE", "IYR": "XLRE",
    "USO": "XLE", "GSG": "XLE",
}


def get_vix() -> dict:
    """
    Current VIX level, 1-month trend, and fear regime.
      regime: 'low' (<18) | 'elevated' (18-25) | 'high' (>25)
      trend:  'rising' | 'falling' | 'stable'
    Returns empty dict on failure.
    """
    try:
        df = yf.download("^VIX", period="3mo", interval="1d", progress=False, auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        close = df["Close"].dropna() if not df.empty and "Close" in df.columns else pd.Series(dtype=float)
        if close.empty:
            return {}
        current = float(close.iloc[-1])
        prior   = float(close.iloc[-22]) if len(close) >= 22 else current
        trend   = "rising" if current > prior * 1.05 else "falling" if current < prior * 0.95 else "stable"
        regime  = "high" if current > 25 else "elevated" if current > 18 else "low"
        return {
            "level":     round(current, 2),
            "month_ago": round(prior, 2),
            "trend":     trend,
            "regime":    regime,
        }
    except Exception:
        return {}


def get_sector_momentum(lookback_days: int = 21) -> pd.DataFrame:
    """
    1-month return for every SPDR sector ETF, sorted best-to-worst.
    Fetched in one batched yfinance call.
    Returns DataFrame [ticker, sector, return_1m_%].
    """
    empty = pd.DataFrame(columns=["ticker", "sector", "return_1m_%"])
    etf_list = list(SECTOR_ETFS.keys())
    try:
        raw = yf.download(etf_list, period="3mo", interval="1d",
                          progress=False, auto_adjust=True)
        # MultiIndex columns: (field, ticker) — grab Close level
        if isinstance(raw.columns, pd.MultiIndex):
            close_df = raw["Close"]
        elif "Close" in raw.columns:
            close_df = raw[["Close"]]
        else:
            return empty

        rows: list[dict] = []
        for etf in etf_list:
            if etf not in close_df.columns:
                continue
            s = close_df[etf].dropna()
            if len(s) < lookback_days:
                continue
            ret = (float(s.iloc[-1]) / float(s.iloc[-lookback_days]) - 1) * 100
            rows.append({"ticker": etf, "sector": SECTOR_ETFS[etf], "return_1m_%": round(ret, 2)})
        if not rows:
            return empty
        return pd.DataFrame(rows).sort_values("return_1m_%", ascending=False).reset_index(drop=True)
    except Exception:
        return empty


def get_earnings_calendar(tickers: list[str]) -> pd.DataFrame:
    """
    Next earnings date for each ticker via yfinance .calendar.
    Returns DataFrame [ticker, earnings_date, days_until], sorted soonest-first.
    """
    today = datetime.today().date()
    rows: list[dict] = []
    for ticker in tickers:
        try:
            cal = yf.Ticker(ticker).calendar
            if cal is None:
                continue
            if isinstance(cal, dict):
                ed = cal.get("Earnings Date")
                if ed is None:
                    continue
                if isinstance(ed, (list, tuple)):
                    ed = ed[0]
                ed = pd.Timestamp(ed).date()
            elif isinstance(cal, pd.DataFrame):
                if "Earnings Date" not in cal.index:
                    continue
                ed = pd.Timestamp(cal.loc["Earnings Date"].iloc[0]).date()
            else:
                continue
            rows.append({
                "ticker":        ticker,
                "earnings_date": str(ed),
                "days_until":    (ed - today).days,
            })
        except Exception:
            continue
    if not rows:
        return pd.DataFrame(columns=["ticker", "earnings_date", "days_until"])
    return pd.DataFrame(rows).sort_values("days_until").reset_index(drop=True)


def get_macro_news(finnhub_key: str, days_back: int = 14) -> pd.DataFrame:
    """
    Macro/geopolitical headlines from Finnhub general_news().
    Covers: Fed decisions, CPI, tariffs, geopolitics, market-moving macro events.
    Returns DataFrame [datetime, headline, source, url].
    """
    empty = pd.DataFrame(columns=["datetime", "headline", "source", "url"])
    if not finnhub_key:
        return empty
    try:
        import finnhub
        client = finnhub.Client(api_key=finnhub_key)
        cutoff = datetime.today() - timedelta(days=days_back)
        items  = client.general_news("general", min_id=0)
        rows: list[dict] = []
        for item in (items or []):
            try:
                dt = datetime.fromtimestamp(item.get("datetime", 0))
                if dt < cutoff:
                    continue
                rows.append({
                    "datetime": dt,
                    "headline": item.get("headline", "")[:140],
                    "source":   item.get("source", ""),
                    "url":      item.get("url", ""),
                })
            except Exception:
                continue
        if not rows:
            return empty
        return pd.DataFrame(rows).sort_values("datetime", ascending=False).reset_index(drop=True)
    except Exception:
        return empty

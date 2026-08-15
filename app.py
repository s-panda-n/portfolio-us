"""
Portfolio-US — Bloomberg-terminal-style Streamlit dashboard.
Run: streamlit run app.py
"""

from __future__ import annotations

import os
from collections import defaultdict

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="PORTFOLIO-US",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="📊",
)

# Inject Streamlit Cloud secrets into os.environ
try:
    for _k in ["FINNHUB_API_KEY", "FRED_API_KEY", "ALPACA_API_KEY", "ALPACA_SECRET_KEY"]:
        if _k in st.secrets and not os.getenv(_k):
            os.environ[_k] = st.secrets[_k]
except Exception:
    pass

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

finnhub_key = os.getenv("FINNHUB_API_KEY")
fred_key    = os.getenv("FRED_API_KEY")

from ui.theme import apply_terminal_theme, ticker_tape, colored_metric, panel_header
from ui.charts import efficient_frontier_chart
from agents.orchestrator import run_pipeline, DEFAULT_EQUITIES
from data.bonds import BOND_ETFS
from data.screener import run_screener, SCREENER_UNIVERSE
from data.portfolio import compute_portfolio_value
from data.holdings_store import load as _load_holdings, save as _save_holdings
from metrics.options_metrics import atm_summary

apply_terminal_theme()

_AMBER = "#FF9F1C"
_GREEN = "#00D964"
_RED   = "#FF3B30"
_MUTED = "#8B949E"
_BG    = "#000000"
_PANEL = "#0D1117"
_GRID  = "#2A2E35"
_TEXT  = "#E6E6E6"
_FONT  = "JetBrains Mono, IBM Plex Mono, Consolas, monospace"

_ETF_SET = set(BOND_ETFS) | {
    "SPY", "QQQ", "IWM", "VTI", "DIA", "EFA", "EEM", "VWO", "INDA", "EWJ",
    "VNQ", "GLD", "SLV", "GSG", "TLT", "AGG", "HYG", "USO", "DBA", "JETS",
    "XLK", "XLE", "XLF", "XLV", "XLI", "XLY", "XLP", "XLRE", "XLU", "XLB", "XLC",
}

# Sector lookup from screener universe
_SECTOR_MAP: dict[str, str] = {t: sec for t, (sec, _) in SCREENER_UNIVERSE.items()}

def _sector_for(ticker: str) -> str:
    return _SECTOR_MAP.get(ticker, "Other")


# ── Session state ──────────────────────────────────────────────────────────────
if "tickers"  not in st.session_state: st.session_state["tickers"]  = list(DEFAULT_EQUITIES)
if "refresh"  not in st.session_state: st.session_state["refresh"]  = False
if "holdings" not in st.session_state: st.session_state["holdings"] = _load_holdings()


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### COMMAND CENTER")
    capital = st.number_input(
        "CAPITAL ($)", min_value=100, max_value=10_000_000,
        value=500, step=100, format="%d",
    )
    risk = st.select_slider("RISK TOLERANCE", options=["LOW", "MEDIUM", "HIGH"], value="MEDIUM")
    return_tgt = st.select_slider("RETURN TARGET", options=["LOW", "HIGH"], value="HIGH")
    # Map (risk tolerance, return target) → optimizer mode
    _PROFILE_MAP = {
        ("LOW",    "LOW"):  "LOW",     # strict min-variance, capital preservation
        ("LOW",    "HIGH"): "MEDIUM",  # best return within low-vol constraint
        ("MEDIUM", "LOW"):  "LOW",     # blended but cautious
        ("MEDIUM", "HIGH"): "HIGH",    # blended but growth-oriented
        ("HIGH",   "LOW"):  "MEDIUM",  # aggressive budget, moderate return push
        ("HIGH",   "HIGH"): "HIGH",    # full max-Sharpe, all signals on
    }
    optimizer_risk = _PROFILE_MAP[(risk, return_tgt)]
    st.caption(f"Profile: **{risk} Risk · {return_tgt} Return** → `{optimizer_risk}` optimizer")
    st.markdown("---")

    with st.expander("WATCHLIST", expanded=False):
        add_col, btn_col = st.columns([3, 1])
        with add_col:
            new_ticker = st.text_input("ticker", placeholder="e.g. TSLA", label_visibility="collapsed")
        with btn_col:
            if st.button("ADD", use_container_width=True) and new_ticker:
                t = new_ticker.strip().upper()
                if t not in st.session_state["tickers"]:
                    st.session_state["tickers"].append(t)
                st.rerun()
        for t in list(st.session_state["tickers"]):
            c1, c2 = st.columns([3, 1])
            c1.markdown(f"`{t}`")
            if c2.button("✕", key=f"rm_{t}"):
                st.session_state["tickers"].remove(t)
                st.rerun()

    st.markdown("---")
    st.markdown("### MY HOLDINGS")
    with st.form("add_holding", clear_on_submit=True):
        h_ticker = st.text_input("TICKER", placeholder="AAPL")
        h_date   = st.date_input("DATE BOUGHT")
        h_col1, h_col2 = st.columns(2)
        with h_col1:
            h_shares = st.number_input("SHARES", min_value=0.0, value=0.0, step=0.01)
        with h_col2:
            h_cost   = st.number_input("COST / SH ($)", min_value=0.0, value=0.0, step=0.01)
        if st.form_submit_button("ADD HOLDING", use_container_width=True):
            if h_ticker and h_shares > 0 and h_cost > 0:
                st.session_state["holdings"].append({
                    "ticker":     h_ticker.strip().upper(),
                    "date":       h_date.isoformat(),
                    "shares":     round(h_shares, 4),
                    "cost_basis": round(h_cost, 4),
                })
                _save_holdings(st.session_state["holdings"])
                st.rerun()

    for i, h in enumerate(list(st.session_state["holdings"])):
        r1, r2 = st.columns([4, 1])
        r1.caption(f"`{h['ticker']}` {h['shares']:.2f} sh @ ${h['cost_basis']:.2f}")
        if r2.button("✕", key=f"rmh_{i}"):
            st.session_state["holdings"].pop(i)
            _save_holdings(st.session_state["holdings"])
            st.rerun()

    st.markdown("---")
    if st.button("▶ RUN / REFRESH", use_container_width=True):
        st.session_state["refresh"] = True
        st.cache_data.clear()
        st.rerun()
    st.caption("PORTFOLIO-US  |  yfinance · Finnhub · FRED")
    if not finnhub_key:
        st.warning("FINNHUB_API_KEY missing — news/sentiment disabled")


# ── Cached pipeline ────────────────────────────────────────────────────────────
@st.cache_data(ttl=300, show_spinner=False)
def _run(tickers_t, capital, risk, finnhub_key, fred_key, refresh):
    return run_pipeline(
        list(tickers_t), capital, risk,
        finnhub_key=finnhub_key, fred_key=fred_key, refresh=refresh,
    )

@st.cache_data(ttl=300, show_spinner=False)
def _portfolio(holdings_t):
    return compute_portfolio_value([dict(h) for h in holdings_t])

@st.cache_data(ttl=3600, show_spinner=False)
def _screener_cached():
    return run_screener(period="3mo")


# ── Run pipeline ───────────────────────────────────────────────────────────────
with st.spinner("FETCHING DATA & RUNNING OPTIMIZER..."):
    try:
        result = _run(
            tuple(st.session_state["tickers"]),
            float(capital), optimizer_risk, finnhub_key, fred_key,
            st.session_state.get("refresh", False),
        )
        st.session_state["refresh"] = False
    except Exception as e:
        st.error(f"Pipeline failed: {e}")
        st.stop()

# Extract frequently used results
vix_info     = result.get("vix", {})
vix_regime   = vix_info.get("regime", "low")
vix_level    = vix_info.get("level", 18.0)
sector_df    = result.get("sector_momentum", pd.DataFrame())
sentiment_df = result.get("sentiment", pd.DataFrame())
macro_news   = result.get("macro_news", pd.DataFrame())

sentiment_map: dict[str, str] = {}
if sentiment_df is not None and not sentiment_df.empty and "signal" in sentiment_df.columns:
    sentiment_map = dict(zip(sentiment_df["ticker"], sentiment_df["signal"]))


# ── Ticker tape ────────────────────────────────────────────────────────────────
tape_items = [
    (t, f"${result['last_prices'][t]:.2f}", result["pct_changes"].get(t, 0.0))
    for t in st.session_state["tickers"] if t in result["last_prices"]
]
if tape_items:
    ticker_tape(tape_items)

st.markdown("# PORTFOLIO-US")
st.caption("MULTI-ASSET ALLOCATION TERMINAL  |  RESEARCH USE ONLY — NOT FINANCIAL ADVICE")

for err in result.get("errors", []):
    st.warning(err)


# ── MY PORTFOLIO ───────────────────────────────────────────────────────────────
pdata: dict = {}
current_w: dict[str, float] = {}

if st.session_state["holdings"]:
    holdings_key = tuple(tuple(sorted(h.items())) for h in st.session_state["holdings"])
    with st.spinner("Computing portfolio P&L..."):
        pdata = _portfolio(holdings_key)
    current_w = pdata.get("weights", {})

    with st.container(border=True):
        panel_header("MY PORTFOLIO", "Live P&L · Actual holdings")
        m1, m2, m3, m4 = st.columns(4)
        with m1: colored_metric("TOTAL VALUE",  f"${pdata['total_value']:,.2f}", 0.0)
        with m2: colored_metric("COST BASIS",   f"${pdata['total_cost']:,.2f}",  0.0)
        with m3: colored_metric("P&L ($)",      f"${pdata['total_pnl']:,.2f}",   pdata["total_pnl"])
        with m4: colored_metric("P&L (%)",      f"{pdata['total_pnl_pct']:.2f}%", pdata["total_pnl_pct"])

        hist = pdata.get("value_history", pd.Series())
        if not (isinstance(hist, pd.Series) and hist.empty):
            fig_hist = go.Figure()
            fig_hist.add_trace(go.Scatter(
                x=hist.index, y=hist.values, mode="lines",
                line=dict(color=_AMBER, width=2),
                fill="tozeroy", fillcolor="rgba(255,159,28,0.08)",
                hovertemplate="$%{y:,.0f}<extra></extra>",
            ))
            fig_hist.update_layout(
                height=180, paper_bgcolor=_BG, plot_bgcolor=_PANEL,
                font=dict(family=_FONT, color=_TEXT),
                margin=dict(l=10, r=10, t=10, b=10),
                xaxis=dict(gridcolor=_GRID),
                yaxis=dict(gridcolor=_GRID, tickprefix="$", tickformat=",.0f"),
                showlegend=False,
            )
            st.plotly_chart(fig_hist)

        positions = pdata.get("positions", pd.DataFrame())
        if isinstance(positions, pd.DataFrame) and not positions.empty:
            st.dataframe(positions, hide_index=True, use_container_width=True)


# ── Top metrics row ────────────────────────────────────────────────────────────
opt = result["optimal"]
c1, c2, c3, c4, c5 = st.columns(5)
with c1: colored_metric("CAPITAL",      f"${capital:,.0f}",             0.0)
with c2: colored_metric("ANN. RETURN",  f"{opt['return']:.1%}",         opt["return"] * 100)
with c3: colored_metric("VOLATILITY",   f"{opt['volatility']:.1%}",     0.0, suffix="")
with c4: colored_metric("SHARPE",       f"{opt['sharpe']:.2f}",         round(opt["sharpe"] - 1.0, 2), suffix="")
with c5: colored_metric("MAX DRAWDOWN", f"{opt['max_drawdown']:.1%}",   opt["max_drawdown"] * 100)


# ── Macro Signals (collapsible) ────────────────────────────────────────────────
with st.expander("▼  MACRO SIGNALS", expanded=False):
    left_col, right_col = st.columns(2)

    # Left: VIX + sentiment for HELD tickers
    with left_col:
        st.markdown("**YOUR HOLDINGS**")
        if vix_info:
            vix_color = _RED if vix_regime == "high" else _AMBER if vix_regime == "elevated" else _GREEN
            st.markdown(
                f"VIX <span style='color:{vix_color}; font-size:2em; font-weight:bold'>"
                f"{vix_level:.1f}</span> "
                f"<span style='color:{_MUTED}'>({vix_regime.upper()})</span>",
                unsafe_allow_html=True,
            )
            st.caption(f"Trend: {vix_info.get('trend', '—')}  ·  Month ago: {vix_info.get('month_ago', 0):.1f}")

        held_tickers = list(current_w.keys())
        if held_tickers and sentiment_df is not None and not sentiment_df.empty and "ticker" in sentiment_df.columns:
            st.caption("News sentiment for your holdings:")
            held_rows = sentiment_df[sentiment_df["ticker"].isin(held_tickers)]
            for _, row in held_rows.iterrows():
                sig = row.get("signal", "NEUTRAL")
                color = _GREEN if sig == "BULLISH" else _RED if sig == "BEARISH" else _MUTED
                st.markdown(
                    f"<span style='color:{color}'>● </span>`{row['ticker']}` — **{sig}**",
                    unsafe_allow_html=True,
                )
        elif not held_tickers:
            st.caption("Add holdings to see sentiment context here.")

    # Right: VIX regime context for suggested portfolio + sector momentum
    with right_col:
        st.markdown("**SUGGESTED PORTFOLIO**")
        if vix_info:
            if vix_regime == "high":
                st.markdown(
                    f"<span style='color:{_RED}'>⚠ High VIX — optimizer reduced signal weight by 50%</span>",
                    unsafe_allow_html=True,
                )
            elif vix_regime == "elevated":
                st.markdown(
                    f"<span style='color:{_AMBER}'>⚡ Elevated VIX — sentiment signals at half weight</span>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"<span style='color:{_GREEN}'>✓ Low VIX — full signal weight active</span>",
                    unsafe_allow_html=True,
                )

        if sector_df is not None and not sector_df.empty and "return_1m_%" in sector_df.columns:
            st.caption("Top/bottom sector rotation (1M):")
            sorted_sec = sector_df.sort_values("return_1m_%", ascending=False)
            for _, row in sorted_sec.iterrows():
                r = row["return_1m_%"]
                color = _GREEN if r >= 0 else _RED
                st.markdown(
                    f"<span style='color:{color}'>{r:+.1f}%</span>  {row['sector']}",
                    unsafe_allow_html=True,
                )


# ── Recommendations ────────────────────────────────────────────────────────────
def _build_recommendations() -> list[dict]:
    opt_w = dict(zip(result["allocation"]["ticker"], result["allocation"]["weight_%"] / 100))
    all_tickers = set(opt_w.keys()) | set(current_w.keys())
    metrics_df = result["metrics"]
    last_prices = result["last_prices"]

    rows = []
    for ticker in all_tickers:
        curr = current_w.get(ticker, 0.0)
        sugg = opt_w.get(ticker, 0.0)
        diff = sugg - curr

        if curr < 0.005 and sugg < 0.005:
            continue

        sent = sentiment_map.get(ticker, "NEUTRAL")

        # Determine action
        if sugg < 0.005 and curr > 0.01:
            action, priority = "SELL", 0
        elif curr < 0.005 and sugg > 0.015:
            action, priority = "BUY", 1
        elif diff > 0.025:
            action, priority = "BUY MORE", 2
        elif diff < -0.025:
            action, priority = "TRIM", 3
        else:
            action, priority = "HOLD", 4

        # Build reason
        parts = []
        if sent == "BULLISH":
            parts.append("bullish news")
        elif sent == "BEARISH":
            parts.append("bearish news")

        if ticker in metrics_df.index:
            sh = float(metrics_df.loc[ticker, "sharpe"])
            ret = float(metrics_df.loc[ticker, "ann_return_%"])
            if sh > 1.5:
                parts.append(f"Sharpe {sh:.2f}")
            elif sh < 0.4 and action in ("SELL", "TRIM"):
                parts.append(f"weak Sharpe {sh:.2f}")
            if ret > 20:
                parts.append(f"{ret:.0f}% 3Y return")

        if vix_regime == "high":
            if action in ("TRIM", "SELL"):
                parts.append("high-VIX risk reduction")
            elif action in ("BUY", "BUY MORE"):
                parts.append("held despite high VIX")

        reason = " · ".join(parts) if parts else "optimizer allocation target"

        # Black-Scholes 30d ATM call for individual equities
        bs_call = None
        if ticker not in _ETF_SET and ticker in last_prices:
            price = last_prices[ticker]
            if price > 0 and ticker in metrics_df.index:
                try:
                    iv = max(float(metrics_df.loc[ticker, "ann_vol_%"]) / 100, 0.05)
                    s = atm_summary(price, 30 / 365, 0.045, iv)
                    bs_call = s["call"]["price"]
                except Exception:
                    pass

        rows.append({
            "ticker":      ticker,
            "sector":      _sector_for(ticker),
            "action":      action,
            "priority":    priority,
            "reason":      reason,
            "current_%":   round(curr * 100, 1),
            "suggested_%": round(sugg * 100, 1),
            "diff_%":      round(diff * 100, 1),
            "dollars":     round(sugg * capital),
            "sentiment":   sent,
            "bs_call":     bs_call,
        })

    rows.sort(key=lambda x: (x["priority"], -abs(x["diff_%"])))
    return rows[:10]


recs = _build_recommendations()

with st.container(border=True):
    panel_header("RECOMMENDATIONS", f"{risk} Risk · {return_tgt} Return · max 10 · by sector")

    if not recs:
        st.caption("Portfolio aligned with optimizer targets — no action needed.")
    else:
        by_sector: dict[str, list] = defaultdict(list)
        for row in recs:
            by_sector[row["sector"]].append(row)

        _action_colors = {
            "BUY": _GREEN, "BUY MORE": _GREEN,
            "SELL": _RED, "TRIM": _AMBER, "HOLD": _MUTED,
        }

        for sector in sorted(by_sector.keys()):
            st.markdown(f"**{sector}**")
            for row in by_sector[sector]:
                acolor = _action_colors.get(row["action"], _TEXT)
                rc1, rc2, rc3 = st.columns([0.9, 3.2, 1.4])

                with rc1:
                    st.markdown(
                        f"<div style='font-family:{_FONT}; color:{acolor}; font-weight:bold; font-size:1.05em; line-height:1.4'>"
                        f"{row['action']}</div>"
                        f"<div style='font-size:1em; color:{_TEXT}'>{row['ticker']}</div>",
                        unsafe_allow_html=True,
                    )
                    scolor = _GREEN if row["sentiment"] == "BULLISH" else _RED if row["sentiment"] == "BEARISH" else _MUTED
                    st.markdown(
                        f"<span style='color:{scolor}; font-size:0.78em'>● {row['sentiment']}</span>",
                        unsafe_allow_html=True,
                    )

                with rc2:
                    st.markdown(
                        f"<span style='color:{_MUTED}; font-size:0.88em'>{row['reason']}</span>",
                        unsafe_allow_html=True,
                    )
                    if row["bs_call"] is not None:
                        st.markdown(
                            f"<span style='color:{_AMBER}; font-size:0.82em'>30d ATM call ~${row['bs_call']:.2f}</span>",
                            unsafe_allow_html=True,
                        )

                with rc3:
                    if row["action"] != "HOLD":
                        diff_sign = "+" if row["diff_%"] > 0 else ""
                        st.markdown(
                            f"<span style='color:{_MUTED}; font-size:0.85em'>"
                            f"{row['current_%']:.1f}% → {row['suggested_%']:.1f}%</span>",
                            unsafe_allow_html=True,
                        )
                        if row["dollars"] > 0:
                            st.markdown(
                                f"<span style='color:{acolor}; font-size:0.85em'>"
                                f"{diff_sign}{row['diff_%']:.1f}%  ${row['dollars']:,}</span>",
                                unsafe_allow_html=True,
                            )

            st.markdown("---")


# ── Efficient Frontier ─────────────────────────────────────────────────────────
current_point: dict | None = None
if current_w:
    metrics_df = result["metrics"]
    held = [t for t in current_w if t in metrics_df.index]
    if held:
        w_arr = np.array([current_w[t] for t in held])
        w_arr = w_arr / w_arr.sum()
        rets = np.array([float(metrics_df.loc[t, "ann_return_%"]) / 100 for t in held])
        vols = np.array([float(metrics_df.loc[t, "ann_vol_%"]) / 100 for t in held])
        current_point = {
            "return":     float(np.dot(w_arr, rets)),
            "volatility": float(np.dot(w_arr, vols)) * 0.75,
        }

with st.container(border=True):
    panel_header(
        "EFFICIENT FRONTIER",
        "◆ = your current portfolio  ★ = suggested allocation" if current_point else "★ = suggested allocation",
    )
    if not result["frontier"].empty:
        st.plotly_chart(
            efficient_frontier_chart(result["frontier"], result["optimal"], current_point=current_point),
            use_container_width=True,
        )
    else:
        st.caption("Frontier data unavailable.")


# ── Stock Screener ─────────────────────────────────────────────────────────────
with st.container(border=True):
    panel_header("STOCK SCREENER", "280+ tickers · ranked by Sharpe · momentum · sector rotation · 1-hour cache")

    with st.spinner("Loading screener..."):
        screener_df = _screener_cached()

    if screener_df.empty:
        st.caption("Screener data unavailable.")
    else:
        sectors_list = ["ALL"] + sorted(screener_df["sector"].unique().tolist())
        sel_sector = st.selectbox("Filter by sector", sectors_list, key="screener_sector", label_visibility="collapsed")
        view_df = screener_df if sel_sector == "ALL" else screener_df[screener_df["sector"] == sel_sector]

        def _style_signal(val: str) -> str:
            return f"color: {_GREEN}" if val == "BULL" else (f"color: {_RED}" if val == "BEAR" else f"color: {_MUTED}")

        def _style_action(val: str) -> str:
            if val == "BUY":
                return f"color: {_GREEN}; font-weight: bold"
            if val == "SELL":
                return f"color: {_RED}; font-weight: bold"
            return f"color: {_MUTED}"

        def _style_score(val: float) -> str:
            return f"color: {_GREEN}" if val >= 0.62 else (f"color: {_AMBER}" if val >= 0.40 else f"color: {_RED}")

        styled = (
            view_df[["rank", "ticker", "sector", "signal", "action",
                      "sharpe", "return_3m_%", "sector_1m_%", "composite_score", "description"]]
            .style
            .applymap(_style_signal, subset=["signal"])
            .applymap(_style_action, subset=["action"])
            .applymap(_style_score,  subset=["composite_score"])
            .format({
                "composite_score": "{:.3f}",
                "return_3m_%":     "{:+.1f}%",
                "sector_1m_%":     "{:+.1f}%",
                "sharpe":          "{:.2f}",
            }, na_rep="—")
        )
        st.dataframe(styled, hide_index=True, use_container_width=True)


# ── Sentiment badges ───────────────────────────────────────────────────────────
if sentiment_df is not None and not sentiment_df.empty and "signal" in sentiment_df.columns:
    with st.container(border=True):
        panel_header("SENTIMENT", "News signal per ticker")
        all_sent_tickers = sentiment_df["ticker"].tolist()
        selected_sent = st.multiselect(
            "Show tickers",
            options=all_sent_tickers,
            default=all_sent_tickers,
            key="sent_ticker_select",
            label_visibility="collapsed",
        )
        filtered_sent = sentiment_df[sentiment_df["ticker"].isin(selected_sent)] if selected_sent else sentiment_df
        if filtered_sent.empty:
            st.caption("Select at least one ticker above.")
        else:
            badge_cols = st.columns(min(len(filtered_sent), 8))
            for (_, row), col in zip(filtered_sent.iterrows(), badge_cols):
                sig = row.get("signal", "NEUTRAL")
                color = _GREEN if sig == "BULLISH" else _RED if sig == "BEARISH" else _MUTED
                col.markdown(
                    f"<div style='text-align:center; padding:6px'>"
                    f"<span style='color:{color}; font-size:1.5em'>●</span><br/>"
                    f"<span style='font-size:0.78em'>{row['ticker']}</span><br/>"
                    f"<span style='color:{color}; font-size:0.72em'>{sig}</span>"
                    f"</div>",
                    unsafe_allow_html=True,
                )


# ── Tips ───────────────────────────────────────────────────────────────────────
def _generate_tips() -> list[tuple[str, str]]:
    tips: list[tuple[str, str]] = []

    # Tip 1: VIX / volatility hedge
    if vix_info:
        lvl = vix_level
        if lvl > 25:
            tips.append((
                "🔥 Hedge alert",
                f"VIX at **{lvl:.1f}** — the market is sweating. 30-day SPY puts cost "
                f"roughly 0.5–1% of notional right now. That's cheap insurance when the "
                f"house might be on fire. Consider buying protection before it gets worse.",
            ))
        elif lvl < 15:
            tips.append((
                "😴 Complacency warning",
                f"VIX at **{lvl:.1f}** — everyone's napping at the wheel. Options are "
                f"dirt cheap. This is exactly when professionals quietly buy protective "
                f"puts *before* the alarm goes off. Don't wait for the panic.",
            ))
        else:
            tips.append((
                "📡 VIX watch",
                f"VIX at **{lvl:.1f}** — neither panic nor euphoria. Historically, spikes "
                f"above 25 mark great dip-buying entries. A collapse below 12 signals "
                f"peak complacency. Watch both boundaries.",
            ))

    # Tip 2: Sector rotation
    if sector_df is not None and not sector_df.empty and "return_1m_%" in sector_df.columns:
        top = sector_df.loc[sector_df["return_1m_%"].idxmax()]
        bot = sector_df.loc[sector_df["return_1m_%"].idxmin()]
        tips.append((
            "🔄 Follow the rotation",
            f"**{top['sector']}** is winning this month (**{top['return_1m_%']:+.1f}%**). "
            f"**{bot['sector']}** is getting left behind (**{bot['return_1m_%']:+.1f}%**). "
            f"Smart money rotates before the crowd catches on — are you positioned for this?",
        ))

    # Tip 3: Macro article
    if macro_news is not None and not macro_news.empty:
        row = macro_news.iloc[0]
        headline = str(row.get("headline", ""))[:100]
        url = str(row.get("url", ""))
        if headline and url.startswith("http"):
            tips.append(("📰 Macro pulse", f"[{headline}…]({url})"))
        elif headline:
            tips.append(("📰 Macro pulse", headline + "…"))

    return tips


tips = _generate_tips()
if tips:
    with st.container(border=True):
        panel_header("TIPS", "3 things worth knowing right now")
        tip_cols = st.columns(len(tips))
        for (title, body), col in zip(tips, tip_cols):
            with col:
                st.markdown(f"**{title}**")
                st.markdown(body)


# ── Reasoning log (hidden) ─────────────────────────────────────────────────────
with st.expander("ALLOCATION REASONING  ›", expanded=False):
    panel_header("SYSTEM LOG", "Decision pipeline output")
    st.code("\n".join(result["reasoning"]), language=None)

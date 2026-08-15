"""
Portfolio-US — Bloomberg-terminal-style Streamlit dashboard.
Run: streamlit run app.py
"""

from __future__ import annotations

import os
import streamlit as st

st.set_page_config(
    page_title="PORTFOLIO-US",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="📊",
)

# Inject Streamlit Cloud secrets into os.environ so data modules use os.getenv()
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
alpaca_key  = os.getenv("ALPACA_API_KEY")

import pandas as pd
import plotly.graph_objects as go

from ui.theme import apply_terminal_theme, ticker_tape, colored_metric, panel_header
from ui.charts import (
    efficient_frontier_chart, correlation_heatmap,
    cumulative_returns_chart, yield_curve_chart, allocation_bar_chart,
)
from agents.orchestrator import run_pipeline, DEFAULT_EQUITIES
from agents.recommendations import generate as gen_recommendations, RISK_CONTEXT
from data.bonds import BOND_ETFS
from data.options import get_atm_options
from data.portfolio import compute_portfolio_value
from metrics.options_metrics import atm_summary

apply_terminal_theme()

_AMBER = "#FF9F1C"
_GREEN = "#00D964"
_RED   = "#FF3B30"
_BG    = "#000000"
_PANEL = "#0D1117"
_GRID  = "#2A2E35"
_TEXT  = "#E6E6E6"
_FONT  = "JetBrains Mono, IBM Plex Mono, Consolas, monospace"

# ── Session state ─────────────────────────────────────────────────────────────
if "tickers"  not in st.session_state: st.session_state["tickers"]  = list(DEFAULT_EQUITIES)
if "refresh"  not in st.session_state: st.session_state["refresh"]  = False
if "holdings" not in st.session_state: st.session_state["holdings"] = []


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### COMMAND CENTER")
    capital = st.number_input(
        "CAPITAL ($)", min_value=100, max_value=10_000_000,
        value=50_000, step=100, format="%d",
    )
    risk = st.select_slider("RISK LEVEL", options=["LOW", "MEDIUM", "HIGH"], value="MEDIUM")
    st.markdown("---")

    # ── Watchlist ──────────────────────────────────────────────────────────────
    st.markdown("### WATCHLIST")
    add_col, btn_col = st.columns([3, 1])
    with add_col:
        new_ticker = st.text_input("ticker", placeholder="e.g. TSLA", label_visibility="collapsed")
    with btn_col:
        if st.button("ADD", width="stretch") and new_ticker:
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

    # ── My holdings ────────────────────────────────────────────────────────────
    st.markdown("### MY HOLDINGS")
    with st.form("add_holding", clear_on_submit=True):
        h_ticker = st.text_input("TICKER", placeholder="AAPL")
        h_date   = st.date_input("DATE BOUGHT")
        h_col1, h_col2 = st.columns(2)
        with h_col1:
            h_shares = st.number_input("SHARES", min_value=0.0, value=0.0, step=0.01)
        with h_col2:
            h_cost   = st.number_input("COST / SH ($)", min_value=0.0, value=0.0, step=0.01)
        submitted = st.form_submit_button("ADD HOLDING", use_container_width=True)
        if submitted and h_ticker and h_shares > 0 and h_cost > 0:
            st.session_state["holdings"].append({
                "ticker":     h_ticker.strip().upper(),
                "date":       h_date.isoformat(),
                "shares":     round(h_shares, 4),
                "cost_basis": round(h_cost, 4),
            })
            st.rerun()

    for i, h in enumerate(list(st.session_state["holdings"])):
        r1, r2 = st.columns([4, 1])
        r1.caption(f"`{h['ticker']}` {h['shares']:.2f} sh @ ${h['cost_basis']:.2f}")
        if r2.button("✕", key=f"rmh_{i}"):
            st.session_state["holdings"].pop(i)
            st.rerun()
    st.markdown("---")

    if st.button("▶ RUN / REFRESH", width="stretch"):
        st.session_state["refresh"] = True
        st.cache_data.clear()
        st.rerun()
    st.markdown("---")
    st.caption("PORTFOLIO-US v0.1  |  yfinance · Finnhub · FRED · Alpaca")
    if not finnhub_key: st.warning("FINNHUB_API_KEY missing — news disabled")
    if not fred_key:    st.info("FRED_API_KEY missing — yield curve disabled")
    if not alpaca_key:  st.info("ALPACA_API_KEY missing — options chain disabled")


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


with st.spinner("FETCHING DATA & RUNNING OPTIMIZER..."):
    try:
        result = _run(
            tuple(st.session_state["tickers"]),
            float(capital), risk, finnhub_key, fred_key,
            st.session_state.get("refresh", False),
        )
        st.session_state["refresh"] = False
    except Exception as e:
        st.error(f"Pipeline failed: {e}")
        st.stop()


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


# ── MY PORTFOLIO (only when holdings are entered) ──────────────────────────────
if st.session_state["holdings"]:
    holdings_key = tuple(tuple(sorted(h.items())) for h in st.session_state["holdings"])
    with st.spinner("Computing portfolio P&L..."):
        pdata = _portfolio(holdings_key)

    with st.container(border=True):
        panel_header("MY PORTFOLIO", "Actual holdings · Live P&L")

        m1, m2, m3, m4 = st.columns(4)
        with m1: colored_metric("TOTAL VALUE",  f"${pdata['total_value']:,.2f}", 0.0)
        with m2: colored_metric("COST BASIS",   f"${pdata['total_cost']:,.2f}",  0.0)
        with m3: colored_metric("P&L ($)",      f"${pdata['total_pnl']:,.2f}",   pdata["total_pnl"])
        with m4: colored_metric("P&L (%)",      f"{pdata['total_pnl_pct']:.2f}%", pdata["total_pnl_pct"])

        if not pdata["value_history"].empty:
            hist = pdata["value_history"]
            fig_hist = go.Figure()
            fig_hist.add_trace(go.Scatter(
                x=hist.index, y=hist.values,
                mode="lines", name="Portfolio Value",
                line=dict(color=_AMBER, width=2),
                fill="tozeroy", fillcolor="rgba(255,159,28,0.08)",
                hovertemplate="$%{y:,.0f}<extra></extra>",
            ))
            fig_hist.update_layout(
                height=220,
                paper_bgcolor=_BG, plot_bgcolor=_PANEL,
                font=dict(family=_FONT, color=_TEXT),
                margin=dict(l=10, r=10, t=10, b=10),
                xaxis=dict(gridcolor=_GRID),
                yaxis=dict(gridcolor=_GRID, tickprefix="$", tickformat=",.0f"),
                showlegend=False,
            )
            st.plotly_chart(fig_hist)

        if not pdata["positions"].empty:
            st.dataframe(pdata["positions"], hide_index=True, width="stretch")

        # Show delta vs optimizer recommendation
        current_w = pdata["weights"]
        if current_w and not result["allocation"].empty:
            st.markdown("**SUGGESTED CHANGES** *(to reach optimal allocation)*")
            delta_rows = []
            all_tickers_union = set(current_w) | set(result["allocation"]["ticker"])
            opt_w = dict(zip(result["allocation"]["ticker"], result["allocation"]["weight_%"] / 100))
            for t in sorted(all_tickers_union):
                curr = current_w.get(t, 0.0)
                sugg = opt_w.get(t, 0.0)
                diff = sugg - curr
                if abs(diff) < 0.005:
                    action = "HOLD"
                elif diff > 0:
                    action = f"BUY +{diff*100:.1f}%"
                else:
                    action = f"TRIM {diff*100:.1f}%"
                delta_rows.append({
                    "ticker": t,
                    "current_%": round(curr * 100, 1),
                    "suggested_%": round(sugg * 100, 1),
                    "delta_%": round(diff * 100, 1),
                    "action": action,
                })
            st.dataframe(pd.DataFrame(delta_rows), hide_index=True, width="stretch")

    st.markdown("---")


# ── Row 1: Portfolio metrics ───────────────────────────────────────────────────
opt = result["optimal"]
c1, c2, c3, c4, c5 = st.columns(5)
with c1: colored_metric("CAPITAL",      f"${capital:,.0f}",             0.0)
with c2: colored_metric("ANN. RETURN",  f"{opt['return']:.1%}",         opt["return"] * 100)
with c3: colored_metric("VOLATILITY",   f"{opt['volatility']:.1%}",     0.0, suffix="")
with c4: colored_metric("SHARPE",       f"{opt['sharpe']:.2f}",         round(opt["sharpe"] - 1.0, 2), suffix="")
with c5: colored_metric("MAX DRAWDOWN", f"{opt['max_drawdown']:.1%}",   opt["max_drawdown"] * 100)


# ── Macro Signals ─────────────────────────────────────────────────────────────
with st.container(border=True):
    panel_header("MACRO SIGNALS", "VIX · Sector Rotation · Earnings Calendar")
    vix_col, sector_col, earn_col = st.columns([0.7, 1.6, 1.2])

    with vix_col:
        vix = result.get("vix", {})
        if vix:
            vix_color = _RED if vix["regime"] == "high" else _AMBER if vix["regime"] == "elevated" else _GREEN
            st.markdown("**VIX — FEAR INDEX**")
            st.markdown(
                f"<span style='color:{vix_color}; font-size:2.4em; font-weight:bold'>"
                f"{vix['level']:.1f}</span>",
                unsafe_allow_html=True,
            )
            st.caption(f"{vix['regime'].upper()} · {vix['trend']}")
            st.caption(f"1 month ago: {vix['month_ago']:.1f}")
            st.caption("< 18 = calm · 18–25 = elevated · > 25 = fear")
        else:
            st.caption("VIX unavailable")

    with sector_col:
        sector_df_display = result.get("sector_momentum")
        if sector_df_display is not None and not sector_df_display.empty:
            panel_header("SECTOR MOMENTUM", "1-Month Return")
            colors = [_GREEN if r >= 0 else _RED for r in sector_df_display["return_1m_%"]]
            fig_sec = go.Figure(go.Bar(
                x=sector_df_display["return_1m_%"],
                y=sector_df_display["sector"],
                orientation="h",
                marker_color=colors,
                text=[f"{r:+.1f}%" for r in sector_df_display["return_1m_%"]],
                textposition="outside",
                hovertemplate="%{y}: %{x:.1f}%<extra></extra>",
            ))
            fig_sec.update_layout(
                height=280,
                paper_bgcolor=_BG, plot_bgcolor=_PANEL,
                font=dict(family=_FONT, color=_TEXT, size=10),
                margin=dict(l=10, r=70, t=10, b=10),
                xaxis=dict(gridcolor=_GRID, zeroline=True, zerolinecolor=_GRID),
                yaxis=dict(gridcolor=_GRID),
                showlegend=False,
            )
            st.plotly_chart(fig_sec)
        else:
            st.caption("Sector momentum unavailable")

    with earn_col:
        earn_df_display = result.get("earnings_calendar")
        if earn_df_display is not None and not earn_df_display.empty:
            panel_header("EARNINGS CALENDAR", "Next report dates")
            st.dataframe(earn_df_display, hide_index=True, width="stretch")
        else:
            st.caption("No upcoming earnings found")


# ── Recommendations ────────────────────────────────────────────────────────────
st.markdown("")
with st.container(border=True):
    panel_header("YOUR PORTFOLIO — EXPLAINED", "Plain-English reasoning for every position")
    st.caption(RISK_CONTEXT.get(risk, ""))

    # Full equity universe — all assets scored, not just what optimizer allocated
    _universe = [t for t in result["metrics"].index.tolist() if t not in set(BOND_ETFS)]
    cards = gen_recommendations(
        result["allocation"], result["metrics"], result["corr_matrix"],
        capital, risk, result.get("sentiment"),
        universe_tickers=_universe,
    )

    # If user has holdings, flag positions that already have large exposure
    current_w = pdata["weights"] if st.session_state["holdings"] else {}

    for card in cards:
        is_allocated = card.get("allocated", True)
        with st.container(border=True):
            meta_col, text_col = st.columns([1, 3])
            with meta_col:
                label = f"#{card['rank']}  {card['ticker']}"
                value = (
                    f"{card['weight_%']:.1f}%  ·  ${card['dollars']:,}"
                    if is_allocated else "—  not allocated"
                )
                st.metric(
                    label, value,
                    delta=f"{card['ann_return_%']:+.1f}% ann." if card["ann_return_%"] != 0 else None,
                )
                st.caption(card["asset_class"])
                if not is_allocated:
                    st.caption("⬡ Not in optimal portfolio")
                if card["sentiment"] == "BULLISH":
                    st.success("● BULLISH")
                elif card["sentiment"] == "BEARISH":
                    st.error("● BEARISH")

                # Show "already own X%" badge if user has holdings
                if current_w.get(card["ticker"], 0) > 0.01:
                    st.info(f"You own {current_w[card['ticker']]*100:.1f}%")

            with text_col:
                st.markdown(card["explanation"])
                k1, k2, k3 = st.columns(3)
                k1.metric("Sharpe",    card["sharpe"])
                k2.metric("3Y Return", f"{card['ann_return_%']:.1f}%")
                k3.metric("Max DD",    f"{card['max_dd_%']:.1f}%")


# ── Row 2: Allocation | Frontier | Correlation ─────────────────────────────────
st.markdown("")
col_a, col_b, col_c = st.columns([1.1, 1.2, 1.1])

with col_a:
    with st.container(border=True):
        panel_header("ALLOCATION", f"{risk}  |  ${capital:,.0f}")
        st.dataframe(result["allocation"], width="stretch", hide_index=True)
        st.plotly_chart(allocation_bar_chart(result["allocation"]))

with col_b:
    with st.container(border=True):
        panel_header("EFFICIENT FRONTIER", "Monte-Carlo  ★ = optimal")
        st.plotly_chart(efficient_frontier_chart(result["frontier"], result["optimal"]))

with col_c:
    with st.container(border=True):
        panel_header("CORRELATION MATRIX", "Pairwise Pearson")
        st.plotly_chart(correlation_heatmap(result["corr_matrix"]))


# ── Row 3: Cumulative returns ──────────────────────────────────────────────────
with st.container(border=True):
    panel_header("CUMULATIVE RETURNS", "PORTFOLIO = weighted blend")
    st.plotly_chart(cumulative_returns_chart(result["cum_returns"]))


# ── Row 4: Asset metrics ───────────────────────────────────────────────────────
with st.container(border=True):
    panel_header("ASSET METRICS", "Per-ticker risk / return")
    st.dataframe(result["metrics"], width="stretch")


# ── Stock Screener ────────────────────────────────────────────────────────────
_screener_df = result.get("screener", pd.DataFrame())
with st.container(border=True):
    panel_header("STOCK SCREENER", "Top picks across 55+ stocks — ranked by Sharpe · momentum · sector rotation")
    if _screener_df.empty:
        st.caption("Screener data unavailable.")
    else:
        st.caption(
            "Composite score = 45% Sharpe (risk-adjusted return) + "
            "35% 1-year price momentum + 20% sector 1-month rotation. "
            "Add any ticker to your watchlist via the sidebar to run the optimizer on it."
        )

        # Sector filter
        sectors = ["ALL"] + sorted(_screener_df["sector"].unique().tolist())
        sel_sector = st.selectbox("Filter by sector", sectors, index=0, label_visibility="collapsed")
        view_df = _screener_df if sel_sector == "ALL" else _screener_df[_screener_df["sector"] == sel_sector]

        # Colour composite score column
        def _score_color(val: float) -> str:
            if val >= 0.7:
                return f"color: {_GREEN}"
            if val >= 0.4:
                return f"color: {_AMBER}"
            return f"color: {_RED}"

        styled = (
            view_df[["rank", "ticker", "sector", "sharpe", "return_1y_%",
                      "sector_1m_%", "sentiment", "composite_score", "description"]]
            .style
            .applymap(_score_color, subset=["composite_score"])
            .format({"composite_score": "{:.3f}", "return_1y_%": "{:+.1f}%",
                     "sector_1m_%": "{:+.1f}%", "sharpe": "{:.2f}"})
        )
        st.dataframe(styled, hide_index=True, width="stretch")


# ── Row 5: Fixed income ────────────────────────────────────────────────────────
with st.container(border=True):
    panel_header("FIXED INCOME", "Bond ETF proxies + Treasury yield curve")
    bond_col, yield_col = st.columns([1, 1.3])
    with bond_col:
        bond_rows = [t for t in BOND_ETFS if t in result["metrics"].index]
        if bond_rows:
            st.dataframe(result["metrics"].loc[bond_rows], width="stretch")
        else:
            st.caption("Bond ETF data unavailable.")
    with yield_col:
        if not result["yield_curve"].empty:
            st.plotly_chart(yield_curve_chart(result["yield_curve"]))
        else:
            st.info("Set FRED_API_KEY in .env to display the live US Treasury yield curve.")


# ── Row 6: News & sentiment ────────────────────────────────────────────────────
with st.container(border=True):
    panel_header("NEWS & SENTIMENT", "Finnhub — 30-day window · all portfolio positions · macro headlines")
    macro_news = result.get("macro_news", pd.DataFrame())
    if result["news"].empty and macro_news.empty:
        st.caption("Set FINNHUB_API_KEY in .env to enable live news and sentiment.")
    else:
        sent_df = result.get("sentiment")
        if sent_df is not None and not sent_df.empty:
            st.dataframe(sent_df, width="stretch", hide_index=True)
        tickers_with_news = result["news"]["ticker"].unique().tolist() if not result["news"].empty else []
        tab_names = tickers_with_news + (["MACRO NEWS"] if not macro_news.empty else [])
        if tab_names:
            tabs = st.tabs(tab_names)
            for tab, t in zip(tabs[:len(tickers_with_news)], tickers_with_news):
                with tab:
                    df_t = result["news"][result["news"]["ticker"] == t]
                    st.dataframe(df_t[["datetime", "headline", "source"]],
                                 width="stretch", hide_index=True)
            if not macro_news.empty:
                with tabs[-1]:
                    st.caption("Macro & geopolitical headlines — Fed, CPI, tariffs, geopolitics")
                    st.dataframe(macro_news[["datetime", "headline", "source"]],
                                 width="stretch", hide_index=True)


# ── Row 7: Options & Derivatives ───────────────────────────────────────────────
with st.container(border=True):
    panel_header("OPTIONS & DERIVATIVES", "Black-Scholes pricing + live Alpaca chain")

    # Options only make sense for individual equities, not broad ETFs
    _ETF_SET = set(BOND_ETFS) | {"SPY", "QQQ", "VTI", "EFA", "EEM", "VWO",
                                  "VNQ", "GLD", "SLV", "GSG", "USO", "IWM"}
    equity_positions = [
        t for t in result["allocation"]["ticker"] if t not in _ETF_SET
    ][:4]

    if not equity_positions:
        st.caption("No individual equity positions in the current allocation for options analysis.")
    else:
        tabs = st.tabs(equity_positions)
        for tab, ticker in zip(tabs, equity_positions):
            with tab:
                price = result["last_prices"].get(ticker, 0.0)
                if not price:
                    st.caption(f"No price data for {ticker}.")
                    continue

                ann_vol = (
                    result["metrics"].loc[ticker, "ann_vol_%"] / 100
                    if ticker in result["metrics"].index else 0.25
                )
                rf_rate = 0.045   # ~current Fed Funds proxy
                T_30    = 30 / 365
                T_60    = 60 / 365

                bs_col, chain_col = st.columns(2)

                with bs_col:
                    panel_header(f"{ticker} — BLACK-SCHOLES",
                                 f"ATM · Stock: ${price:.2f} · IV proxy: {ann_vol*100:.0f}%")

                    rows_bs = []
                    for T_days, T in [(30, T_30), (60, T_60)]:
                        s = atm_summary(price, T, rf_rate, ann_vol)
                        for opt_type in ("call", "put"):
                            g = s[opt_type]
                            rows_bs.append({
                                "expiry": f"{T_days}d",
                                "type":   opt_type,
                                "price":  f"${g['price']:.2f}",
                                "delta":  g["delta"],
                                "theta/day": g["theta"],
                                "vega/1%":   g["vega"],
                                "gamma":     g["gamma"],
                            })
                    st.dataframe(pd.DataFrame(rows_bs), hide_index=True, width="stretch")

                    st.caption(
                        "**Delta**: $ move per $1 stock move  |  "
                        "**Theta**: daily time decay ($)  |  "
                        "**Vega**: $ change per 1% IV move"
                    )

                with chain_col:
                    panel_header(f"{ticker} — LIVE CHAIN",
                                 "Alpaca/OPRA · nearest strikes · ≤45 days")
                    if not alpaca_key:
                        st.info("Set ALPACA_API_KEY + ALPACA_SECRET_KEY in .env to enable live chains.")
                    else:
                        chain_df = get_atm_options(ticker, price, max_expiry_days=45, n_strikes=3)
                        if chain_df.empty:
                            st.info(
                                "No live options data returned. "
                                "OPRA subscription may be required on your Alpaca account."
                            )
                        else:
                            st.dataframe(chain_df, hide_index=True, width="stretch")


# ── Row 8: Reasoning ───────────────────────────────────────────────────────────
with st.expander("ALLOCATION REASONING  ›", expanded=False):
    panel_header("SYSTEM LOG", "Decision pipeline output")
    st.code("\n".join(result["reasoning"]), language=None)

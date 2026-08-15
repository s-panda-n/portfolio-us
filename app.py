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

# Inject Streamlit Cloud secrets into os.environ so data modules can use os.getenv()
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
from ui.charts import (
    efficient_frontier_chart,
    correlation_heatmap,
    cumulative_returns_chart,
    yield_curve_chart,
    allocation_bar_chart,
)
from agents.orchestrator import run_pipeline, DEFAULT_EQUITIES
from data.bonds import BOND_ETFS

apply_terminal_theme()

# ── Session state ─────────────────────────────────────────────────────────────
if "tickers" not in st.session_state:
    st.session_state["tickers"] = list(DEFAULT_EQUITIES)
if "refresh" not in st.session_state:
    st.session_state["refresh"] = False


# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### COMMAND CENTER")
    capital = st.number_input(
        "CAPITAL ($)", min_value=1_000, max_value=10_000_000,
        value=50_000, step=1_000, format="%d",
    )
    risk = st.select_slider("RISK LEVEL", options=["LOW", "MEDIUM", "HIGH"], value="MEDIUM")
    st.markdown("---")

    st.markdown("### WATCHLIST")
    add_col, btn_col = st.columns([3, 1])
    with add_col:
        new_ticker = st.text_input("ticker", placeholder="e.g. TSLA",
                                   label_visibility="collapsed")
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
    if st.button("▶ RUN / REFRESH", use_container_width=True):
        st.session_state["refresh"] = True
        st.cache_data.clear()
        st.rerun()

    st.markdown("---")
    st.caption("PORTFOLIO-US v0.1")
    st.caption("Data: yfinance · Finnhub · FRED")
    if not finnhub_key:
        st.warning("FINNHUB_API_KEY missing — news disabled")
    if not fred_key:
        st.info("FRED_API_KEY missing — yield curve disabled")


# ── Cached pipeline call ──────────────────────────────────────────────────────
@st.cache_data(ttl=300, show_spinner=False)
def _run(
    tickers_t: tuple[str, ...],
    capital: float,
    risk: str,
    finnhub_key: str | None,
    fred_key: str | None,
    refresh: bool,
) -> dict:
    return run_pipeline(
        list(tickers_t), capital, risk,
        finnhub_key=finnhub_key,
        fred_key=fred_key,
        refresh=refresh,
    )


with st.spinner("FETCHING DATA & RUNNING OPTIMIZER..."):
    try:
        result = _run(
            tuple(st.session_state["tickers"]),
            float(capital), risk,
            finnhub_key, fred_key,
            st.session_state.get("refresh", False),
        )
        st.session_state["refresh"] = False
    except Exception as e:
        st.error(f"Pipeline failed: {e}")
        st.stop()


# ── Ticker tape ───────────────────────────────────────────────────────────────
tape_items = [
    (t, f"${result['last_prices'][t]:.2f}", result["pct_changes"].get(t, 0.0))
    for t in st.session_state["tickers"]
    if t in result["last_prices"]
]
if tape_items:
    ticker_tape(tape_items)

st.markdown("# PORTFOLIO-US")
st.caption("MULTI-ASSET ALLOCATION TERMINAL  |  FOR RESEARCH USE ONLY — NOT FINANCIAL ADVICE")

for err in result.get("errors", []):
    st.warning(err)


# ── Row 1: Portfolio metrics ──────────────────────────────────────────────────
opt = result["optimal"]
st.markdown("---")
c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    colored_metric("CAPITAL", f"${capital:,.0f}", 0.0)
with c2:
    colored_metric("ANN. RETURN", f"{opt['return']:.1%}", opt["return"] * 100)
with c3:
    colored_metric("VOLATILITY", f"{opt['volatility']:.1%}", 0.0, suffix="")
with c4:
    colored_metric("SHARPE", f"{opt['sharpe']:.2f}",
                   round(opt["sharpe"] - 1.0, 2), suffix="")
with c5:
    colored_metric("MAX DRAWDOWN", f"{opt['max_drawdown']:.1%}",
                   opt["max_drawdown"] * 100)


# ── Row 2: Allocation | Frontier | Correlation ────────────────────────────────
st.markdown("")
col_a, col_b, col_c = st.columns([1.1, 1.2, 1.1])

with col_a:
    with st.container(border=True):
        panel_header("ALLOCATION", f"{risk}  |  ${capital:,.0f}")
        st.dataframe(result["allocation"], use_container_width=True, hide_index=True)
        st.plotly_chart(
            allocation_bar_chart(result["allocation"]),
            use_container_width=True,
        )

with col_b:
    with st.container(border=True):
        panel_header("EFFICIENT FRONTIER", "Monte-Carlo  ★ = optimal")
        st.plotly_chart(
            efficient_frontier_chart(result["frontier"], result["optimal"]),
            use_container_width=True,
        )

with col_c:
    with st.container(border=True):
        panel_header("CORRELATION MATRIX", "Pairwise Pearson")
        st.plotly_chart(
            correlation_heatmap(result["corr_matrix"]),
            use_container_width=True,
        )


# ── Row 3: Cumulative returns ─────────────────────────────────────────────────
with st.container(border=True):
    panel_header("CUMULATIVE RETURNS", "PORTFOLIO = weighted blend")
    st.plotly_chart(
        cumulative_returns_chart(result["cum_returns"]),
        use_container_width=True,
    )


# ── Row 4: Asset metrics table ────────────────────────────────────────────────
with st.container(border=True):
    panel_header("ASSET METRICS", "Per-ticker risk / return")
    st.dataframe(result["metrics"], use_container_width=True)


# ── Row 5: Fixed income ───────────────────────────────────────────────────────
with st.container(border=True):
    panel_header("FIXED INCOME", "Bond ETF proxies + Treasury yield curve")
    bond_col, yield_col = st.columns([1, 1.3])

    with bond_col:
        bond_rows = [t for t in BOND_ETFS if t in result["metrics"].index]
        if bond_rows:
            st.dataframe(result["metrics"].loc[bond_rows], use_container_width=True)
        else:
            st.caption("Bond ETF data unavailable.")

    with yield_col:
        if not result["yield_curve"].empty:
            st.plotly_chart(
                yield_curve_chart(result["yield_curve"]),
                use_container_width=True,
            )
        else:
            st.info("Set FRED_API_KEY in .env to display the live US Treasury yield curve.")


# ── Row 6: News & sentiment ───────────────────────────────────────────────────
with st.container(border=True):
    panel_header("NEWS & SENTIMENT", "Finnhub — top equity holdings")

    if result["news"].empty:
        st.caption("Set FINNHUB_API_KEY in .env to enable live news and sentiment.")
    else:
        sent_df = result.get("sentiment")
        if sent_df is not None and not sent_df.empty:
            st.dataframe(sent_df, use_container_width=True, hide_index=True)

        tickers_with_news = result["news"]["ticker"].unique().tolist()
        if tickers_with_news:
            st.markdown("")
            tabs = st.tabs(tickers_with_news)
            for tab, t in zip(tabs, tickers_with_news):
                with tab:
                    df_t = result["news"][result["news"]["ticker"] == t]
                    st.dataframe(
                        df_t[["datetime", "headline", "source"]],
                        use_container_width=True,
                        hide_index=True,
                    )


# ── Row 7: Reasoning ──────────────────────────────────────────────────────────
with st.expander("ALLOCATION REASONING  ›", expanded=False):
    panel_header("SYSTEM LOG", "Decision pipeline output")
    st.code("\n".join(result["reasoning"]), language=None)

"""
Bloomberg-terminal-theme demo for Portfolio-US.

Run with:
    streamlit run app_demo.py

This is a standalone visual demo with fake data - not wired to any real data agent yet.
Use it to confirm the theme looks right before building actual panels on top of it.
"""

import streamlit as st
import pandas as pd
import numpy as np
from ui.theme import apply_terminal_theme, ticker_tape, colored_metric, panel_header

st.set_page_config(page_title="PORTFOLIO-US", layout="wide", initial_sidebar_state="expanded")
apply_terminal_theme()

# ---------- Ticker tape header ----------
ticker_tape([
    ("AAPL", "$231.42", 0.84),
    ("MSFT", "$418.10", -0.32),
    ("SPY", "$612.10", 0.21),
    ("AGG", "$98.44", -0.05),
    ("TLT", "$91.20", 0.63),
    ("NVDA", "$142.88", 2.14),
])

st.markdown("# PORTFOLIO-US")
st.caption("MULTI-ASSET ALLOCATION TERMINAL — DEMO DATA")

# ---------- Sidebar: command bar ----------
with st.sidebar:
    st.markdown("### CONTROLS")
    capital = st.number_input("CAPITAL ($)", min_value=0, value=50000, step=1000)
    risk = st.select_slider("RISK LEVEL", options=["LOW", "MEDIUM", "HIGH"], value="MEDIUM")
    st.markdown("---")
    st.markdown("### WATCHLIST")
    st.text_input("ADD TICKER", placeholder="e.g. AAPL")

# ---------- Top row: key metrics ----------
col1, col2, col3, col4 = st.columns(4)
with col1:
    colored_metric("PORTFOLIO VALUE", "$50,000", 0.0)
with col2:
    colored_metric("1D CHANGE", "+$412.30", 0.83)
with col3:
    colored_metric("SHARPE RATIO", "1.42", 0.0, suffix="")
with col4:
    colored_metric("MAX DRAWDOWN", "-8.2%", -8.2)

st.markdown("")

# ---------- Two-panel layout: allocation table + placeholder chart ----------
left, right = st.columns([1.2, 1])

with left:
    with st.container(border=True):
        panel_header("ALLOCATION", "Current portfolio composition")
        df = pd.DataFrame({
            "TICKER": ["AAPL", "MSFT", "SPY", "AGG", "TLT"],
            "ASSET CLASS": ["Equity", "Equity", "ETF", "Bond ETF", "Bond ETF"],
            "WEIGHT %": [22.5, 18.0, 30.0, 15.0, 14.5],
            "VALUE ($)": [11250, 9000, 15000, 7500, 7250],
        })
        st.dataframe(df, use_container_width=True, hide_index=True)

with right:
    with st.container(border=True):
        panel_header("RETURN SERIES", "90-day simulated equity curve")
        chart_data = pd.DataFrame(
            np.cumsum(np.random.randn(90)) + 100,
            columns=["Portfolio Value"],
        )
        st.line_chart(chart_data, height=280)

# ---------- Bottom row: news/sentiment panel ----------
with st.container(border=True):
    panel_header("NEWS & SENTIMENT", "Latest headlines — demo data")
    news_df = pd.DataFrame({
        "TIME": ["09:41", "09:22", "08:58"],
        "TICKER": ["AAPL", "NVDA", "SPY"],
        "HEADLINE": [
            "Apple reports strong services growth in Q3",
            "Nvidia announces new AI chip partnership",
            "S&P 500 opens flat amid mixed earnings",
        ],
        "SENTIMENT": ["+0.62", "+0.81", "+0.04"],
    })
    st.dataframe(news_df, use_container_width=True, hide_index=True)

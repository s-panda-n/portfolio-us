"""
Bloomberg-terminal-style theming for Streamlit.

Usage:
    from ui.theme import apply_terminal_theme, ticker_tape, colored_metric, panel_header

    st.set_page_config(page_title="Portfolio-US", layout="wide")
    apply_terminal_theme()
"""

from __future__ import annotations

import streamlit as st

TERMINAL_CSS = """
<style>
/* ---------- Font: monospace everywhere ---------- */
html, body, [class*="css"], .stMarkdown, .stMetric, .stDataFrame, .stTextInput input,
.stSelectbox, .stNumberInput input, .stButton button {
    font-family: 'JetBrains Mono', 'IBM Plex Mono', 'Consolas', 'Courier New', monospace !important;
}

/* ---------- Reduce default Streamlit padding for density ---------- */
.block-container {
    padding-top: 1rem;
    padding-bottom: 1rem;
    padding-left: 1.5rem;
    padding-right: 1.5rem;
    max-width: 100%;
}

/* ---------- Panel-style containers (use st.container(border=True)) ---------- */
div[data-testid="stVerticalBlockBorderWrapper"] {
    background-color: #0D1117;
    border: 1px solid #2A2E35 !important;
    border-radius: 0px !important;
}

/* ---------- Metric widgets: tighten, uppercase labels, letter-spacing ---------- */
div[data-testid="stMetric"] {
    background-color: #0D1117;
    border: 1px solid #2A2E35;
    padding: 10px 14px;
}
div[data-testid="stMetricLabel"] {
    text-transform: uppercase;
    letter-spacing: 1.5px;
    font-size: 0.7rem !important;
    color: #8B949E !important;
}
div[data-testid="stMetricValue"] {
    font-size: 1.4rem !important;
    font-weight: 700;
}
/* Up = green, down = red, matches terminal price feeds */
div[data-testid="stMetricDelta"] svg { display: none; } /* hide default arrow, we color text instead */

/* ---------- Dataframes / tables: dense, amber headers ---------- */
[data-testid="stDataFrame"] thead tr th {
    background-color: #0D1117 !important;
    color: #FF9F1C !important;
    text-transform: uppercase;
    letter-spacing: 1px;
    font-size: 0.72rem !important;
    border-bottom: 1px solid #FF9F1C !important;
}
[data-testid="stDataFrame"] tbody tr:nth-child(odd) {
    background-color: #050708 !important;
}
[data-testid="stDataFrame"] tbody tr:hover {
    background-color: #1A1E24 !important;
}

/* ---------- Headers: amber, uppercase, tracked-out like terminal section titles ---------- */
h1, h2, h3 {
    color: #FF9F1C !important;
    text-transform: uppercase;
    letter-spacing: 2px;
    font-weight: 700 !important;
}
h1 { font-size: 1.4rem !important; border-bottom: 1px solid #2A2E35; padding-bottom: 8px; }
h2 { font-size: 1.05rem !important; }
h3 { font-size: 0.9rem !important; color: #8B949E !important; }

/* ---------- Sidebar as a "command bar" ---------- */
section[data-testid="stSidebar"] {
    border-right: 1px solid #2A2E35;
}

/* ---------- Buttons: sharp, amber outline, no rounded corners ---------- */
.stButton button {
    background-color: transparent;
    border: 1px solid #FF9F1C;
    color: #FF9F1C;
    text-transform: uppercase;
    letter-spacing: 1px;
    font-size: 0.75rem;
}
.stButton button:hover {
    background-color: #FF9F1C;
    color: #000000;
}

/* ---------- Custom scrollbar ---------- */
::-webkit-scrollbar { width: 10px; height: 10px; }
::-webkit-scrollbar-track { background: #000000; }
::-webkit-scrollbar-thumb { background: #FF9F1C; }

/* ---------- Ticker tape marquee ---------- */
.ticker-wrap {
    width: 100%;
    overflow: hidden;
    background-color: #0D1117;
    border-top: 1px solid #2A2E35;
    border-bottom: 1px solid #2A2E35;
    padding: 6px 0;
    margin-bottom: 12px;
}
.ticker-move {
    display: inline-block;
    white-space: nowrap;
    animation: ticker-scroll 30s linear infinite;
    font-size: 0.85rem;
    letter-spacing: 1px;
}
.ticker-item { display: inline-block; padding: 0 28px; }
.ticker-up { color: #00D964; }
.ticker-down { color: #FF3B30; }
.ticker-flat { color: #8B949E; }
@keyframes ticker-scroll {
    0%   { transform: translateX(0%); }
    100% { transform: translateX(-50%); }
}
</style>
"""


def apply_terminal_theme() -> None:
    """Call once, right after st.set_page_config(), before any other UI."""
    st.markdown(TERMINAL_CSS, unsafe_allow_html=True)


def ticker_tape(items: list[tuple[str, str, float]]) -> None:
    """
    Render a scrolling ticker tape like the Bloomberg terminal header.

    items: list of (symbol, price_str, pct_change) e.g.
        [("AAPL", "$231.42", 0.84), ("SPY", "$612.10", -0.12), ...]
    """
    spans = []
    # duplicate the list once so the marquee loop looks seamless
    for symbol, price, pct in items * 2:
        css_class = "ticker-up" if pct > 0 else "ticker-down" if pct < 0 else "ticker-flat"
        arrow = "▲" if pct > 0 else "▼" if pct < 0 else "▬"
        spans.append(
            f'<span class="ticker-item {css_class}">{symbol} {price} {arrow} {pct:+.2f}%</span>'
        )
    html = f'<div class="ticker-wrap"><div class="ticker-move">{"".join(spans)}</div></div>'
    st.markdown(html, unsafe_allow_html=True)


def colored_metric(label: str, value: str, delta: float | None = None, suffix: str = "%") -> None:
    """
    st.metric wrapper that colors the delta green/red like a terminal price feed,
    instead of Streamlit's default red-is-always-bad coloring.
    """
    if delta is None:
        st.metric(label, value)
        return
    delta_str = f"{delta:+.2f}{suffix}"
    color = "normal" if delta >= 0 else "inverse"
    st.metric(label, value, delta=delta_str, delta_color=color)


def panel_header(title: str, subtitle: str | None = None) -> None:
    """Small section header used at the top of each terminal panel/container."""
    st.markdown(f"### {title}")
    if subtitle:
        st.caption(subtitle.upper())

"""
ui.charts — Plotly chart helpers for the Bloomberg-terminal dashboard.

All functions return a go.Figure. Caller does st.plotly_chart(fig, use_container_width=True).
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

AMBER = "#FF9F1C"
GREEN = "#00D964"
RED   = "#FF3B30"
MUTED = "#8B949E"
BG    = "#000000"
PANEL = "#0D1117"
GRID  = "#2A2E35"
TEXT  = "#E6E6E6"

_FONT = "JetBrains Mono, IBM Plex Mono, Consolas, monospace"
_LINE_COLORS = [AMBER, GREEN, "#4FC3F7", "#CE93D8", "#EF9A9A", "#A5D6A7", "#FFF176", "#80CBC4"]


def _base(fig: go.Figure, height: int = 300) -> go.Figure:
    fig.update_layout(
        height=height,
        paper_bgcolor=BG,
        plot_bgcolor=PANEL,
        font=dict(family=_FONT, color=TEXT, size=11),
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis=dict(gridcolor=GRID, zerolinecolor=GRID),
        yaxis=dict(gridcolor=GRID, zerolinecolor=GRID),
        legend=dict(bgcolor=PANEL, bordercolor=GRID, font=dict(size=10)),
    )
    return fig


def efficient_frontier_chart(frontier_df: pd.DataFrame, optimal: dict | None = None) -> go.Figure:
    """
    Scatter of Monte-Carlo portfolios coloured by Sharpe ratio.
    optimal: dict with keys 'return' and 'volatility' (fractions, not percent).
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=frontier_df["volatility"] * 100,
        y=frontier_df["return"] * 100,
        mode="markers",
        name="Portfolios",
        marker=dict(
            color=frontier_df["sharpe"],
            colorscale=[[0, PANEL], [0.5, "#6B4500"], [1, AMBER]],
            size=3,
            opacity=0.6,
            colorbar=dict(title="Sharpe", thickness=10, tickfont=dict(size=9)),
        ),
        hovertemplate="Vol: %{x:.1f}%  Ret: %{y:.1f}%<extra></extra>",
    ))
    if optimal:
        ret = optimal.get("return", 0.0) * 100
        vol = optimal.get("volatility", 0.0) * 100
        fig.add_trace(go.Scatter(
            x=[vol], y=[ret],
            mode="markers",
            name="★ OPTIMAL",
            marker=dict(symbol="star", size=18, color=GREEN,
                        line=dict(color=TEXT, width=1)),
            hovertemplate=f"OPTIMAL  Vol:{vol:.1f}%  Ret:{ret:.1f}%<extra></extra>",
        ))
    fig.update_layout(
        xaxis=dict(title="VOLATILITY (%)", gridcolor=GRID),
        yaxis=dict(title="RETURN (%)", gridcolor=GRID),
    )
    return _base(fig, height=300)


def correlation_heatmap(corr_df: pd.DataFrame) -> go.Figure:
    """Annotated heatmap. Red = -1, dark = 0, green = +1."""
    labels = corr_df.columns.tolist()
    z = corr_df.values.round(2)
    fig = go.Figure(go.Heatmap(
        z=z,
        x=labels,
        y=labels,
        colorscale=[[0, RED], [0.5, PANEL], [1, GREEN]],
        zmin=-1, zmax=1,
        text=z,
        texttemplate="%{text:.2f}",
        textfont=dict(size=10, color=TEXT),
        showscale=True,
        colorbar=dict(thickness=10, tickfont=dict(size=9)),
    ))
    return _base(fig, height=300)


def cumulative_returns_chart(cum_returns: dict[str, pd.Series]) -> go.Figure:
    """
    Multi-line cumulative return chart.
    Values should be fractions (0.23 = +23%). PORTFOLIO key gets amber + thicker line.
    """
    fig = go.Figure()
    for i, (ticker, series) in enumerate(cum_returns.items()):
        is_port = ticker == "PORTFOLIO"
        color = AMBER if is_port else _LINE_COLORS[i % len(_LINE_COLORS)]
        fig.add_trace(go.Scatter(
            x=series.index,
            y=series.values,
            mode="lines",
            name=ticker,
            line=dict(color=color, width=2.5 if is_port else 1.5),
            hovertemplate=f"{ticker}: %{{y:.1%}}<extra></extra>",
        ))
    fig.update_layout(
        yaxis=dict(tickformat=".1%", gridcolor=GRID),
        xaxis=dict(gridcolor=GRID),
        hovermode="x unified",
    )
    return _base(fig, height=280)


def yield_curve_chart(yield_df: pd.DataFrame) -> go.Figure:
    """Treasury yield curve — filled line chart, amber colour."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=yield_df["maturity"],
        y=yield_df["yield_pct"],
        mode="lines+markers",
        line=dict(color=AMBER, width=2),
        marker=dict(size=8, color=AMBER),
        fill="tozeroy",
        fillcolor="rgba(255,159,28,0.10)",
        hovertemplate="%{x}: %{y:.2f}%<extra></extra>",
    ))
    fig.update_layout(
        yaxis=dict(title="YIELD (%)", ticksuffix="%", gridcolor=GRID),
        xaxis=dict(gridcolor=GRID),
    )
    return _base(fig, height=240)


def allocation_bar_chart(alloc_df: pd.DataFrame) -> go.Figure:
    """Horizontal bar chart of portfolio weights, sorted ascending for display."""
    df = alloc_df.sort_values("weight_%")
    fig = go.Figure(go.Bar(
        x=df["weight_%"],
        y=df["ticker"],
        orientation="h",
        marker=dict(color=AMBER, line=dict(color=GRID, width=1)),
        text=df["weight_%"].apply(lambda v: f"{v:.1f}%"),
        textposition="outside",
        textfont=dict(color=TEXT, size=10),
        hovertemplate="%{y}: %{x:.1f}%<extra></extra>",
    ))
    fig.update_layout(
        xaxis=dict(title="WEIGHT (%)", gridcolor=GRID),
        yaxis=dict(gridcolor=GRID),
        showlegend=False,
    )
    return _base(fig, height=max(200, len(df) * 36))

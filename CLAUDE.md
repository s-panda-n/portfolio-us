# Portfolio-US — Project Context for Claude Code

## What this project is
An AI-agent-based, multi-asset portfolio tracker built in Python. It pulls data from
multiple free/freemium sources, models diversification and risk/return tradeoffs, and
presents everything through a **Bloomberg-terminal-styled Streamlit dashboard** — dark
background, monospace font, amber/green accents, dense multi-panel layout.

This is a learning-driven build: the person building this is learning quant finance,
statistics, and portfolio math hands-on, in parallel with development. Study tasks are
tracked alongside build tasks. Keep explanations applied and concrete rather than academic
when discussing the "why" behind a technique — the goal is working code plus real
understanding, not lecture notes.

## Tech stack
- **Language**: Python 3.11+, virtual env named `port-env`
- **UI**: Streamlit, themed to look like a Bloomberg terminal (see Week 1/6 tasks)
- **Data sources**:
  - Equities/ETFs: `yfinance`
  - Bonds: FRED API (Treasury yields) + bond ETF proxies (AGG, TLT, SHY) via yfinance
  - Options (backlog/optional): Alpaca Market Data API (OPRA)
  - News/Sentiment: Finnhub (news + built-in sentiment endpoints)
- **Core libraries**: pandas, numpy, scipy (optimizer), plotly, pytest, python-dotenv
- **API keys**: stored in `.env` (git-ignored) — `ALPACA_API_KEY`, `ALPACA_SECRET_KEY`, `FINNHUB_API_KEY`

## Recommended repo structure
```
portfolio-us/
├── data/           # data agents: fetch + cache OHLCV, bonds, news
├── metrics/        # returns, Sharpe/Sortino/Calmar, drawdown, correlation
├── allocation/      # mean-variance optimizer, risk/return toggle logic
├── agents/          # orchestration layer tying data/allocation/sentiment together
├── ui/              # Streamlit app + Bloomberg-terminal theme (config.toml, custom CSS)
├── tests/
├── .env.example
├── .gitignore
├── requirements.txt
└── CLAUDE.md         # this file
```

## Scope boundaries — IMPORTANT
- Core MVP = **6 weeks**. Options, IPO tracking, derivatives, real-time paid data,
  backtesting, and meta-labeling are all in the **Backlog** — optional, no deadline,
  picked up only if/when requested. Do not pull backlog scope into the MVP path
  unless explicitly asked.
- The reference repo `virattt/ai-hedge-fund` (MIT licensed) is useful ONLY for its
  agent-orchestration pattern (see Week 5). Do not adopt its investor-persona design —
  it doesn't serve this project's diversification/allocation goal.

---

## Week-by-week plan

### Week 1 — Setup, Data Foundation, Terminal UI Shell
- [ ] Confirm repo, `port-env`, Claude Code all working end-to-end
- [ ] Sign up Alpaca + Finnhub, generate API keys, store in `.env` (git-ignored)
- [ ] STUDY: Linear algebra for portfolio math — vectors, matrix multiplication, covariance
      matrices (Michael Brenndoerfer "Linear Algebra for Quantitative Finance"; Quantt.co.uk)
- [ ] STUDY: Statistics for finance — correlation vs covariance, distributions, stationarity
      of returns (Quant Visualized, YouTube)
- [ ] BUILD: Data agent — yfinance OHLCV pull + local cache for watchlist
- [ ] BUILD: Metrics module — returns, Sharpe/Sortino/Calmar, max drawdown, correlation matrix
- [ ] BUILD: Bloomberg-terminal-style Streamlit theme — black bg, amber/green accents,
      monospace font, dense panel grid

### Week 2 — Allocation Engine
- [ ] STUDY: Portfolio variance & risk decomposition via covariance matrix — eigenvalues,
      positive semi-definiteness (Quantt.co.uk "Portfolio Theory and CAPM")
- [ ] STUDY: Constrained optimization — Lagrange multipliers, quadratic programming for
      Markowitz efficient frontier (Xilinx Vitis quant finance docs; Quantt.co.uk)
- [ ] BUILD: Mean-variance optimizer using scipy.optimize
- [ ] BUILD: Risk/return toggles mapped to optimizer constraints + capital-to-dollar
      allocation logic
- [ ] BUILD: Efficient frontier chart + portfolio composition panel in terminal-style UI

### Week 3 — Bonds / Fixed Income
- [ ] STUDY: Yield curve mechanics, duration/convexity — applied, not intro (Patrick Boyle,
      YouTube)
- [ ] STUDY: FRED API data structure for Treasury yield series
- [ ] BUILD: Bond agent — FRED Treasury yields + bond ETF proxies (AGG/TLT/SHY) via yfinance
- [ ] BUILD: Integrate bonds into asset universe + add bond panel to terminal UI

### Week 4 — News & Sentiment
- [ ] STUDY: How sentiment scoring models work in finance (FinBERT/Finnhub) — applied
      overview, skip NLP fundamentals
- [ ] BUILD: Finnhub news + sentiment integration for watchlist tickers
- [ ] BUILD: Scrolling news/sentiment panel in terminal UI (Bloomberg-style ticker feed)

### Week 5 — Orchestration + Risk Deep-Dive
- [ ] STUDY: Risk metrics deep dive — VaR, CVaR, tail risk, why Sharpe alone is insufficient
      (Dimitri Bianco, YouTube)
- [ ] STUDY: Review `virattt/ai-hedge-fund` `src/agents/` orchestration pattern
      (LangGraph-style) — architecture ideas ONLY, not the investor-persona design
- [ ] BUILD: Orchestration layer combining data + allocation + bond + sentiment agents
- [ ] BUILD: Reasoning/transparency panel — terminal command-output style explanation of
      the allocation decision

### Week 6 — Terminal UI Polish + MVP Ship
- [ ] BUILD: Full terminal UI polish pass — multi-panel grid, color-coded up/down, ticker
      search command bar
- [ ] BUILD: End-to-end test — capital + risk/return toggles produce full portfolio
      suggestion with reasoning shown
- [ ] BUILD: Write README + record short demo
- [ ] MILESTONE: **MVP COMPLETE**

---

## Backlog (optional modules — no deadline, pick up anytime)
- Options module: Black-Scholes + Greeks + Alpaca options chains (OPRA) + payoff diagrams
  (Quant Visualized options playlist; Hull ch.1-3 when ready)
- IPO tracker module — IPO calendar aggregation, S-1 basics
- Derivatives/futures module — contango/backwardation, margin (Patrick Boyle, light)
- Upgrade to real-time paid data feed (Massive/Polygon or Alpaca paid tier)
- Backtesting engine integration
- Meta-labeling / regime detection exploration (Lopez de Prado, *Advances in Financial
  Machine Learning*)

---

## Working conventions for Claude Code
- Check off `[ ]` → `[x]` in this file as tasks are completed, in the same commit as the
  code that completes them.
- When starting a new session, read this file first to know current week/status before
  proposing work.
- Prefer small, focused commits per task rather than one big commit per week.

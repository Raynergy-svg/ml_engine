"""Equity daily OHLCV data layer for the equity-beta harvester.

The harvester needs point-in-time, dividend/split adjusted daily prices for a
liquid US equity universe. Every downstream story (factor signals, beta hedge,
gate, paper-trade) reads from this loader, so the contract is:

- Point-in-time: rows are dated by close-of-session and never look ahead.
- Adjusted: OHLCV are split/dividend adjusted at the source (IBKR adjusted=True
  or yfinance ``auto_adjust=True``); we store what we got and tag the source.
- Atomic: cache writes go to ``.tmp`` then ``os.replace``; readers never see a
  half-written file.
- Fail-loud: stale (last row older than freshness window) or NaN-bearing data
  raises ``EquityDataError`` — no silent forward-fill, no empty frame.
- Source-explicit: every cache CSV has a JSON sidecar recording which live
  source produced it (``ibkr`` preferred for live trading; ``yfinance`` for
  backtest only).
"""

from __future__ import annotations

EQUITY_PIPELINE_VERSION = "2026-06-18-eq1"

# A small, liquid, well-covered default universe. The loader accepts any
# ticker; this list exists so US-002+ have a default to iterate.
DEFAULT_UNIVERSE = [
    "SPY",
    "QQQ",
    "IWM",
    "AAPL",
    "MSFT",
    "GOOGL",
    "AMZN",
    "META",
    "NVDA",
    "JPM",
]

__all__ = ["EQUITY_PIPELINE_VERSION", "DEFAULT_UNIVERSE"]

# US-002 universe builder symbols are accessible via src.equity.universe; we
# avoid importing them at package init to keep the data layer dependency-free
# for downstream packages that only need EQUITY_PIPELINE_VERSION.
#
# US-003 backtest harness lives in src.equity.backtest and exposes
# `stats`, `overlay`, `BacktestResult`, `run_portfolio_backtest`, `save_result`.
# It is imported on-demand by the runtime / scripts (not at package init).

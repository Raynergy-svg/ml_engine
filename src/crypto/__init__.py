"""Crypto research data layer + experiments (research/backtest only).

PAPER/RESEARCH ONLY. No live execution, no real money, no order path. This package
exists solely to feed the pre-registered, gated edge-hunt harness (see
``docs/experiment-crypto-*.md``) with free, survivorship-aware crypto data.

Data sources (all free, no API keys):
  - Binance static dumps (``data.binance.vision``): funding (8h) + klines, monthly
    archives from 2020-01, full symbol set INCLUDING delisted -> survivorship-aware.
    NOTE: the static dump bucket is reachable here even though the trading API is
    geo-blocked from this environment.
  - OKX public REST: OHLCV history (to 2020) + recent funding (~3mo retention).
    Independent CEX cross-check.
  - Hyperliquid public REST: hourly funding from 2023-07 (full retention), on-chain
    perp DEX. Independent venue + mechanism cross-check.
"""

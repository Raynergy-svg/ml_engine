"""Daily FX factor portfolio (carry + trend + value).

Operator-approved strategy pivot (2026-06-12) away from the no-edge intraday
direction stack. See ``tasks/prd-fx-factor-portfolio.md``.

This package is deliberately small, dependency-light (pandas/numpy only), and
fully deterministic: every signal is causal (uses only data available at the
signal date), every artifact is versioned and written atomically, and there is
no LLM anywhere in the decision path.
"""

from __future__ import annotations

# Bump on ANY change to how signals/weights are computed so cross-version
# backtest artifacts are never silently compared. (Mirrors FEATURE_PIPELINE_VERSION.)
FACTOR_PIPELINE_VERSION = "2026-06-12-fp1"

# The G10 USD majors the portfolio trades. USD-quote convention varies per pair;
# the loader stores raw OANDA mid candles and the signal layer is sign-aware.
PAIRS = [
    "EUR_USD",
    "USD_JPY",
    "GBP_USD",
    "AUD_USD",
    "NZD_USD",
    "USD_CAD",
    "USD_CHF",
]

# FP-1: liquid G10 CROSSES (no USD leg) — they give carry/value a real
# cross-section instead of 7 trades that all share the USD as one leg (where the
# rank degenerates into a USD-cycle bet). Every currency here has a short-rate
# series in rates.FRED_RATE_IDS, so carry differentials are computable for all.
CROSSES = [
    "EUR_JPY",
    "GBP_JPY",
    "AUD_JPY",
    "NZD_JPY",
    "CAD_JPY",
    "CHF_JPY",
    "EUR_GBP",
    "EUR_CHF",
    "EUR_AUD",
    "GBP_AUD",
    "AUD_NZD",
    "EUR_CAD",
]

# Full default trading universe for the breadth backtest.
ALL_PAIRS = PAIRS + CROSSES

__all__ = ["FACTOR_PIPELINE_VERSION", "PAIRS", "CROSSES", "ALL_PAIRS"]

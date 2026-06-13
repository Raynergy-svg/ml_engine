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

__all__ = ["FACTOR_PIPELINE_VERSION", "PAIRS"]

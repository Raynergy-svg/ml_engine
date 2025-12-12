"""Forex paper trading helpers (OANDA practice).

This module provides:
- conversion of OANDA candle JSON into a DataFrame
- simple, conservative risk rules for position sizing
- a minimal rule-based "setup" signal (placeholder until ML target is trained)

Important: This is for OANDA PRACTICE only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Optional

import numpy as np
import pandas as pd


Signal = Literal["buy", "sell", "hold"]


def candles_to_ohlcv_df(oanda_candles_response: Any) -> pd.DataFrame:
    candles = oanda_candles_response.get("candles", [])
    if not candles:
        raise ValueError("No candles in OANDA response")

    rows: list[dict[str, Any]] = []
    for c in candles:
        if not c.get("complete", True):
            continue
        mid = c.get("mid") or {}
        rows.append(
            {
                "time": c.get("time"),
                "open": float(mid.get("o")),
                "high": float(mid.get("h")),
                "low": float(mid.get("l")),
                "close": float(mid.get("c")),
                "volume": float(c.get("volume", 0.0)),
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError("No complete candles available")

    return df


def atr(df: pd.DataFrame, period: int = 14) -> float:
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)

    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    atr_series = tr.rolling(period).mean()
    value = float(atr_series.iloc[-1])
    if not np.isfinite(value) or value <= 0:
        value = float(tr.iloc[-period:].mean())
    return float(value)


def pip_size(instrument: str) -> float:
    # Simple default: JPY pairs use 0.01, most others 0.0001
    return 0.01 if instrument.endswith("_JPY") else 0.0001


@dataclass(frozen=True)
class RiskRules:
    equity: float
    risk_per_trade_pct: float = 0.005  # 0.5%
    max_daily_loss_pct: float = 0.02
    max_open_positions: int = 1
    atr_stop_mult: float = 1.5
    rr_take_profit: float = 1.5  # risk-reward


def position_size_units(
    *,
    instrument: str,
    equity: float,
    risk_per_trade_pct: float,
    stop_distance_price: float,
    price: float,
    pip_value_per_unit: Optional[float] = None,
) -> int:
    """Compute position size in units, conservative approximation.

    If pip_value_per_unit is unknown, assume 1 unit moves ~1 quote unit per 1.0 price move.
    That's not exact across pairs; for practice this is a starting point.
    """
    risk_amount = float(equity) * float(risk_per_trade_pct)
    if stop_distance_price <= 0:
        raise ValueError("stop_distance_price must be positive")

    if pip_value_per_unit is None:
        # Approximate value per unit for a price move.
        value_per_unit_per_price = 1.0
    else:
        value_per_unit_per_price = float(pip_value_per_unit) / pip_size(instrument)

    units = risk_amount / (stop_distance_price * value_per_unit_per_price)
    # round down to integer units
    return int(max(1, np.floor(units)))


def setup_signal(df: pd.DataFrame) -> Signal:
    """A simple setup-based signal (placeholder until ML setup model is trained).

    - trend: close vs EMA(20)
    - momentum: RSI(14)
    """
    close = df["close"].astype(float)

    ema_20 = close.ewm(span=20, adjust=False).mean()
    delta = close.diff()
    up = delta.clip(lower=0)
    down = (-delta).clip(lower=0)
    rs = up.rolling(14).mean() / (down.rolling(14).mean() + 1e-12)
    rsi_14 = 100 - (100 / (1 + rs))

    last_close = float(close.iloc[-1])
    last_ema = float(ema_20.iloc[-1])
    last_rsi = float(rsi_14.iloc[-1]) if np.isfinite(float(rsi_14.iloc[-1])) else 50.0

    if last_close > last_ema and 45 <= last_rsi <= 70:
        return "buy"
    if last_close < last_ema and 30 <= last_rsi <= 55:
        return "sell"
    return "hold"

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

from candle_smoothing import resample_5min_ohlcv_and_ema_close


Signal = Literal["buy", "sell", "hold"]


TpSlHitType = Literal["tp", "sl", "timeout"]


@dataclass(frozen=True)
class TpSlResult:
    """Result from TP/SL simulation with next-candle entry.

    - outcome: 1 for TP hit, 0 for SL/timeout
    - hit_type: which event occurred (tp/sl/timeout)
    - exit_price: realized exit price (tp/sl level or timeout close)
    - bars_held: candles between entry candle and exit candle
    - entry_price: spread-adjusted entry price
    """

    outcome: int
    hit_type: TpSlHitType
    exit_price: float
    bars_held: int
    entry_price: float

    def to_label(self) -> int:
        return int(self.outcome)

    @property
    def is_win(self) -> bool:
        return bool(int(self.outcome) == 1)

    def __repr__(self) -> str:
        status = "WIN" if self.is_win else "LOSS"
        return f"TpSlResult({status}, {self.hit_type}, exit={self.exit_price:.5f}, held={int(self.bars_held)})"


def simulate_tp_sl_outcome(
    df: pd.DataFrame,
    *,
    instrument: str,
    signal_index: int,
    direction: Literal["long", "short"],
    stop_loss_pips: float,
    take_profit_pips: float,
    spread_pips: float = 0.0,
    max_horizon_candles: int | None = None,
    tie_break: Literal["sl", "tp"] = "sl",
) -> TpSlResult:
    """Simulate TP-before-SL with next-candle entry, returning a conservative outcome.

    This is designed for calibration/backtesting when only candle OHLC is available.

        Timeline:
            - signal_index: signal fires at close of this candle
            - signal_index+1: enter at open of next candle (spread-adjusted)
            - signal_index+2+: first candle that can hit TP/SL

        - Entry price is adjusted against you by `spread_pips` (conservative):
            - long: entry += spread
            - short: entry -= spread
    - Intrabar ambiguity (both TP and SL inside same candle): resolved by `tie_break`.
      For Tier-2 calibration we default to worst-case: `tie_break='sl'`.
    """
    if "open" not in df.columns:
        raise ValueError("simulate_tp_sl_outcome requires 'open' column for next-candle entry")

    if signal_index < 0 or signal_index >= len(df):
        raise IndexError("signal_index out of range")

    # Need entry candle (signal_index+1) and at least one scan candle (signal_index+2).
    if int(signal_index) + 2 >= len(df):
        raise ValueError(
            f"Insufficient data: need signal_index+2 < len(df)={len(df)}, got signal_index={int(signal_index)}"
        )

    pip = float(pip_size(instrument))
    if stop_loss_pips <= 0 or take_profit_pips <= 0:
        raise ValueError("stop_loss_pips and take_profit_pips must be positive")

    try:
        spread_pips_f = float(spread_pips)
    except Exception:
        raise ValueError("Invalid spread_pips")
    if not np.isfinite(spread_pips_f) or spread_pips_f < 0:
        raise ValueError(f"Invalid spread_pips={spread_pips_f}")
    # Clamp tiny positive spreads up to 0.5 pips; reject extreme outliers.
    if 0.0 < spread_pips_f < 0.5:
        spread_pips_f = 0.5
    # Allow wider spreads for exotic pairs or volatile conditions (up to 50 pips)
    if spread_pips_f > 50.0:
        raise ValueError(f"Invalid spread_pips={spread_pips_f} (expect <= 50)")

    entry_open = float(df["open"].iloc[int(signal_index) + 1])
    if not np.isfinite(entry_open) or entry_open <= 0:
        raise ValueError(f"Invalid entry open price at index {int(signal_index)+1}")

    spread_px = float(spread_pips_f) * pip
    if direction == "long":
        entry_price = entry_open + spread_px
        stop_price = entry_price - (float(stop_loss_pips) * pip)
        take_profit_price = entry_price + (float(take_profit_pips) * pip)
    else:
        entry_price = entry_open - spread_px
        stop_price = entry_price + (float(stop_loss_pips) * pip)
        take_profit_price = entry_price - (float(take_profit_pips) * pip)

    start = int(signal_index) + 2
    if max_horizon_candles is None:
        end = len(df) - 1
    else:
        end = min(len(df) - 1, int(signal_index) + 1 + int(max_horizon_candles))

    if start > end:
        # No full candle after entry to test; treat as no-op timeout.
        return TpSlResult(
            outcome=0,
            hit_type="timeout",
            exit_price=float(entry_price),
            bars_held=0,
            entry_price=float(entry_price),
        )

    tie_break = "sl" if str(tie_break).lower().strip() != "tp" else "tp"

    for i in range(start, end + 1):
        high = float(df["high"].iloc[i])
        low = float(df["low"].iloc[i])

        if direction == "long":
            hit_tp = np.isfinite(high) and high >= take_profit_price
            hit_sl = np.isfinite(low) and low <= stop_price
        else:
            hit_tp = np.isfinite(low) and low <= take_profit_price
            hit_sl = np.isfinite(high) and high >= stop_price

        if hit_tp and hit_sl:
            hit_type: TpSlHitType = "sl" if tie_break == "sl" else "tp"
        elif hit_sl:
            hit_type = "sl"
        elif hit_tp:
            hit_type = "tp"
        else:
            continue

        exit_price = float(stop_price) if hit_type == "sl" else float(take_profit_price)
        outcome = 1 if hit_type == "tp" else 0
        return TpSlResult(
            outcome=int(outcome),
            hit_type=hit_type,
            exit_price=float(exit_price),
            bars_held=int(i - (int(signal_index) + 1)),
            entry_price=float(entry_price),
        )

    exit_price = float(df["close"].iloc[int(end)])
    return TpSlResult(
        outcome=0,
        hit_type="timeout",
        exit_price=float(exit_price),
        bars_held=int(end - (int(signal_index) + 1)),
        entry_price=float(entry_price),
    )


def _safe_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def _extract_ohlc(d: dict[str, Any]) -> tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
    return (
        _safe_float(d.get("o")),
        _safe_float(d.get("h")),
        _safe_float(d.get("l")),
        _safe_float(d.get("c")),
    )


def _mid_close_from_bid_ask(bid_c: Optional[float], ask_c: Optional[float]) -> Optional[float]:
    if bid_c is None or ask_c is None:
        return None
    return (bid_c + ask_c) / 2.0


def _fill_ohl_with_close(
    open_: Optional[float],
    high: Optional[float],
    low: Optional[float],
    close: Optional[float],
) -> tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
    if close is None:
        return open_, high, low, close
    return (
        close if open_ is None else open_,
        close if high is None else high,
        close if low is None else low,
        close,
    )


def _parse_oanda_candle_row(candle: dict[str, Any]) -> Optional[dict[str, Any]]:
    if not candle.get("complete", True):
        return None

    mid = candle.get("mid") or {}
    bid = candle.get("bid") or {}
    ask = candle.get("ask") or {}

    # Prefer mid for OHLCV, but preserve bid/ask closes when available for spread checks.
    open_, high, low, close = _extract_ohlc(mid)
    bid_c = _safe_float(bid.get("c"))
    ask_c = _safe_float(ask.get("c"))

    close = close if close is not None else _mid_close_from_bid_ask(bid_c, ask_c)
    open_, high, low, close = _fill_ohl_with_close(open_, high, low, close)

    return {
        "time": candle.get("time"),
        "open": float(open_) if open_ is not None else float("nan"),
        "high": float(high) if high is not None else float("nan"),
        "low": float(low) if low is not None else float("nan"),
        "close": float(close) if close is not None else float("nan"),
        "volume": float(candle.get("volume", 0.0)),
        "bid_close": float(bid_c) if bid_c is not None else np.nan,
        "ask_close": float(ask_c) if ask_c is not None else np.nan,
    }


def candles_to_ohlcv_df(oanda_candles_response: Any) -> pd.DataFrame:
    candles = oanda_candles_response.get("candles", [])
    if not candles:
        raise ValueError("No candles in OANDA response")

    rows: list[dict[str, Any]] = []
    for c in candles:
        row = _parse_oanda_candle_row(c)
        if row is not None:
            rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError("No complete candles available")

    # Drop rows with missing core OHLC.
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["open", "high", "low", "close"])
    if df.empty:
        raise ValueError("No usable candles after cleaning")

    # Back-compat: only keep bid/ask columns if present in data.
    for col in ("bid_close", "ask_close"):
        if col in df.columns and df[col].isna().all():
            df = df.drop(columns=[col])

    # Smooth candles: resample to 5-minute bars and EMA(14) the close.
    # This ensures downstream models don't see raw close series.
    return resample_5min_ohlcv_and_ema_close(df, ema_span=14, time_col="time")


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


def spread_pips_from_df(df: pd.DataFrame, instrument: str) -> Optional[float]:
    """Compute spread (ask - bid) in pips from the last candle, if available."""
    if "bid_close" not in df.columns or "ask_close" not in df.columns:
        return None
    bid = float(df["bid_close"].iloc[-1])
    ask = float(df["ask_close"].iloc[-1])
    if not np.isfinite(bid) or not np.isfinite(ask) or bid <= 0 or ask <= 0:
        return None
    return (ask - bid) / pip_size(instrument)


def conservative_slippage_pips(*, spread_pips: float, min_pips: float, spread_mult: float) -> float:
    """Conservative slippage buffer (pips) applied against you."""
    return float(max(float(min_pips), float(spread_mult) * float(spread_pips)))


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
# — Raynergy-svg —

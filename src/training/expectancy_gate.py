"""Cost-aware expectancy gate — the promotion-time answer to "is it profitable?".

Raw directional accuracy cannot answer the only question that matters: an
all-SHORT predictor scores ~70% on a SHORT-heavy holdout slice while being
untradeable after costs (the May-2026 incident). This module simulates the
holdout predictions as ATR-bracketed trades with real spread costs and gates
promotion on per-trade expectancy + balanced accuracy + class-collapse.

Design notes:
- Path-aware bracket simulation: each prediction enters at the bar close with
  SL = ATR*sl_mult and TP = ATR*tp_mult (mirroring the live execution layer's
  ATR-based SL/TP invariant); subsequent bars' high/low decide first-touch.
  When SL and TP are both touchable within one bar the SL is assumed first
  (pessimistic — we'd rather under-promote than over-promote).
- Spread is charged once per trade, in pips, at entry. Defaults are
  conservative OANDA-practice spreads; pass per-pair values where known.
- Fail closed: too few trades is a REJECT, not a pass-with-warning
  (the 2/3-pass deploy gate that shipped val_acc=0.4839 must not recur).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Conservative default spreads (pips) — OANDA practice, typical session.
DEFAULT_SPREAD_PIPS = {
    "EUR_USD": 1.0, "GBP_USD": 1.4, "USD_JPY": 1.2, "USD_CHF": 1.6,
    "AUD_USD": 1.4, "USD_CAD": 1.6, "NZD_USD": 1.8, "EUR_GBP": 1.6,
    "EUR_JPY": 1.6, "GBP_JPY": 2.2, "AUD_NZD": 2.4, "EUR_AUD": 2.4,
    "GBP_CHF": 2.8, "AUD_JPY": 1.8, "EUR_CHF": 2.0, "CAD_JPY": 2.0,
}
FALLBACK_SPREAD_PIPS = 2.0


def pip_size_for(instrument: str) -> float:
    """0.01 for JPY-quoted pairs, 0.0001 otherwise."""
    return 0.01 if instrument.upper().endswith("JPY") else 0.0001


def spread_for(instrument: str) -> float:
    return DEFAULT_SPREAD_PIPS.get(instrument.upper(), FALLBACK_SPREAD_PIPS)


@dataclass
class ExpectancyReport:
    """Cost-aware simulation summary for one prediction set."""
    n_trades: int = 0
    wins: int = 0
    losses: int = 0
    win_rate: float = 0.0
    balanced_accuracy: float = 0.0
    long_share: float = 0.0
    short_share: float = 0.0
    class_collapse: bool = False
    expectancy_pips: float = 0.0
    profit_factor: float = 0.0
    avg_win_pips: float = 0.0
    avg_loss_pips: float = 0.0
    spread_pips: float = 0.0
    trade_pips: List[float] = field(default_factory=list)

    def as_metrics(self) -> dict:
        """Flat dict for promotion-policy / W&B consumption."""
        return {
            "n_trades": self.n_trades,
            "win_rate": round(self.win_rate, 4),
            "balanced_accuracy": round(self.balanced_accuracy, 4),
            "long_share": round(self.long_share, 4),
            "short_share": round(self.short_share, 4),
            "class_collapse": self.class_collapse,
            "expectancy_pips": round(self.expectancy_pips, 3),
            "profit_factor": round(self.profit_factor, 3),
            "spread_pips": self.spread_pips,
        }


def _atr(df: pd.DataFrame, period: int = 14) -> np.ndarray:
    """Simple ATR over OHLC (Wilder-free rolling mean — sufficient for sim)."""
    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    close = df["close"].to_numpy(dtype=float)
    prev_close = np.concatenate([[close[0]], close[:-1]])
    tr = np.maximum(high - low, np.maximum(np.abs(high - prev_close), np.abs(low - prev_close)))
    atr = pd.Series(tr).rolling(period, min_periods=1).mean().to_numpy()
    return atr


def simulate_trades(
    df: pd.DataFrame,
    predictions: Sequence[Tuple[int, str]],
    *,
    spread_pips: float,
    pip_size: float,
    sl_atr_mult: float = 1.5,
    tp_atr_mult: float = 2.0,
    atr_period: int = 14,
    max_hold_bars: int = 24,
    max_class_share: float = 0.85,
) -> ExpectancyReport:
    """Simulate predictions as ATR-bracketed trades with spread costs.

    Args:
        df: OHLC DataFrame (open/high/low/close), chronological.
        predictions: (bar_index, "LONG"|"SHORT") pairs; entry at close[bar_index].
        spread_pips: round-trip cost charged once per trade, in pips.
        pip_size: 0.0001 (most) or 0.01 (JPY-quoted).
        sl_atr_mult / tp_atr_mult: bracket distances in ATR multiples
            (defaults mirror live execution; R:R = 2.0/1.5 ≈ 1.33 ≥ 1.2 gate).
        max_hold_bars: time-stop — exit at close if neither bracket hit.
        max_class_share: prediction-share above which class_collapse trips.

    Returns:
        ExpectancyReport. Trades whose forward window runs off the end of df
        are skipped (no peeking, no padding).
    """
    closes = df["close"].to_numpy(dtype=float)
    highs = df["high"].to_numpy(dtype=float)
    lows = df["low"].to_numpy(dtype=float)
    atr = _atr(df, atr_period)
    n = len(df)

    pips: List[float] = []
    n_long = 0
    n_short = 0
    # Direction-correctness ledger for balanced accuracy:
    # actual direction over the hold window, keyed by actual class.
    actual_long_total = 0
    actual_long_correct = 0
    actual_short_total = 0
    actual_short_correct = 0

    for idx, direction in predictions:
        direction = str(direction).upper()
        if direction not in ("LONG", "SHORT"):
            continue
        if idx < 0 or idx + 1 >= n:
            continue
        end = min(idx + max_hold_bars, n - 1)
        if end <= idx:
            continue

        entry = closes[idx]
        bracket = atr[idx]
        if not np.isfinite(bracket) or bracket <= 0:
            continue
        sl_dist = bracket * sl_atr_mult
        tp_dist = bracket * tp_atr_mult

        # Balanced-accuracy ledger (direction over the full hold window).
        actual_up = closes[end] > entry
        if actual_up:
            actual_long_total += 1
            if direction == "LONG":
                actual_long_correct += 1
        else:
            actual_short_total += 1
            if direction == "SHORT":
                actual_short_correct += 1

        if direction == "LONG":
            n_long += 1
            sl_price, tp_price = entry - sl_dist, entry + tp_dist
        else:
            n_short += 1
            sl_price, tp_price = entry + sl_dist, entry - tp_dist

        gross_pips: Optional[float] = None
        for j in range(idx + 1, end + 1):
            if direction == "LONG":
                hit_sl = lows[j] <= sl_price
                hit_tp = highs[j] >= tp_price
            else:
                hit_sl = highs[j] >= sl_price
                hit_tp = lows[j] <= tp_price
            if hit_sl:  # pessimistic: SL wins ties within a bar
                gross_pips = -sl_dist / pip_size
                break
            if hit_tp:
                gross_pips = tp_dist / pip_size
                break
        if gross_pips is None:  # time-stop exit at window close
            move = closes[end] - entry
            gross_pips = (move if direction == "LONG" else -move) / pip_size

        pips.append(gross_pips - spread_pips)

    report = ExpectancyReport(spread_pips=spread_pips)
    report.n_trades = len(pips)
    if not pips:
        return report

    arr = np.asarray(pips)
    wins = arr[arr > 0]
    losses = arr[arr <= 0]
    report.wins = int(len(wins))
    report.losses = int(len(losses))
    report.win_rate = len(wins) / len(arr)
    report.expectancy_pips = float(arr.mean())
    report.avg_win_pips = float(wins.mean()) if len(wins) else 0.0
    report.avg_loss_pips = float(losses.mean()) if len(losses) else 0.0
    gross_profit = float(wins.sum()) if len(wins) else 0.0
    gross_loss = float(-losses.sum()) if len(losses) else 0.0
    report.profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float("inf")
    report.trade_pips = [float(p) for p in arr]

    total_dir = n_long + n_short
    report.long_share = n_long / total_dir if total_dir else 0.0
    report.short_share = n_short / total_dir if total_dir else 0.0
    report.class_collapse = max(report.long_share, report.short_share) > max_class_share

    # Balanced accuracy = mean per-actual-class recall. Degenerate one-class
    # actuals fall back to the observed class recall (flagged via collapse).
    recalls = []
    if actual_long_total:
        recalls.append(actual_long_correct / actual_long_total)
    if actual_short_total:
        recalls.append(actual_short_correct / actual_short_total)
    report.balanced_accuracy = float(np.mean(recalls)) if recalls else 0.0
    return report


def combine_reports(reports: Iterable[ExpectancyReport]) -> ExpectancyReport:
    """Merge per-pair reports into one (trade-weighted) aggregate.

    Expectancy/profit-factor recompute exactly from the concatenated trade
    pips; balanced accuracy and class shares are n_trades-weighted means
    (adequate for gating; per-pair detail stays in the individual reports).
    """
    reports = [r for r in reports if r.n_trades > 0]
    combined = ExpectancyReport()
    if not reports:
        return combined

    all_pips: List[float] = []
    for r in reports:
        all_pips.extend(r.trade_pips)
    arr = np.asarray(all_pips, dtype=float)
    wins = arr[arr > 0]
    losses = arr[arr <= 0]

    total = sum(r.n_trades for r in reports)
    combined.n_trades = total
    combined.trade_pips = [float(p) for p in arr]
    combined.wins = int(len(wins))
    combined.losses = int(len(losses))
    combined.win_rate = len(wins) / len(arr) if len(arr) else 0.0
    combined.expectancy_pips = float(arr.mean()) if len(arr) else 0.0
    combined.avg_win_pips = float(wins.mean()) if len(wins) else 0.0
    combined.avg_loss_pips = float(losses.mean()) if len(losses) else 0.0
    gross_loss = float(-losses.sum()) if len(losses) else 0.0
    combined.profit_factor = (
        (float(wins.sum()) / gross_loss) if gross_loss > 0 else float("inf")
    )
    combined.balanced_accuracy = sum(
        r.balanced_accuracy * r.n_trades for r in reports
    ) / total
    combined.long_share = sum(r.long_share * r.n_trades for r in reports) / total
    combined.short_share = sum(r.short_share * r.n_trades for r in reports) / total
    combined.class_collapse = max(combined.long_share, combined.short_share) > 0.85
    combined.spread_pips = sum(r.spread_pips * r.n_trades for r in reports) / total
    return combined


def gate_decision(
    report: ExpectancyReport,
    *,
    min_trades: int = 20,
    min_balanced_accuracy: float = 0.52,
    min_expectancy_pips: float = 0.0,
    max_class_share: float = 0.85,
) -> Tuple[bool, List[str]]:
    """Fail-closed promotion verdict from an ExpectancyReport.

    Returns (passed, reasons). Every failed criterion contributes a reason;
    an under-sampled report is a REJECT (absence of evidence != fitness).
    """
    reasons: List[str] = []
    if report.n_trades < min_trades:
        reasons.append(
            f"insufficient trades ({report.n_trades} < {min_trades}) — fail closed"
        )
    if max(report.long_share, report.short_share) > max_class_share:
        reasons.append(
            f"class collapse: long_share={report.long_share:.0%} "
            f"short_share={report.short_share:.0%} (max {max_class_share:.0%})"
        )
    if report.balanced_accuracy < min_balanced_accuracy:
        reasons.append(
            f"balanced_accuracy {report.balanced_accuracy:.3f} < {min_balanced_accuracy:.2f}"
        )
    if report.expectancy_pips < min_expectancy_pips:
        reasons.append(
            f"expectancy {report.expectancy_pips:+.2f} pips/trade < "
            f"{min_expectancy_pips:+.2f} (not profitable after {report.spread_pips} pip spread)"
        )
    return (len(reasons) == 0), reasons

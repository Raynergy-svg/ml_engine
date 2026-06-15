"""
Error correlation and conditional performance analysis.

Answers the second question that matters:
  "Does the SOTA model win where the legacy model loses?"

If yes → errors are decorrelated → stacking yields diversification benefit.
If no → errors are correlated → stacking just doubles down on the same mistakes.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .model_comparison import BarPrediction, ModelComparisonResult

logger = logging.getLogger(__name__)


@dataclass
class RegimeSlice:
    """Performance metrics for a specific market regime slice."""
    regime: str
    n_bars: int
    legacy_accuracy: float = 0.0
    sota_accuracy: float = 0.0
    legacy_pnl: float = 0.0
    sota_pnl: float = 0.0
    conditional_lift: float = 0.0  # SOTA accuracy when legacy is WRONG


@dataclass
class ErrorCorrelation:
    """Deep analysis of where and why models disagree."""
    pair: str
    n_bars: int

    # Overall
    error_correlation: float = 0.0
    prediction_correlation: float = 0.0
    win_overlap: float = 0.0
    loss_overlap: float = 0.0

    # Conditional: does SOTA rescue legacy's losses?
    sota_when_legacy_right: float = 0.0   # SOTA accuracy | legacy correct
    sota_when_legacy_wrong: float = 0.0   # SOTA accuracy | legacy WRONG
    legacy_when_sota_right: float = 0.0
    legacy_when_sota_wrong: float = 0.0

    # Direction-specific
    long_lift: float = 0.0   # SOTA long accuracy - legacy long accuracy
    short_lift: float = 0.0  # SOTA short accuracy - legacy short accuracy

    # Regime slices
    regime_slices: List[RegimeSlice] = field(default_factory=list)

    # Decision
    recommendation: str = "UNDECIDED"  # STACK / REPLACE / DISCARD
    confidence: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pair": self.pair,
            "n_bars": self.n_bars,
            "error_correlation": round(self.error_correlation, 4),
            "prediction_correlation": round(self.prediction_correlation, 4),
            "win_overlap": round(self.win_overlap, 4),
            "loss_overlap": round(self.loss_overlap, 4),
            "sota_when_legacy_right": round(self.sota_when_legacy_right, 4),
            "sota_when_legacy_wrong": round(self.sota_when_legacy_wrong, 4),
            "long_lift": round(self.long_lift, 4),
            "short_lift": round(self.short_lift, 4),
            "regime_slices": [s.__dict__ for s in self.regime_slices],
            "recommendation": self.recommendation,
            "confidence": round(self.confidence, 4),
        }


def _direction_correct(pred_dir: str, actual_ret: float) -> bool:
    if pred_dir == "LONG":
        return actual_ret > 0
    elif pred_dir == "SHORT":
        return actual_ret < 0
    return False


def analyze_error_correlation(
    result: ModelComparisonResult,
    regime_labels: Optional[List[str]] = None,
) -> ErrorCorrelation:
    """Deep-dive: where do the models disagree, and who is right?"""
    ec = ErrorCorrelation(pair=result.pair, n_bars=result.n_bars)
    ec.error_correlation = result.error_correlation
    ec.prediction_correlation = result.prediction_correlation
    ec.win_overlap = result.win_overlap

    leg_preds = result.legacy_predictions
    sota_preds = result.sota_predictions
    actuals = result.actual_returns

    if len(leg_preds) == 0:
        return ec

    # --- Error flags ---
    leg_correct = np.array([_direction_correct(p.direction, a) for p, a in zip(leg_preds, actuals)])
    sota_correct = np.array([_direction_correct(p.direction, a) for p, a in zip(sota_preds, actuals)])
    both_wrong = np.logical_and(~leg_correct, ~sota_correct)
    ec.loss_overlap = float(both_wrong.sum() / len(both_wrong)) if len(both_wrong) > 0 else 0.0

    # --- Conditional accuracy ---
    # SOTA accuracy WHEN legacy got it right
    legacy_right_mask = leg_correct
    if legacy_right_mask.sum() > 0:
        ec.sota_when_legacy_right = float(sota_correct[legacy_right_mask].mean())

    # SOTA accuracy WHEN legacy got it WRONG (the rescue metric)
    legacy_wrong_mask = ~leg_correct
    if legacy_wrong_mask.sum() > 0:
        ec.sota_when_legacy_wrong = float(sota_correct[legacy_wrong_mask].mean())

    # Legacy accuracy WHEN SOTA got it right
    sota_right_mask = sota_correct
    if sota_right_mask.sum() > 0:
        ec.legacy_when_sota_right = float(leg_correct[sota_right_mask].mean())

    # Legacy accuracy WHEN SOTA got it WRONG
    sota_wrong_mask = ~sota_correct
    if sota_wrong_mask.sum() > 0:
        ec.legacy_when_sota_wrong = float(leg_correct[sota_wrong_mask].mean())

    # --- Direction lift ---
    ec.long_lift = result.sota_long_accuracy - result.legacy_long_accuracy
    ec.short_lift = result.sota_short_accuracy - result.legacy_short_accuracy

    # --- Regime slices ---
    if regime_labels and len(regime_labels) == len(actuals):
        for regime in sorted(set(regime_labels)):
            mask = np.array([r == regime for r in regime_labels])
            if mask.sum() < 10:
                continue
            slice_ = RegimeSlice(regime=regime, n_bars=int(mask.sum()))
            slice_.legacy_accuracy = float(leg_correct[mask].mean())
            slice_.sota_accuracy = float(sota_correct[mask].mean())
            # PnL
            leg_pnls = []
            sota_pnls = []
            for i, m in enumerate(mask):
                if not m:
                    continue
                a = actuals[i]
                if leg_preds[i].direction == "LONG":
                    leg_pnls.append(a)
                elif leg_preds[i].direction == "SHORT":
                    leg_pnls.append(-a)
                if sota_preds[i].direction == "LONG":
                    sota_pnls.append(a)
                elif sota_preds[i].direction == "SHORT":
                    sota_pnls.append(-a)
            slice_.legacy_pnl = float(sum(leg_pnls))
            slice_.sota_pnl = float(sum(sota_pnls))
            # Conditional lift: SOTA rescue rate in this regime
            legacy_wrong_in_regime = np.logical_and(mask, ~leg_correct)
            if legacy_wrong_in_regime.sum() > 0:
                slice_.conditional_lift = float(sota_correct[legacy_wrong_in_regime].mean())
            ec.regime_slices.append(slice_)

    # --- Decision recommendation ---
    ec.recommendation, ec.confidence = _make_recommendation(ec, result)
    logger.info(
        f"{result.pair}: recommendation={ec.recommendation} (confidence={ec.confidence:.2f}) "
        f"sota_rescue_rate={ec.sota_when_legacy_wrong:.3f}"
    )
    return ec


def _make_recommendation(ec: ErrorCorrelation, result: ModelComparisonResult) -> Tuple[str, float]:
    """Automated decision gate.

    Rules (in order of priority):
    1. DISCARD: SOTA accuracy < 0.51 (doesn't beat chance) OR SOTA PnL < 0
    2. REPLACE: SOTA beats legacy by > 5% accuracy AND error correlation > 0.7
       (it's just a better version of the same thing)
    3. STACK: SOTA beats legacy by > 2% accuracy AND error correlation < 0.5
       AND sota_when_legacy_wrong > 0.55 (it rescues legacy's mistakes)
    4. UNDECIDED: everything else (needs more data)
    """
    sota_acc = result.sota_accuracy
    legacy_acc = result.legacy_accuracy
    sota_pnl = result.sota_pnl
    err_corr = ec.error_correlation
    rescue_rate = ec.sota_when_legacy_wrong

    # Rule 1: DISCARD
    if sota_acc < 0.51 or sota_pnl <= 0:
        return "DISCARD", 0.7

    # Rule 2: REPLACE
    if sota_acc > legacy_acc + 0.05 and err_corr > 0.7:
        return "REPLACE", min(0.9, (sota_acc - legacy_acc) * 5)

    # Rule 3: STACK
    if sota_acc > legacy_acc + 0.02 and err_corr < 0.5 and rescue_rate > 0.55:
        return "STACK", min(0.9, (sota_acc - legacy_acc) * 10)

    # Rule 4: UNDECIDED
    return "UNDECIDED", 0.5

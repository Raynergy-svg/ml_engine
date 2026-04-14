"""
Adaptive R:R Calculator — Phase 52 (US-325).

Adjusts risk:reward targets based on:
1. Volatility regime (LOW → conservative, EXTREME → very wide)
2. Confidence level (high → tighter, low → wider)
3. Pair recent form (winning pairs → tighter, losing → wider)

Overrides atr_tp_multiplier in execution while respecting the 1.2:1 floor.

Usage:
    rr_calc = AdaptiveRRCalculator()
    result = rr_calc.calculate(
        regime="NORMAL",
        confidence=0.62,
        atr_sl_multiplier=1.0,
        pair_recent_win_rate=0.50,
    )
    # result.tp_multiplier replaces the static atr_tp_multiplier
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


@dataclass
class AdaptiveRRConfig:
    """Configuration for adaptive R:R targeting."""
    # Regime-based base R:R targets (TP:SL ratio)
    regime_rr: Dict[str, float] = field(default_factory=lambda: {
        "LOW": 1.5,
        "NORMAL": 1.8,
        "HIGH": 2.2,
        "EXTREME": 2.5,
    })

    # Confidence modulation
    high_conf_threshold: float = 0.65   # Above → tighten R:R
    high_conf_adjustment: float = -0.2  # Tighten by this much
    low_conf_threshold: float = 0.45    # Below → widen R:R
    low_conf_adjustment: float = 0.3    # Widen by this much

    # Pair form modulation
    good_form_threshold: float = 0.55   # Win rate above → tighten
    good_form_adjustment: float = -0.15
    bad_form_threshold: float = 0.30    # Win rate below → widen
    bad_form_adjustment: float = 0.2

    # Safety floor (from trading rules)
    min_rr_ratio: float = 1.2  # Trading rule: minimum R:R floor


@dataclass
class AdaptiveRRResult:
    """Result of adaptive R:R calculation."""
    base_rr: float = 1.8           # Regime base R:R
    adjusted_rr: float = 1.8      # After confidence + form adjustments
    tp_multiplier: float = 1.5    # Final atr_tp_multiplier (adjusted_rr × atr_sl_multiplier)
    regime: str = "NORMAL"
    confidence_adj: float = 0.0
    form_adj: float = 0.0
    floor_applied: bool = False
    details: Dict[str, Any] = field(default_factory=dict)


class AdaptiveRRCalculator:
    """Calculates adaptive R:R targets based on regime, confidence, and pair form."""

    def __init__(self, config: Optional[AdaptiveRRConfig] = None):
        self.config = config or AdaptiveRRConfig()
        self._history: list = []  # Track R:R vs outcome for learning

    def calculate(
        self,
        regime: str = "NORMAL",
        confidence: float = 0.50,
        atr_sl_multiplier: float = 1.0,
        pair_recent_win_rate: float = 0.50,
    ) -> AdaptiveRRResult:
        """Calculate adaptive R:R target.

        Args:
            regime: Volatility regime (LOW, NORMAL, HIGH, EXTREME)
            confidence: Trade confidence 0.0-1.0
            atr_sl_multiplier: Current ATR SL multiplier
            pair_recent_win_rate: Recent win rate for this pair 0.0-1.0

        Returns:
            AdaptiveRRResult with adjusted tp_multiplier.
        """
        c = self.config

        # Base R:R from regime
        base_rr = c.regime_rr.get(regime.upper(), c.regime_rr.get("NORMAL", 1.8))

        # Confidence modulation
        conf_adj = 0.0
        if confidence > c.high_conf_threshold:
            conf_adj = c.high_conf_adjustment  # Tighten
        elif confidence < c.low_conf_threshold:
            conf_adj = c.low_conf_adjustment   # Widen

        # Pair form modulation
        form_adj = 0.0
        if pair_recent_win_rate > c.good_form_threshold:
            form_adj = c.good_form_adjustment  # Tighten
        elif pair_recent_win_rate < c.bad_form_threshold:
            form_adj = c.bad_form_adjustment   # Widen

        adjusted_rr = base_rr + conf_adj + form_adj

        # Apply floor
        floor_applied = False
        if adjusted_rr < c.min_rr_ratio:
            adjusted_rr = c.min_rr_ratio
            floor_applied = True

        # Convert to tp_multiplier: tp_mult = rr_ratio × sl_mult
        tp_multiplier = round(adjusted_rr * atr_sl_multiplier, 4)

        result = AdaptiveRRResult(
            base_rr=base_rr,
            adjusted_rr=round(adjusted_rr, 4),
            tp_multiplier=tp_multiplier,
            regime=regime.upper(),
            confidence_adj=conf_adj,
            form_adj=form_adj,
            floor_applied=floor_applied,
            details={
                "confidence": confidence,
                "pair_win_rate": pair_recent_win_rate,
                "atr_sl_mult": atr_sl_multiplier,
            },
        )

        logger.info(
            "Phase 52 (US-325): R:R %s base=%.2f conf_adj=%.2f form_adj=%.2f → %.2f:1 "
            "(tp_mult=%.3f%s)",
            regime, base_rr, conf_adj, form_adj, adjusted_rr, tp_multiplier,
            " FLOOR" if floor_applied else "",
        )

        return result

    def record_outcome(
        self, rr_target: float, actual_rr: float, won: bool, regime: str,
    ) -> None:
        """Record R:R outcome for learning (future use)."""
        self._history.append({
            "rr_target": rr_target,
            "actual_rr": actual_rr,
            "won": won,
            "regime": regime,
        })
        # Keep only last 200
        if len(self._history) > 200:
            self._history = self._history[-200:]

    def get_stats(self) -> Dict[str, Any]:
        """Get R:R targeting stats."""
        if not self._history:
            return {"total": 0, "avg_target_rr": 0.0, "avg_actual_rr": 0.0}
        n = len(self._history)
        return {
            "total": n,
            "avg_target_rr": round(sum(h["rr_target"] for h in self._history) / n, 3),
            "avg_actual_rr": round(sum(h["actual_rr"] for h in self._history) / n, 3),
            "win_rate": round(sum(1 for h in self._history if h["won"]) / n, 3),
        }

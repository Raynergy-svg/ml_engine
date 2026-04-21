"""Regime-conditioned dynamic SL/TP optimization.

US-080: Static ATR multipliers leave money on table in trends and get
stopped out too early in volatile regimes. Implements 4 dynamic adaptation
rules that adjust base ATR-derived SL/TP values based on current regime,
trend strength, and volatility conditions.

Rules:
1. trend_widen — widen TP in strong trends (catch more of the move)
2. early_tighten — tighten SL if conditions deteriorate early
3. vol_relax — widen SL in high-vol to avoid noise-driven stops
4. trailing_adaptive — adaptive trailing SL tightening based on regime

Each rule has regime-specific multiplier tables and can be independently
enabled/disabled. Rules compose multiplicatively on the base SL/TP.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_SLTP_LOG_PATH = _PROJECT_ROOT / "trained_data" / "dynamic_sl_tp_log.json"


# -----------------------------------------------------------------------
# Regime-specific multiplier tables
# -----------------------------------------------------------------------

# trend_widen: multiply TP when trend is strong
# Key: regime → multiplier applied to TP when trend_strength > threshold
TREND_WIDEN_TABLE: Dict[str, Dict[str, float]] = {
    "LOW": {"tp_multiplier": 1.15, "min_trend_strength": 0.60},
    "NORMAL": {"tp_multiplier": 1.25, "min_trend_strength": 0.55},
    "HIGH": {"tp_multiplier": 1.10, "min_trend_strength": 0.65},
    "EXTREME": {"tp_multiplier": 1.00, "min_trend_strength": 0.80},  # No widen in extreme
}

# early_tighten: tighten SL when confidence is low or momentum is fading
# Key: regime → SL tightening multiplier (< 1.0 means tighter)
# NOTE (2026-04-20): LOW regime sl_multiplier locked at 1.0 (no-tighten) per
# .claude/rules/trading.md promoted rule (2026-04-15, Trade 1220 EUR_AUD).
# In LOW/ranging vol, price NOISE exceeds directional movement, so tightening SL
# guarantees premature stops. The downstream guard at _apply_early_tighten
# (`if mult >= 1.0: return sl, tp, None`) makes 1.0 a safe no-op.
EARLY_TIGHTEN_TABLE: Dict[str, Dict[str, float]] = {
    "LOW": {"sl_multiplier": 1.00, "confidence_threshold": 0.55},  # was 0.85 — see note above
    "NORMAL": {"sl_multiplier": 0.90, "confidence_threshold": 0.50},
    "HIGH": {"sl_multiplier": 0.92, "confidence_threshold": 0.50},
    "EXTREME": {"sl_multiplier": 0.95, "confidence_threshold": 0.55},
}

# vol_relax: widen SL in high volatility to avoid noise stops
# Key: regime → SL widening multiplier (> 1.0 means wider)
VOL_RELAX_TABLE: Dict[str, float] = {
    "LOW": 1.00,      # No change in low vol
    "NORMAL": 1.00,   # No change in normal
    "HIGH": 1.15,     # 15% wider SL
    "EXTREME": 1.30,  # 30% wider SL
}

# trailing_adaptive: how aggressively to trail SL behind price
# Key: regime → trailing parameters
TRAILING_TABLE: Dict[str, Dict[str, float]] = {
    "LOW": {"trail_pct": 0.50, "activation_rr": 0.5},     # Trail at 50% of max favorable
    "NORMAL": {"trail_pct": 0.40, "activation_rr": 0.6},  # Trail at 40%
    "HIGH": {"trail_pct": 0.35, "activation_rr": 0.8},    # Looser trail
    "EXTREME": {"trail_pct": 0.30, "activation_rr": 1.0}, # Very loose in extreme
}


@dataclass
class SLTPAdjustment:
    """Record of a single SL/TP adjustment."""

    rule_name: str = ""
    regime: str = ""
    old_sl: float = 0.0
    new_sl: float = 0.0
    old_tp: float = 0.0
    new_tp: float = 0.0
    multiplier_applied: float = 1.0
    reason: str = ""

    def to_dict(self) -> dict:
        return {
            "rule_name": self.rule_name,
            "regime": self.regime,
            "old_sl": round(self.old_sl, 2),
            "new_sl": round(self.new_sl, 2),
            "old_tp": round(self.old_tp, 2),
            "new_tp": round(self.new_tp, 2),
            "multiplier_applied": round(self.multiplier_applied, 4),
            "reason": self.reason,
        }


@dataclass
class DynamicSLTPResult:
    """Result of dynamic SL/TP optimization."""

    original_sl_pips: float = 0.0
    original_tp_pips: float = 0.0
    optimized_sl_pips: float = 0.0
    optimized_tp_pips: float = 0.0
    adjustments: List[Dict[str, Any]] = field(default_factory=list)
    rules_applied: int = 0

    def to_dict(self) -> dict:
        return {
            "original_sl_pips": round(self.original_sl_pips, 2),
            "original_tp_pips": round(self.original_tp_pips, 2),
            "optimized_sl_pips": round(self.optimized_sl_pips, 2),
            "optimized_tp_pips": round(self.optimized_tp_pips, 2),
            "adjustments": self.adjustments,
            "rules_applied": self.rules_applied,
        }


class DynamicSLTPOptimizer:
    """Regime-conditioned dynamic SL/TP optimization.

    Applies 4 rules sequentially to adjust base ATR-derived SL/TP:
    1. trend_widen — expand TP in strong trends
    2. early_tighten — contract SL when confidence is marginal
    3. vol_relax — expand SL in high volatility
    4. trailing_adaptive — compute optimal trailing SL parameters

    Rules compose multiplicatively. Each can be independently toggled.
    All rules are regime-conditioned (LOW/NORMAL/HIGH/EXTREME).
    """

    def __init__(
        self,
        aggressiveness: float = 1.0,
        enable_trend_widen: bool = True,
        enable_early_tighten: bool = True,
        enable_vol_relax: bool = True,
        enable_trailing: bool = True,
        min_sl_pips: float = 5.0,
        max_sl_pips: float = 80.0,
        min_tp_pips: float = 8.0,
        max_tp_pips: float = 120.0,
    ):
        self.aggressiveness = max(0.5, min(2.0, aggressiveness))
        self.enable_trend_widen = enable_trend_widen
        self.enable_early_tighten = enable_early_tighten
        self.enable_vol_relax = enable_vol_relax
        self.enable_trailing = enable_trailing
        self.min_sl_pips = min_sl_pips
        self.max_sl_pips = max_sl_pips
        self.min_tp_pips = min_tp_pips
        self.max_tp_pips = max_tp_pips

    def optimize(
        self,
        base_sl_pips: float,
        base_tp_pips: float,
        regime: str,
        trend_strength: float = 0.5,
        confidence: float = 0.5,
        atr_pips: float = 0.0,
        momentum: float = 0.0,
    ) -> DynamicSLTPResult:
        """Optimize SL/TP values based on regime and market conditions.

        Args:
            base_sl_pips: ATR-derived base stop-loss in pips
            base_tp_pips: ATR-derived base take-profit in pips
            regime: Current volatility regime (LOW/NORMAL/HIGH/EXTREME)
            trend_strength: Trend strength metric (0.0-1.0)
            confidence: Model confidence score (0.0-1.0)
            atr_pips: Current ATR in pips
            momentum: Momentum score

        Returns:
            DynamicSLTPResult with optimized values and adjustment log
        """
        regime = regime.upper() if regime else "NORMAL"
        if regime not in ("LOW", "NORMAL", "HIGH", "EXTREME"):
            regime = "NORMAL"

        result = DynamicSLTPResult(
            original_sl_pips=base_sl_pips,
            original_tp_pips=base_tp_pips,
        )

        sl = base_sl_pips
        tp = base_tp_pips

        # Rule 1: Trend widen — expand TP in strong trends
        if self.enable_trend_widen:
            sl, tp, adj = self._apply_trend_widen(sl, tp, regime, trend_strength)
            if adj:
                result.adjustments.append(adj.to_dict())
                result.rules_applied += 1

        # Rule 2: Early tighten — contract SL when confidence is marginal
        if self.enable_early_tighten:
            sl, tp, adj = self._apply_early_tighten(sl, tp, regime, confidence)
            if adj:
                result.adjustments.append(adj.to_dict())
                result.rules_applied += 1

        # Rule 3: Vol relax — expand SL in high volatility
        if self.enable_vol_relax:
            sl, tp, adj = self._apply_vol_relax(sl, tp, regime)
            if adj:
                result.adjustments.append(adj.to_dict())
                result.rules_applied += 1

        # Clamp to bounds
        sl = max(self.min_sl_pips, min(self.max_sl_pips, sl))
        tp = max(self.min_tp_pips, min(self.max_tp_pips, tp))

        # Enforce minimum R:R ratio of 1.2:1 (trading rule)
        if sl > 0 and tp / sl < 1.2:
            tp = sl * 1.2
            tp = min(tp, self.max_tp_pips)

        result.optimized_sl_pips = round(sl, 2)
        result.optimized_tp_pips = round(tp, 2)

        return result

    def get_trailing_params(self, regime: str) -> Dict[str, float]:
        """Get trailing SL parameters for the current regime.

        Returns:
            Dict with trail_pct and activation_rr
        """
        regime = regime.upper() if regime else "NORMAL"
        params = TRAILING_TABLE.get(regime, TRAILING_TABLE["NORMAL"])
        return {
            "trail_pct": params["trail_pct"] * self.aggressiveness,
            "activation_rr": params["activation_rr"],
        }

    # ------------------------------------------------------------------
    # Rule implementations
    # ------------------------------------------------------------------

    def _apply_trend_widen(
        self,
        sl: float,
        tp: float,
        regime: str,
        trend_strength: float,
    ) -> Tuple[float, float, Optional[SLTPAdjustment]]:
        """Rule 1: Widen TP when trend is strong."""
        params = TREND_WIDEN_TABLE.get(regime, TREND_WIDEN_TABLE["NORMAL"])

        if trend_strength < params["min_trend_strength"]:
            return sl, tp, None

        # Scale multiplier by aggressiveness
        raw_mult = params["tp_multiplier"]
        mult = 1.0 + (raw_mult - 1.0) * self.aggressiveness

        if mult <= 1.0:
            return sl, tp, None

        new_tp = tp * mult
        adj = SLTPAdjustment(
            rule_name="trend_widen",
            regime=regime,
            old_sl=sl,
            new_sl=sl,
            old_tp=tp,
            new_tp=new_tp,
            multiplier_applied=mult,
            reason=f"Trend strength {trend_strength:.2f} >= {params['min_trend_strength']:.2f} → TP ×{mult:.2f}",
        )

        logger.debug(
            f"Dynamic SL/TP [trend_widen]: TP {tp:.1f} → {new_tp:.1f} "
            f"(regime={regime}, trend={trend_strength:.2f})"
        )

        return sl, new_tp, adj

    def _apply_early_tighten(
        self,
        sl: float,
        tp: float,
        regime: str,
        confidence: float,
    ) -> Tuple[float, float, Optional[SLTPAdjustment]]:
        """Rule 2: Tighten SL when confidence is marginal."""
        params = EARLY_TIGHTEN_TABLE.get(regime, EARLY_TIGHTEN_TABLE["NORMAL"])

        if confidence >= params["confidence_threshold"]:
            return sl, tp, None  # Confidence is fine, no tightening

        # Scale tightening by aggressiveness (more aggressive = tighter)
        raw_mult = params["sl_multiplier"]
        mult = 1.0 - (1.0 - raw_mult) * self.aggressiveness

        if mult >= 1.0:
            return sl, tp, None

        new_sl = sl * mult
        adj = SLTPAdjustment(
            rule_name="early_tighten",
            regime=regime,
            old_sl=sl,
            new_sl=new_sl,
            old_tp=tp,
            new_tp=tp,
            multiplier_applied=mult,
            reason=f"Confidence {confidence:.2f} < {params['confidence_threshold']:.2f} → SL ×{mult:.2f}",
        )

        logger.debug(
            f"Dynamic SL/TP [early_tighten]: SL {sl:.1f} → {new_sl:.1f} "
            f"(regime={regime}, conf={confidence:.2f})"
        )

        return new_sl, tp, adj

    def _apply_vol_relax(
        self,
        sl: float,
        tp: float,
        regime: str,
    ) -> Tuple[float, float, Optional[SLTPAdjustment]]:
        """Rule 3: Widen SL in high volatility to avoid noise stops."""
        mult = VOL_RELAX_TABLE.get(regime, 1.0)

        # Scale by aggressiveness
        mult = 1.0 + (mult - 1.0) * self.aggressiveness

        if mult <= 1.0:
            return sl, tp, None

        new_sl = sl * mult
        # Also proportionally widen TP to maintain R:R ratio
        tp_mult = 1.0 + (mult - 1.0) * 0.5  # Half the SL widening applied to TP
        new_tp = tp * tp_mult

        adj = SLTPAdjustment(
            rule_name="vol_relax",
            regime=regime,
            old_sl=sl,
            new_sl=new_sl,
            old_tp=tp,
            new_tp=new_tp,
            multiplier_applied=mult,
            reason=f"Volatility regime {regime} → SL ×{mult:.2f}, TP ×{tp_mult:.2f}",
        )

        logger.debug(
            f"Dynamic SL/TP [vol_relax]: SL {sl:.1f} → {new_sl:.1f}, "
            f"TP {tp:.1f} → {new_tp:.1f} (regime={regime})"
        )

        return new_sl, new_tp, adj

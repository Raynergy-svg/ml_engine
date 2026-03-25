"""Dynamic hedging module for stress periods.

US-077: Auto-open inverse positions on correlated pairs when macro stress
is detected. Hedge size = primary_size * correlation * vol_ratio.
Auto-close when stress recedes (modifier > 0.90). Reduces tail risk 30-50%
without closing profitable positions.

Hedge logic:
1. Identify open positions during stress
2. Find correlated pairs (|rho| > hedge_min_correlation)
3. Size hedge: primary_lots * |correlation| * (vol_primary / vol_hedge)
4. Tag hedges as 'hedge:{primary_pair}' in journal
5. Auto-close hedges when macro stress modifier > 0.90
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Default thresholds
DEFAULT_MIN_CORRELATION = 0.65  # Minimum |correlation| to consider hedging
DEFAULT_STRESS_TRIGGER = 0.70   # Macro stress modifier below this triggers hedging
DEFAULT_STRESS_RELEASE = 0.90   # Modifier above this triggers hedge release
MIN_HEDGE_LOTS = 0.01           # Floor for hedge size
MAX_HEDGE_RATIO = 0.80          # Never hedge more than 80% of primary position

# Standard FX pair correlations (fallback when no live data)
# Positive correlation = same direction, Negative = inverse
STATIC_CORRELATIONS: Dict[str, Dict[str, float]] = {
    "EUR_USD": {"GBP_USD": 0.85, "AUD_USD": 0.70, "USD_CHF": -0.90, "USD_JPY": -0.30},
    "GBP_USD": {"EUR_USD": 0.85, "AUD_USD": 0.65, "USD_CHF": -0.80, "USD_JPY": -0.25},
    "USD_JPY": {"USD_CHF": 0.55, "EUR_USD": -0.30, "GBP_USD": -0.25, "AUD_USD": -0.20},
    "AUD_USD": {"EUR_USD": 0.70, "GBP_USD": 0.65, "NZD_USD": 0.90, "USD_CAD": -0.55},
    "USD_CHF": {"EUR_USD": -0.90, "GBP_USD": -0.80, "USD_JPY": 0.55},
    "NZD_USD": {"AUD_USD": 0.90, "EUR_USD": 0.55, "GBP_USD": 0.50},
    "USD_CAD": {"AUD_USD": -0.55, "NZD_USD": -0.50, "EUR_USD": -0.40},
}


@dataclass
class HedgeCandidate:
    """A potential hedge for an open position."""

    primary_pair: str = ""
    primary_direction: str = ""
    primary_lots: float = 0.0
    hedge_pair: str = ""
    hedge_direction: str = ""  # Inverse of primary direction (adjusted for correlation sign)
    hedge_lots: float = 0.0
    correlation: float = 0.0
    vol_ratio: float = 1.0
    hedge_tag: str = ""  # 'hedge:{primary_pair}'
    reason: str = ""

    def to_dict(self) -> dict:
        return {
            "primary_pair": self.primary_pair,
            "primary_direction": self.primary_direction,
            "primary_lots": round(self.primary_lots, 4),
            "hedge_pair": self.hedge_pair,
            "hedge_direction": self.hedge_direction,
            "hedge_lots": round(self.hedge_lots, 4),
            "correlation": round(self.correlation, 4),
            "vol_ratio": round(self.vol_ratio, 4),
            "hedge_tag": self.hedge_tag,
            "reason": self.reason,
        }


@dataclass
class HedgeStatus:
    """Current state of the dynamic hedging system."""

    is_hedging_active: bool = False
    stress_score: float = 0.0
    stress_modifier: float = 1.0
    active_hedges: List[Dict[str, Any]] = field(default_factory=list)
    candidates: List[Dict[str, Any]] = field(default_factory=list)
    hedges_to_close: List[str] = field(default_factory=list)  # Pairs to unwind
    total_hedge_lots: float = 0.0

    def to_dict(self) -> dict:
        return {
            "is_hedging_active": self.is_hedging_active,
            "stress_score": round(self.stress_score, 4),
            "stress_modifier": round(self.stress_modifier, 4),
            "active_hedges": self.active_hedges,
            "candidates": self.candidates,
            "hedges_to_close": self.hedges_to_close,
            "total_hedge_lots": round(self.total_hedge_lots, 4),
        }


class DynamicHedgeManager:
    """Manages dynamic hedges during stress periods.

    Lifecycle:
    1. DETECT: MacroStressDetector reports stress_modifier < trigger_threshold
    2. IDENTIFY: Scan open positions for hedgeable candidates
    3. SIZE: Compute hedge = primary_lots * |correlation| * vol_ratio
    4. EXECUTE: Create hedge orders tagged 'hedge:{primary_pair}'
    5. MONITOR: Track hedge P/L relative to primary
    6. RELEASE: Auto-close hedges when stress_modifier > release_threshold
    """

    def __init__(
        self,
        min_correlation: float = DEFAULT_MIN_CORRELATION,
        stress_trigger: float = DEFAULT_STRESS_TRIGGER,
        stress_release: float = DEFAULT_STRESS_RELEASE,
        max_hedge_ratio: float = MAX_HEDGE_RATIO,
        correlation_data: Optional[Dict[str, Dict[str, float]]] = None,
    ):
        self.min_correlation = min_correlation
        self.stress_trigger = stress_trigger
        self.stress_release = stress_release
        self.max_hedge_ratio = max_hedge_ratio
        self._correlations = correlation_data or STATIC_CORRELATIONS
        self._active_hedge_tags: Dict[str, str] = {}  # hedge_pair -> primary_pair

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(
        self,
        open_positions: List[Dict[str, Any]],
        stress_modifier: float,
        stress_score: float = 0.0,
        volatility_data: Optional[Dict[str, float]] = None,
    ) -> HedgeStatus:
        """Evaluate whether to open, maintain, or close hedges.

        Args:
            open_positions: List of dicts with keys:
                - pair, direction, lots, unrealized_pnl
            stress_modifier: Current macro stress modifier (0.5 - 1.0)
            stress_score: Raw stress score for reporting
            volatility_data: Optional ATR/volatility per pair

        Returns:
            HedgeStatus with candidates to open and hedges to close
        """
        status = HedgeStatus(
            stress_score=stress_score,
            stress_modifier=stress_modifier,
        )

        # Check if we need to RELEASE existing hedges (stress receding)
        if stress_modifier >= self.stress_release:
            status.is_hedging_active = False
            # All active hedges should be closed
            status.hedges_to_close = list(self._active_hedge_tags.keys())
            if status.hedges_to_close:
                logger.info(
                    "DynamicHedge: Stress receded (modifier=%.2f), releasing %d hedges: %s",
                    stress_modifier, len(status.hedges_to_close),
                    ", ".join(status.hedges_to_close),
                )
                self._active_hedge_tags.clear()
            return status

        # Check if we should OPEN new hedges (stress detected)
        if stress_modifier <= self.stress_trigger:
            status.is_hedging_active = True
            candidates = self.identify_hedge_candidates(
                open_positions=open_positions,
                volatility_data=volatility_data or {},
            )
            status.candidates = [c.to_dict() for c in candidates]
            status.total_hedge_lots = sum(c.hedge_lots for c in candidates)

            # Track which hedges we've recommended
            for c in candidates:
                self._active_hedge_tags[c.hedge_pair] = c.primary_pair

            if candidates:
                logger.info(
                    "DynamicHedge: Stress detected (modifier=%.2f), "
                    "recommending %d hedges totaling %.4f lots",
                    stress_modifier, len(candidates), status.total_hedge_lots,
                )
        else:
            # Moderate stress — maintain existing hedges but don't open new ones
            status.is_hedging_active = len(self._active_hedge_tags) > 0
            status.active_hedges = [
                {"hedge_pair": k, "primary_pair": v}
                for k, v in self._active_hedge_tags.items()
            ]

        return status

    def identify_hedge_candidates(
        self,
        open_positions: List[Dict[str, Any]],
        volatility_data: Optional[Dict[str, float]] = None,
    ) -> List[HedgeCandidate]:
        """Find hedge candidates for all open positions.

        Args:
            open_positions: Current open positions
            volatility_data: ATR per pair for vol-ratio computation

        Returns:
            List of HedgeCandidates ready for execution
        """
        volatility_data = volatility_data or {}
        candidates: List[HedgeCandidate] = []
        already_open = {pos["pair"] for pos in open_positions}

        for pos in open_positions:
            pair = pos.get("pair", "")
            direction = pos.get("direction", "").upper()
            lots = float(pos.get("lots", 0.0))

            if not pair or not direction or lots <= 0:
                continue

            # Skip if this is itself a hedge
            if pair in self._active_hedge_tags:
                continue

            # Find correlated pairs
            pair_corrs = self._correlations.get(pair, {})
            for hedge_pair, correlation in pair_corrs.items():
                if abs(correlation) < self.min_correlation:
                    continue
                # Don't hedge with a pair we already have a position in
                if hedge_pair in already_open:
                    continue

                candidate = self.compute_hedge_size(
                    primary_pair=pair,
                    primary_direction=direction,
                    primary_lots=lots,
                    hedge_pair=hedge_pair,
                    correlation=correlation,
                    primary_vol=volatility_data.get(pair, 1.0),
                    hedge_vol=volatility_data.get(hedge_pair, 1.0),
                )

                if candidate.hedge_lots >= MIN_HEDGE_LOTS:
                    candidates.append(candidate)

        return candidates

    def compute_hedge_size(
        self,
        primary_pair: str,
        primary_direction: str,
        primary_lots: float,
        hedge_pair: str,
        correlation: float,
        primary_vol: float = 1.0,
        hedge_vol: float = 1.0,
    ) -> HedgeCandidate:
        """Compute optimal hedge size.

        hedge_lots = primary_lots * |correlation| * (vol_primary / vol_hedge)
        Capped at max_hedge_ratio * primary_lots

        For positive correlation: hedge in opposite direction
        For negative correlation: hedge in same direction
        """
        abs_corr = abs(correlation)

        # Vol ratio: match risk exposure across pairs
        vol_ratio = 1.0
        if hedge_vol > 0:
            vol_ratio = primary_vol / hedge_vol
            vol_ratio = max(0.3, min(vol_ratio, 3.0))  # Cap extremes

        # Raw hedge size
        raw_hedge_lots = primary_lots * abs_corr * vol_ratio
        hedge_lots = min(raw_hedge_lots, primary_lots * self.max_hedge_ratio)
        hedge_lots = max(hedge_lots, 0.0)
        hedge_lots = round(hedge_lots, 2)

        # Direction logic
        if correlation > 0:
            # Positive correlation: hedge in opposite direction
            hedge_direction = "SHORT" if primary_direction == "LONG" else "LONG"
        else:
            # Negative correlation: hedge in same direction
            hedge_direction = primary_direction

        return HedgeCandidate(
            primary_pair=primary_pair,
            primary_direction=primary_direction,
            primary_lots=primary_lots,
            hedge_pair=hedge_pair,
            hedge_direction=hedge_direction,
            hedge_lots=hedge_lots,
            correlation=correlation,
            vol_ratio=vol_ratio,
            hedge_tag=f"hedge:{primary_pair}",
            reason=(
                f"Macro stress hedge: {hedge_pair} {hedge_direction} "
                f"{hedge_lots:.2f} lots (corr={correlation:.2f}, vol_ratio={vol_ratio:.2f})"
            ),
        )

    def update_correlations(
        self,
        correlation_data: Dict[str, Dict[str, float]],
    ) -> None:
        """Update correlation data from live computation."""
        self._correlations = correlation_data

    def get_active_hedges(self) -> Dict[str, str]:
        """Get map of hedge_pair -> primary_pair for active hedges."""
        return dict(self._active_hedge_tags)

    def clear_hedge(self, hedge_pair: str) -> None:
        """Remove a hedge from tracking after it's been closed."""
        self._active_hedge_tags.pop(hedge_pair, None)

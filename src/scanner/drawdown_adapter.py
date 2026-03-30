"""
Drawdown Adapter — Phase 52 (US-326).

Monitors rolling drawdown and progressively restricts trading behavior:

Tier 1 (dd > 3%):  Tighten confidence threshold by +0.05
Tier 2 (dd > 5%):  Restrict to top-3 performing pairs only + Tier 1
Tier 3 (dd > 8%):  Restrict to London session only + Tier 2
Tier 4 (dd > 12%): Halt all new trades until drawdown recovers below 8%

Recovery: each tier releases when drawdown drops 2% below tier entry.

Usage:
    adapter = DrawdownAdapter()
    result = adapter.check(
        current_nav=98000,
        peak_nav=100000,
        pair="EUR_USD",
        confidence=0.62,
        session_name="LONDON",
        pair_win_rates={"EUR_USD": 0.55, "GBP_USD": 0.52, ...},
    )
    if not result.allowed:
        # Block trade — result.reason explains why
    else:
        # Use result.adjusted_confidence for tightened threshold
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class DrawdownAdapterConfig:
    """Configuration for drawdown-adaptive behavior."""
    # Tier thresholds (drawdown percentage)
    tier1_threshold: float = 3.0   # Tighten confidence
    tier2_threshold: float = 5.0   # Restrict pairs
    tier3_threshold: float = 8.0   # Restrict sessions
    tier4_threshold: float = 12.0  # Halt all trades

    # Recovery hysteresis: tier releases when dd drops this much below entry
    recovery_margin: float = 2.0

    # Tier 1: confidence tightening amount
    confidence_tighten: float = 0.05

    # Tier 2: max pairs allowed (by win rate)
    max_pairs_in_tier2: int = 3

    # Tier 3: allowed sessions
    tier3_allowed_sessions: List[str] = field(default_factory=lambda: [
        "LONDON", "OVERLAP",
    ])

    # Tier 4: recovery target (halt until dd drops below this)
    tier4_recovery_target: float = 8.0


@dataclass
class DrawdownCheckResult:
    """Result of drawdown adapter check."""
    allowed: bool = True
    drawdown_pct: float = 0.0
    active_tier: int = 0           # 0 = no restriction, 1-4
    adjusted_confidence: float = 0.0
    confidence_penalty: float = 0.0
    pair_restricted: bool = False
    session_restricted: bool = False
    reason: str = ""
    details: Dict[str, Any] = field(default_factory=dict)


class DrawdownAdapter:
    """Monitors drawdown and progressively restricts trading behavior.

    Implements 4-tier escalation with hysteresis-based recovery to prevent
    trading during extended drawdowns while allowing gradual re-entry.
    """

    def __init__(self, config: Optional[DrawdownAdapterConfig] = None):
        self.config = config or DrawdownAdapterConfig()
        self._peak_nav: float = 0.0
        self._active_tier: int = 0
        self._tier_entry_dd: Dict[int, float] = {}  # tier → drawdown % when entered

    def update_peak(self, nav: float) -> None:
        """Update peak NAV if current is higher."""
        if nav > self._peak_nav:
            self._peak_nav = nav

    def get_drawdown_pct(self, current_nav: float, peak_nav: Optional[float] = None) -> float:
        """Calculate drawdown percentage from peak.

        Args:
            current_nav: Current NAV
            peak_nav: Optional peak override (uses tracked peak if None)

        Returns:
            Drawdown as positive percentage (e.g., 5.0 for 5% drawdown)
        """
        peak = peak_nav or self._peak_nav
        if peak <= 0:
            return 0.0
        dd = ((peak - current_nav) / peak) * 100.0
        return max(0.0, dd)

    def _determine_tier(self, dd_pct: float) -> int:
        """Determine active tier based on drawdown with hysteresis.

        Tier escalation: immediate when dd crosses threshold.
        Tier de-escalation: only when dd drops recovery_margin below tier entry.
        """
        c = self.config

        # Check escalation (higher tiers override lower)
        new_tier = 0
        if dd_pct >= c.tier4_threshold:
            new_tier = 4
        elif dd_pct >= c.tier3_threshold:
            new_tier = 3
        elif dd_pct >= c.tier2_threshold:
            new_tier = 2
        elif dd_pct >= c.tier1_threshold:
            new_tier = 1

        # If escalating, record entry drawdown
        if new_tier > self._active_tier:
            for t in range(self._active_tier + 1, new_tier + 1):
                if t not in self._tier_entry_dd:
                    self._tier_entry_dd[t] = dd_pct
            self._active_tier = new_tier
            return new_tier

        # If potentially de-escalating, check hysteresis
        if new_tier < self._active_tier:
            # Check if current tier should be released
            current = self._active_tier
            while current > 0:
                # Tier threshold is the minimum — use whichever is higher
                tier_threshold = {
                    1: c.tier1_threshold,
                    2: c.tier2_threshold,
                    3: c.tier3_threshold,
                    4: c.tier4_recovery_target,  # Tier 4 uses special recovery target
                }.get(current, 0.0)
                recovery_point = tier_threshold - c.recovery_margin
                if dd_pct <= recovery_point:
                    # Release this tier
                    self._tier_entry_dd.pop(current, None)
                    current -= 1
                else:
                    break
            self._active_tier = max(0, current)

        return self._active_tier

    def check(
        self,
        current_nav: float,
        peak_nav: Optional[float] = None,
        pair: str = "",
        confidence: float = 0.50,
        session_name: str = "LONDON",
        pair_win_rates: Optional[Dict[str, float]] = None,
    ) -> DrawdownCheckResult:
        """Check if trade is allowed under current drawdown conditions.

        Args:
            current_nav: Current account NAV
            peak_nav: Peak NAV (uses tracked peak if None)
            pair: Instrument name
            confidence: Raw confidence score
            session_name: Current trading session
            pair_win_rates: Dict of {pair: win_rate} for pair restriction

        Returns:
            DrawdownCheckResult with allowed/blocked status and adjustments.
        """
        c = self.config

        # Update peak tracking
        if peak_nav and peak_nav > self._peak_nav:
            self._peak_nav = peak_nav
        self.update_peak(current_nav)

        dd_pct = self.get_drawdown_pct(current_nav, peak_nav)
        tier = self._determine_tier(dd_pct)

        result = DrawdownCheckResult(
            allowed=True,
            drawdown_pct=round(dd_pct, 2),
            active_tier=tier,
            adjusted_confidence=confidence,
            details={
                "peak_nav": self._peak_nav,
                "current_nav": current_nav,
                "tier_entries": dict(self._tier_entry_dd),
            },
        )

        if tier == 0:
            return result

        # ── Tier 1+: Tighten confidence (with floor to prevent sub-gate values) ──
        if tier >= 1:
            result.confidence_penalty = c.confidence_tighten
            result.adjusted_confidence = max(0.40, confidence - c.confidence_tighten)
            result.details["confidence_tightened_by"] = c.confidence_tighten

        # ── Tier 2+: Restrict to top-N pairs ──
        if tier >= 2 and pair_win_rates:
            sorted_pairs = sorted(
                pair_win_rates.items(), key=lambda x: x[1], reverse=True,
            )
            top_pairs = [p for p, _ in sorted_pairs[:c.max_pairs_in_tier2]]
            if pair not in top_pairs:
                result.allowed = False
                result.pair_restricted = True
                result.reason = (
                    f"BLOCKED: Tier {tier} drawdown ({dd_pct:.1f}%) — "
                    f"{pair} not in top-{c.max_pairs_in_tier2} pairs {top_pairs}"
                )
                logger.info(
                    "Phase 52 (US-326): %s BLOCKED by drawdown adapter — %s",
                    pair, result.reason,
                )
                return result
            result.details["allowed_pairs"] = top_pairs

        # ── Tier 3+: Restrict to allowed sessions ──
        if tier >= 3:
            if session_name.upper() not in [s.upper() for s in c.tier3_allowed_sessions]:
                result.allowed = False
                result.session_restricted = True
                result.reason = (
                    f"BLOCKED: Tier {tier} drawdown ({dd_pct:.1f}%) — "
                    f"session {session_name} not allowed (only {c.tier3_allowed_sessions})"
                )
                logger.info(
                    "Phase 52 (US-326): %s BLOCKED by drawdown adapter — %s",
                    pair, result.reason,
                )
                return result

        # ── Tier 4: Halt all trades ──
        if tier >= 4:
            result.allowed = False
            result.reason = (
                f"BLOCKED: Tier 4 drawdown ({dd_pct:.1f}%) — "
                f"all trades halted until recovery below {c.tier4_recovery_target}%"
            )
            logger.info(
                "Phase 52 (US-326): %s BLOCKED by drawdown adapter — %s",
                pair, result.reason,
            )
            return result

        # Allowed with restrictions
        if tier > 0:
            logger.info(
                "Phase 52 (US-326): %s drawdown tier %d (dd=%.1f%%) — "
                "confidence %.3f → %.3f%s",
                pair, tier, dd_pct, confidence, result.adjusted_confidence,
                f", pairs restricted to top-{c.max_pairs_in_tier2}" if tier >= 2 else "",
            )

        return result

    def get_state(self) -> Dict[str, Any]:
        """Get current adapter state for persistence."""
        return {
            "peak_nav": self._peak_nav,
            "active_tier": self._active_tier,
            "tier_entry_dd": dict(self._tier_entry_dd),
        }

    def load_state(self, state: Dict[str, Any]) -> None:
        """Restore adapter state from persistence."""
        self._peak_nav = float(state.get("peak_nav", 0.0))
        self._active_tier = int(state.get("active_tier", 0))
        self._tier_entry_dd = {
            int(k): float(v) for k, v in state.get("tier_entry_dd", {}).items()
        }

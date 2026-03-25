"""Smart execution order slicing router.

US-105: Wire smart_execution (US-075) into the trade execution pipeline.
SmartExecutionEngine exists but execute_trade() always sends full position
immediately. This router decides single-shot vs sliced execution and
feeds per-slice slippage back to execution_quality_tracker.

Decision logic:
- Small orders (<= lot_threshold): single-shot execution
- Large orders (> lot_threshold): split into TWAP slices
- HIGH/EXTREME regime: lower threshold (more aggressive slicing)
- Feed slice-level slippage to execution_quality_tracker
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_LOG_PATH = _PROJECT_ROOT / "trained_data" / "execution_routing_log.json"

# Slicing thresholds per regime
REGIME_LOT_THRESHOLDS: Dict[str, float] = {
    "LOW": 0.10,  # Only slice above 0.10 lots in calm markets
    "NORMAL": 0.07,  # More cautious in normal conditions
    "HIGH": 0.05,  # Slice smaller in volatile markets
    "EXTREME": 0.03,  # Always slice in extreme conditions
}
DEFAULT_LOT_THRESHOLD = 0.07

# TWAP parameters
DEFAULT_N_SLICES = 3  # Split into 3 equal slices
SLICE_DELAY_MS = 500  # 500ms between slices
ADAPTIVE_REDUCTION = 0.80  # Reduce next slice by 20% if slippage exceeds threshold
SLIPPAGE_THRESHOLD_PIPS = 1.5  # Trigger adaptive reduction above this


@dataclass
class SlicePlan:
    """Plan for a single execution slice."""

    slice_num: int = 0
    lot_size: float = 0.0
    delay_ms: int = 0
    expected_slippage_pips: float = 0.0

    def to_dict(self) -> dict:
        return {
            "slice_num": self.slice_num,
            "lot_size": round(self.lot_size, 4),
            "delay_ms": self.delay_ms,
            "expected_slippage_pips": round(self.expected_slippage_pips, 2),
        }


@dataclass
class ExecutionPlan:
    """Full execution plan for an order."""

    strategy: str = "SINGLE"  # SINGLE or TWAP
    total_lots: float = 0.0
    n_slices: int = 1
    slices: List[SlicePlan] = field(default_factory=list)
    reason: str = ""
    regime: str = "NORMAL"

    def to_dict(self) -> dict:
        return {
            "strategy": self.strategy,
            "total_lots": round(self.total_lots, 4),
            "n_slices": self.n_slices,
            "slices": [s.to_dict() for s in self.slices],
            "reason": self.reason,
            "regime": self.regime,
        }


class ExecutionRouter:
    """Routes order execution between single-shot and sliced strategies.

    Bridges smart_execution (which plans slices) and execution.py
    (which sends orders). Decides when to slice and feeds slippage
    data back to execution_quality_tracker.

    Usage:
        router = ExecutionRouter()
        plan = router.route_execution("EUR_USD", 0.12, "HIGH")
        # plan.strategy = "TWAP", plan.slices = [SlicePlan(0.04), ...]
    """

    def __init__(
        self,
        persistence_path: Optional[Path] = None,
    ):
        self.persistence_path = persistence_path or _LOG_PATH

        # History of routing decisions
        self._decisions: List[Dict[str, Any]] = []
        self._total_routed: int = 0
        self._total_sliced: int = 0

        # Per-pair historical slippage (for adaptive planning)
        self._pair_avg_slippage: Dict[str, float] = {}

        self._load_state()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def route_execution(
        self,
        pair: str,
        lot_size: float,
        regime: str = "NORMAL",
    ) -> ExecutionPlan:
        """Decide execution strategy for an order.

        Args:
            pair: Currency pair.
            lot_size: Total position size in lots.
            regime: Current volatility regime.

        Returns:
            ExecutionPlan with strategy and slice details.
        """
        threshold = REGIME_LOT_THRESHOLDS.get(regime, DEFAULT_LOT_THRESHOLD)

        self._total_routed += 1

        if lot_size <= threshold:
            # Single-shot execution
            plan = ExecutionPlan(
                strategy="SINGLE",
                total_lots=lot_size,
                n_slices=1,
                slices=[
                    SlicePlan(
                        slice_num=0,
                        lot_size=lot_size,
                        delay_ms=0,
                        expected_slippage_pips=self._estimate_slippage(pair, lot_size, regime),
                    )
                ],
                reason=f"Below threshold ({lot_size:.4f} <= {threshold:.4f})",
                regime=regime,
            )
        else:
            # TWAP sliced execution
            plan = self.plan_slices(lot_size, pair, regime)
            self._total_sliced += 1

        # Log decision
        self._decisions.append({
            "pair": pair,
            "lot_size": round(lot_size, 4),
            "regime": regime,
            "threshold": threshold,
            "strategy": plan.strategy,
            "n_slices": plan.n_slices,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        if len(self._decisions) > 300:
            self._decisions = self._decisions[-150:]

        return plan

    def plan_slices(
        self,
        lot_size: float,
        pair: str,
        regime: str = "NORMAL",
        n_slices: Optional[int] = None,
    ) -> ExecutionPlan:
        """Plan TWAP sliced execution.

        Args:
            lot_size: Total position size.
            pair: Currency pair.
            regime: Volatility regime.
            n_slices: Override number of slices.

        Returns:
            ExecutionPlan with TWAP slices.
        """
        n = n_slices or self._compute_optimal_slices(lot_size, regime)
        base_slice = lot_size / n

        slices = []
        for i in range(n):
            expected_slip = self._estimate_slippage(pair, base_slice, regime)
            slices.append(
                SlicePlan(
                    slice_num=i,
                    lot_size=round(base_slice, 4),
                    delay_ms=SLICE_DELAY_MS if i > 0 else 0,
                    expected_slippage_pips=expected_slip,
                )
            )

        return ExecutionPlan(
            strategy="TWAP",
            total_lots=lot_size,
            n_slices=n,
            slices=slices,
            reason=f"Above threshold, TWAP with {n} slices",
            regime=regime,
        )

    def record_slice_result(
        self,
        pair: str,
        slice_num: int,
        intended_size: float,
        actual_fill: float,
        slippage_pips: float,
    ) -> Optional[float]:
        """Record execution result for a slice and return adjustment.

        Args:
            pair: Currency pair.
            slice_num: Which slice was executed.
            intended_size: Planned lot size.
            actual_fill: Actual filled size.
            slippage_pips: Realized slippage in pips.

        Returns:
            Adjusted lot size for next slice (or None if no adjustment needed).
        """
        # Update pair average slippage
        prev_avg = self._pair_avg_slippage.get(pair, 0.0)
        self._pair_avg_slippage[pair] = 0.8 * prev_avg + 0.2 * abs(slippage_pips)

        # If slippage exceeds threshold, reduce next slice
        if abs(slippage_pips) > SLIPPAGE_THRESHOLD_PIPS:
            adjusted = intended_size * ADAPTIVE_REDUCTION
            logger.debug(
                f"ExecutionRouter: Slice {slice_num} slippage={slippage_pips:.1f} > "
                f"{SLIPPAGE_THRESHOLD_PIPS}, reducing next slice: "
                f"{intended_size:.4f} → {adjusted:.4f}"
            )
            return round(adjusted, 4)

        return None

    def get_status(self) -> Dict[str, Any]:
        """Get router status for observability."""
        return {
            "total_routed": self._total_routed,
            "total_sliced": self._total_sliced,
            "slice_rate": round(
                self._total_sliced / max(self._total_routed, 1), 2
            ),
            "pair_avg_slippage": {
                k: round(v, 2) for k, v in self._pair_avg_slippage.items()
            },
            "recent_decisions": self._decisions[-5:],
        }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _compute_optimal_slices(self, lot_size: float, regime: str) -> int:
        """Compute optimal number of slices based on order size and regime."""
        # More slices for larger orders and volatile regimes
        regime_mult = {"LOW": 1.0, "NORMAL": 1.0, "HIGH": 1.5, "EXTREME": 2.0}
        mult = regime_mult.get(regime, 1.0)
        n = max(2, min(8, int(lot_size * 20 * mult)))
        return n

    def _estimate_slippage(
        self,
        pair: str,
        slice_size: float,
        regime: str,
    ) -> float:
        """Estimate expected slippage for a slice."""
        # Base slippage from historical data
        base = self._pair_avg_slippage.get(pair, 0.5)

        # Size impact: larger slices = more slippage
        size_factor = 1.0 + slice_size * 5.0

        # Regime impact
        regime_mult = {"LOW": 0.7, "NORMAL": 1.0, "HIGH": 1.5, "EXTREME": 2.5}
        r_mult = regime_mult.get(regime, 1.0)

        return round(base * size_factor * r_mult, 2)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_state(self) -> None:
        """Persist routing state."""
        try:
            data = {
                "version": 1,
                "total_routed": self._total_routed,
                "total_sliced": self._total_sliced,
                "pair_avg_slippage": {
                    k: round(v, 4) for k, v in self._pair_avg_slippage.items()
                },
                "decisions": self._decisions[-150:],
                "last_updated": datetime.now(timezone.utc).isoformat(),
            }
            self.persistence_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                from src.scanner.automation.safe_json import safe_json_write
                safe_json_write(self.persistence_path, data)
            except ImportError:
                self.persistence_path.write_text(json.dumps(data, indent=2))
        except Exception as e:
            logger.debug(f"ExecutionRouter: Failed to save: {e}")

    def _load_state(self) -> None:
        """Load persisted state."""
        if not self.persistence_path.exists():
            return
        try:
            try:
                from src.scanner.automation.safe_json import safe_json_read
                data = safe_json_read(self.persistence_path, default={})
            except ImportError:
                data = json.loads(self.persistence_path.read_text(encoding="utf-8"))

            if isinstance(data, dict):
                self._total_routed = data.get("total_routed", 0)
                self._total_sliced = data.get("total_sliced", 0)
                self._pair_avg_slippage = data.get("pair_avg_slippage", {})
                self._decisions = data.get("decisions", [])
        except Exception as e:
            logger.debug(f"ExecutionRouter: Failed to load: {e}")

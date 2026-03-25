"""Adaptive VWAP/TWAP smart execution engine.

US-075: Slice larger orders into micro-orders over VWAP/TWAP windows.
VWAP sizes slices proportional to 1-min volume profile. TWAP uses equal
slices evenly spaced. Adaptive retry reduces slice size if fill is worse
than expected.

Usage:
    engine = SmartExecutionEngine(strategy="VWAP")
    plan = engine.plan_execution(
        position_size_lots=0.10,
        pair="EUR_USD",
        market_data={"avg_volume": 1200, "atr_pips": 8.5},
    )
    # plan.slices = [ExecutionSlice(...), ...]
"""

from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Minimum position to bother slicing (below this, just execute atomically)
MIN_SLICE_LOTS = 0.01
# Default execution window in seconds
DEFAULT_WINDOW_SECONDS = 300  # 5 minutes
# Maximum slices to prevent micro-dust orders
MAX_SLICES = 20
MIN_SLICES = 2
# Slippage threshold to trigger adaptive retry
SLIPPAGE_THRESHOLD_PIPS = 0.5


@dataclass
class ExecutionSlice:
    """A single slice within an execution plan."""

    slice_index: int = 0
    size_lots: float = 0.0
    delay_seconds: float = 0.0  # Delay before executing this slice
    weight: float = 0.0  # Proportion of total (for VWAP)
    status: str = "pending"  # pending, filled, failed, skipped
    fill_price: Optional[float] = None
    benchmark_price: Optional[float] = None
    slippage_pips: float = 0.0
    executed_at: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "slice_index": self.slice_index,
            "size_lots": round(self.size_lots, 4),
            "delay_seconds": round(self.delay_seconds, 1),
            "weight": round(self.weight, 4),
            "status": self.status,
            "fill_price": self.fill_price,
            "benchmark_price": self.benchmark_price,
            "slippage_pips": round(self.slippage_pips, 4),
            "executed_at": self.executed_at,
        }


@dataclass
class ExecutionPlan:
    """Complete execution plan with all slices and strategy metadata."""

    pair: str = ""
    direction: str = ""
    strategy: str = "TWAP"  # VWAP or TWAP
    total_lots: float = 0.0
    slice_count: int = 0
    window_seconds: float = DEFAULT_WINDOW_SECONDS
    slices: List[ExecutionSlice] = field(default_factory=list)
    created_at: str = ""
    # Tracking
    total_filled_lots: float = 0.0
    avg_fill_price: float = 0.0
    total_slippage_pips: float = 0.0
    status: str = "planned"  # planned, executing, completed, aborted
    adaptive_reductions: int = 0  # Times slice was reduced due to slippage

    def to_dict(self) -> dict:
        return {
            "pair": self.pair,
            "direction": self.direction,
            "strategy": self.strategy,
            "total_lots": round(self.total_lots, 4),
            "slice_count": self.slice_count,
            "window_seconds": self.window_seconds,
            "slices": [s.to_dict() for s in self.slices],
            "total_filled_lots": round(self.total_filled_lots, 4),
            "avg_fill_price": round(self.avg_fill_price, 6),
            "total_slippage_pips": round(self.total_slippage_pips, 4),
            "status": self.status,
            "adaptive_reductions": self.adaptive_reductions,
            "created_at": self.created_at,
        }


class SmartExecutionEngine:
    """Slices large orders into micro-orders using VWAP or TWAP strategies.

    VWAP (Volume Weighted Average Price):
        Sizes slices proportional to historical intraday volume profile.
        Front-loads execution in high-volume periods for better fills.

    TWAP (Time Weighted Average Price):
        Equal-sized slices at regular intervals.
        Minimizes market impact by spreading evenly.

    Adaptive retry:
        If any slice fills worse than SLIPPAGE_THRESHOLD_PIPS from benchmark,
        subsequent slices are reduced by 20% to lower market impact.
    """

    def __init__(
        self,
        strategy: str = "TWAP",
        window_seconds: float = DEFAULT_WINDOW_SECONDS,
        max_slices: int = MAX_SLICES,
        min_slice_lots: float = MIN_SLICE_LOTS,
        slippage_threshold_pips: float = SLIPPAGE_THRESHOLD_PIPS,
    ):
        self.strategy = strategy.upper()
        self.window_seconds = window_seconds
        self.max_slices = max_slices
        self.min_slice_lots = min_slice_lots
        self.slippage_threshold_pips = slippage_threshold_pips

        if self.strategy not in ("VWAP", "TWAP"):
            logger.warning(f"Unknown strategy '{strategy}', falling back to TWAP")
            self.strategy = "TWAP"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def plan_execution(
        self,
        position_size_lots: float,
        pair: str = "",
        direction: str = "",
        market_data: Optional[Dict[str, Any]] = None,
    ) -> ExecutionPlan:
        """Create an execution plan for the given position size.

        Args:
            position_size_lots: Total position size in lots
            pair: Instrument name
            direction: LONG or SHORT
            market_data: Optional dict with avg_volume, atr_pips, volume_profile

        Returns:
            ExecutionPlan with slices ready for execution
        """
        market_data = market_data or {}

        plan = ExecutionPlan(
            pair=pair,
            direction=direction,
            strategy=self.strategy,
            total_lots=position_size_lots,
            window_seconds=self.window_seconds,
            created_at=datetime.now(timezone.utc).isoformat(),
        )

        # For small positions, single-slice execution
        if position_size_lots <= self.min_slice_lots * 3:
            plan.slice_count = 1
            plan.slices = [
                ExecutionSlice(
                    slice_index=0,
                    size_lots=position_size_lots,
                    delay_seconds=0.0,
                    weight=1.0,
                )
            ]
            return plan

        # Determine optimal slice count based on position size
        # Heuristic: ~1 slice per 0.02 lots, capped at max_slices
        raw_slices = max(MIN_SLICES, int(position_size_lots / 0.02))
        slice_count = min(raw_slices, self.max_slices)

        # Ensure each slice meets minimum lot size
        while slice_count > MIN_SLICES and (position_size_lots / slice_count) < self.min_slice_lots:
            slice_count -= 1

        plan.slice_count = slice_count

        if self.strategy == "VWAP":
            plan.slices = self._plan_vwap(position_size_lots, slice_count, market_data)
        else:
            plan.slices = self._plan_twap(position_size_lots, slice_count)

        logger.info(
            "SmartExecution: %s plan for %s — %d slices over %.0fs, total=%.4f lots",
            self.strategy, pair, slice_count, self.window_seconds, position_size_lots,
        )

        return plan

    def record_fill(
        self,
        plan: ExecutionPlan,
        slice_index: int,
        fill_price: float,
        benchmark_price: float,
        pip_value: float = 0.0001,
    ) -> ExecutionPlan:
        """Record a fill for a slice and trigger adaptive logic.

        Args:
            plan: The execution plan
            slice_index: Which slice was filled
            fill_price: Actual fill price
            benchmark_price: Expected price at time of execution
            pip_value: Pip value for the pair (default 0.0001)

        Returns:
            Updated ExecutionPlan (may have adjusted remaining slices)
        """
        if slice_index >= len(plan.slices):
            return plan

        sl = plan.slices[slice_index]
        sl.status = "filled"
        sl.fill_price = fill_price
        sl.benchmark_price = benchmark_price
        sl.executed_at = datetime.now(timezone.utc).isoformat()

        # Compute slippage
        price_diff = abs(fill_price - benchmark_price)
        sl.slippage_pips = price_diff / pip_value if pip_value > 0 else 0.0

        # Update plan totals
        plan.total_filled_lots += sl.size_lots
        _filled = [s for s in plan.slices if s.status == "filled" and s.fill_price is not None]
        if _filled:
            plan.avg_fill_price = sum(
                s.fill_price * s.size_lots for s in _filled
            ) / sum(s.size_lots for s in _filled)
        plan.total_slippage_pips += sl.slippage_pips

        # Adaptive retry: reduce remaining slices if slippage exceeds threshold
        if sl.slippage_pips > self.slippage_threshold_pips:
            plan = self._apply_adaptive_reduction(plan, slice_index)

        # Check if plan is complete
        _pending = [s for s in plan.slices if s.status == "pending"]
        if not _pending:
            plan.status = "completed"

        return plan

    def should_use_smart_execution(
        self,
        position_size_lots: float,
        atr_pips: float = 0.0,
    ) -> bool:
        """Determine if smart execution is warranted for this trade.

        Smart execution adds latency; only use for larger positions or
        volatile markets where market impact matters.

        Returns:
            True if smart execution should be used
        """
        # Large enough position to benefit from slicing
        if position_size_lots >= 0.05:
            return True
        # Moderate position in volatile market
        if position_size_lots >= 0.03 and atr_pips > 10.0:
            return True
        return False

    # ------------------------------------------------------------------
    # TWAP: Equal slices at regular intervals
    # ------------------------------------------------------------------

    def _plan_twap(
        self,
        total_lots: float,
        slice_count: int,
    ) -> List[ExecutionSlice]:
        """Generate equal-sized slices at evenly-spaced intervals."""
        slice_size = total_lots / slice_count
        interval = self.window_seconds / slice_count

        slices = []
        for i in range(slice_count):
            slices.append(
                ExecutionSlice(
                    slice_index=i,
                    size_lots=round(slice_size, 4),
                    delay_seconds=round(i * interval, 1),
                    weight=1.0 / slice_count,
                )
            )

        # Adjust last slice for rounding
        _total = sum(s.size_lots for s in slices)
        if abs(_total - total_lots) > 1e-6:
            slices[-1].size_lots = round(slices[-1].size_lots + (total_lots - _total), 4)

        return slices

    # ------------------------------------------------------------------
    # VWAP: Volume-proportional slices
    # ------------------------------------------------------------------

    def _plan_vwap(
        self,
        total_lots: float,
        slice_count: int,
        market_data: Dict[str, Any],
    ) -> List[ExecutionSlice]:
        """Generate volume-proportional slices.

        Uses volume_profile from market_data if available, otherwise
        generates a synthetic U-shaped intraday profile (high volume at
        session open/close, lower in between).
        """
        volume_profile = market_data.get("volume_profile")

        if volume_profile and len(volume_profile) >= slice_count:
            # Use actual volume profile
            weights = [float(v) for v in volume_profile[:slice_count]]
        else:
            # Synthetic U-shaped volume profile (common intraday pattern)
            weights = self._generate_u_shaped_profile(slice_count)

        # Normalize weights
        total_weight = sum(weights)
        if total_weight <= 0:
            weights = [1.0 / slice_count] * slice_count
            total_weight = 1.0
        weights = [w / total_weight for w in weights]

        interval = self.window_seconds / slice_count

        slices = []
        for i in range(slice_count):
            size = max(self.min_slice_lots, round(total_lots * weights[i], 4))
            slices.append(
                ExecutionSlice(
                    slice_index=i,
                    size_lots=size,
                    delay_seconds=round(i * interval, 1),
                    weight=round(weights[i], 4),
                )
            )

        # Normalize sizes to match total_lots exactly
        _total = sum(s.size_lots for s in slices)
        if _total > 0 and abs(_total - total_lots) > 1e-6:
            scale = total_lots / _total
            for s in slices:
                s.size_lots = round(s.size_lots * scale, 4)
            # Fix rounding residual on last slice
            _total2 = sum(s.size_lots for s in slices)
            if abs(_total2 - total_lots) > 1e-6:
                slices[-1].size_lots = round(
                    slices[-1].size_lots + (total_lots - _total2), 4
                )

        return slices

    @staticmethod
    def _generate_u_shaped_profile(n: int) -> List[float]:
        """Generate a U-shaped volume profile (high at edges, low in middle).

        Models the typical intraday FX volume pattern: high at session
        open and close, lower during mid-session.
        """
        if n <= 2:
            return [1.0] * n

        profile = []
        mid = (n - 1) / 2.0
        for i in range(n):
            # Distance from center, normalized to [0, 1]
            dist = abs(i - mid) / mid
            # U-shape: min weight 0.5 at center, max 1.5 at edges
            weight = 0.5 + dist * 1.0
            profile.append(weight)

        return profile

    # ------------------------------------------------------------------
    # Adaptive reduction
    # ------------------------------------------------------------------

    def _apply_adaptive_reduction(
        self,
        plan: ExecutionPlan,
        failed_slice_index: int,
    ) -> ExecutionPlan:
        """Reduce remaining slice sizes by 20% due to adverse slippage.

        Redistributes the reduced volume to extend the execution window,
        adding extra slices if needed.
        """
        REDUCTION_FACTOR = 0.80  # Reduce to 80%
        plan.adaptive_reductions += 1

        reduced_lots = 0.0
        for s in plan.slices:
            if s.slice_index > failed_slice_index and s.status == "pending":
                old_size = s.size_lots
                s.size_lots = round(max(self.min_slice_lots, old_size * REDUCTION_FACTOR), 4)
                reduced_lots += old_size - s.size_lots

        # If we reduced volume, add a cleanup slice at the end
        if reduced_lots > self.min_slice_lots:
            cleanup = ExecutionSlice(
                slice_index=len(plan.slices),
                size_lots=round(reduced_lots, 4),
                delay_seconds=plan.window_seconds + 30.0,  # 30s after window
                weight=0.0,  # Extra slice, not in original plan
                status="pending",
            )
            plan.slices.append(cleanup)
            plan.slice_count = len(plan.slices)
            logger.info(
                "SmartExecution: Adaptive reduction #%d — added cleanup slice "
                "of %.4f lots after slippage on slice %d",
                plan.adaptive_reductions, reduced_lots, failed_slice_index,
            )

        return plan

    # ------------------------------------------------------------------
    # Execution tracking
    # ------------------------------------------------------------------

    def get_execution_summary(self, plan: ExecutionPlan) -> Dict[str, Any]:
        """Get a summary of execution quality metrics.

        Returns:
            Dict with avg_slippage, fill_rate, vwap_vs_benchmark, etc.
        """
        filled = [s for s in plan.slices if s.status == "filled"]
        pending = [s for s in plan.slices if s.status == "pending"]
        failed = [s for s in plan.slices if s.status == "failed"]

        fill_rate = len(filled) / max(len(plan.slices), 1)
        avg_slippage = (
            sum(s.slippage_pips for s in filled) / max(len(filled), 1)
        ) if filled else 0.0

        return {
            "strategy": plan.strategy,
            "pair": plan.pair,
            "total_lots": plan.total_lots,
            "filled_lots": plan.total_filled_lots,
            "fill_rate": round(fill_rate, 3),
            "avg_fill_price": round(plan.avg_fill_price, 6),
            "avg_slippage_pips": round(avg_slippage, 4),
            "total_slippage_pips": round(plan.total_slippage_pips, 4),
            "slices_filled": len(filled),
            "slices_pending": len(pending),
            "slices_failed": len(failed),
            "adaptive_reductions": plan.adaptive_reductions,
            "status": plan.status,
        }

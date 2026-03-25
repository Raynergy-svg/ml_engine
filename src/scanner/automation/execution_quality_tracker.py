"""Execution quality tracking and feedback loop.

US-097: Track actual vs expected slippage, fill rates, and execution
timing. Feed execution quality metrics back to agent confidence
penalties. Agents whose high-confidence signals consistently result
in excessive slippage get penalized. Creates the missing execution →
learning feedback loop.

Based on 2024 TCA (Transaction Cost Analysis) research.

Metrics tracked per pair per regime:
- avg_slippage_pips: Rolling average of |expected - actual| fill price
- fill_rate: Fraction of orders fully filled (vs partial/rejected)
- timing_score: 0-1 score based on execution delay and price movement
- cost_efficiency: Ratio of realized vs expected trade cost
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_QUALITY_LOG_PATH = _PROJECT_ROOT / "trained_data" / "execution_quality.json"

# Penalty thresholds
HIGH_SLIPPAGE_THRESHOLD = 1.5  # pips — above this, penalize
LOW_FILL_RATE_THRESHOLD = 0.85  # below 85% fill rate, penalize
MAX_CONFIDENCE_PENALTY = 0.10  # Maximum penalty from execution quality
QUALITY_WINDOW = 50  # Rolling window size


@dataclass
class FillRecord:
    """Record of a single fill event."""

    pair: str = ""
    regime: str = "NORMAL"
    expected_price: float = 0.0
    fill_price: float = 0.0
    slippage_pips: float = 0.0
    fill_status: str = "filled"  # filled, partial, rejected
    execution_ms: float = 0.0
    timestamp: str = ""


@dataclass
class PairQualityStats:
    """Execution quality statistics for a pair."""

    pair: str = ""
    regime: str = "NORMAL"
    avg_slippage_pips: float = 0.0
    max_slippage_pips: float = 0.0
    fill_rate: float = 1.0
    timing_score: float = 1.0
    cost_efficiency: float = 1.0
    sample_count: int = 0
    quality_score: float = 1.0  # Combined 0-1 score

    def to_dict(self) -> dict:
        return {
            "pair": self.pair,
            "regime": self.regime,
            "avg_slippage_pips": round(self.avg_slippage_pips, 4),
            "max_slippage_pips": round(self.max_slippage_pips, 4),
            "fill_rate": round(self.fill_rate, 4),
            "timing_score": round(self.timing_score, 4),
            "cost_efficiency": round(self.cost_efficiency, 4),
            "sample_count": self.sample_count,
            "quality_score": round(self.quality_score, 4),
        }


class ExecutionQualityTracker:
    """Track execution quality and compute agent confidence penalties.

    Closes the feedback loop between execution outcomes and signal
    quality. Pairs with consistently poor execution get confidence
    penalties, naturally reducing position sizes and trade frequency.

    Usage:
        tracker = ExecutionQualityTracker()
        tracker.track_fill(pair, expected, actual, slippage, status)
        stats = tracker.get_pair_quality(pair)
        penalty = tracker.compute_agent_penalty(pair, confidence)
    """

    def __init__(
        self,
        window: int = QUALITY_WINDOW,
        persistence_path: Optional[Path] = None,
    ):
        self.window = window
        self.persistence_path = persistence_path or _QUALITY_LOG_PATH

        # Per-pair fill history: {pair: deque of FillRecord}
        self._fills: Dict[str, deque] = defaultdict(lambda: deque(maxlen=window))

        # Per-pair-regime quality cache
        self._quality_cache: Dict[str, PairQualityStats] = {}
        self._cache_dirty: Dict[str, bool] = defaultdict(lambda: True)

        # Aggregate stats
        self._total_fills: int = 0
        self._total_rejections: int = 0

        self._load_state()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def track_fill(
        self,
        pair: str,
        expected_price: float,
        fill_price: float,
        slippage_pips: float,
        fill_status: str = "filled",
        regime: str = "NORMAL",
        execution_ms: float = 0.0,
    ) -> None:
        """Log a fill event for execution quality tracking.

        Args:
            pair: Currency pair (e.g., "EUR_USD").
            expected_price: Expected fill price.
            fill_price: Actual fill price.
            slippage_pips: Slippage in pips (positive = adverse).
            fill_status: "filled", "partial", or "rejected".
            regime: Current volatility regime.
            execution_ms: Execution latency in milliseconds.
        """
        record = FillRecord(
            pair=pair,
            regime=regime,
            expected_price=expected_price,
            fill_price=fill_price,
            slippage_pips=abs(slippage_pips),
            fill_status=fill_status,
            execution_ms=execution_ms,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

        self._fills[pair].append(record)
        self._cache_dirty[pair] = True
        self._total_fills += 1
        if fill_status == "rejected":
            self._total_rejections += 1

        logger.debug(
            f"ExecQuality: {pair} [{regime}] slip={slippage_pips:+.2f} "
            f"status={fill_status} latency={execution_ms:.0f}ms"
        )

    def get_pair_quality(
        self,
        pair: str,
        regime: Optional[str] = None,
    ) -> PairQualityStats:
        """Get execution quality stats for a pair.

        Args:
            pair: Currency pair.
            regime: Filter by regime (None = all regimes).

        Returns:
            PairQualityStats with aggregated metrics.
        """
        cache_key = f"{pair}_{regime or 'ALL'}"

        if not self._cache_dirty.get(pair, True) and cache_key in self._quality_cache:
            return self._quality_cache[cache_key]

        fills = list(self._fills.get(pair, []))
        if regime:
            fills = [f for f in fills if f.regime == regime]

        stats = PairQualityStats(pair=pair, regime=regime or "ALL")

        if not fills:
            self._quality_cache[cache_key] = stats
            return stats

        stats.sample_count = len(fills)

        # Slippage stats
        slippages = [f.slippage_pips for f in fills]
        stats.avg_slippage_pips = float(np.mean(slippages))
        stats.max_slippage_pips = float(np.max(slippages))

        # Fill rate
        filled_count = sum(1 for f in fills if f.fill_status == "filled")
        stats.fill_rate = filled_count / len(fills)

        # Timing score: lower latency and lower slippage = higher score
        if fills[0].execution_ms > 0:
            latencies = [f.execution_ms for f in fills if f.execution_ms > 0]
            if latencies:
                # Normalize: <100ms = 1.0, >1000ms = 0.0
                avg_latency = float(np.mean(latencies))
                stats.timing_score = max(0.0, min(1.0, 1.0 - (avg_latency - 100) / 900))
        else:
            stats.timing_score = 0.8  # Default when no latency data

        # Cost efficiency: ratio of expected vs actual cost
        # Lower slippage = higher efficiency
        if stats.avg_slippage_pips > 0:
            stats.cost_efficiency = max(0.0, 1.0 - stats.avg_slippage_pips / 5.0)
        else:
            stats.cost_efficiency = 1.0

        # Combined quality score (weighted average)
        stats.quality_score = (
            0.40 * max(0, 1.0 - stats.avg_slippage_pips / HIGH_SLIPPAGE_THRESHOLD)
            + 0.30 * stats.fill_rate
            + 0.15 * stats.timing_score
            + 0.15 * stats.cost_efficiency
        )
        stats.quality_score = max(0.0, min(1.0, stats.quality_score))

        self._quality_cache[cache_key] = stats
        self._cache_dirty[pair] = False

        return stats

    def compute_agent_penalty(
        self,
        pair: str,
        agent_confidence: float,
        regime: Optional[str] = None,
    ) -> float:
        """Compute confidence penalty based on execution quality.

        Higher confidence signals on pairs with poor execution quality
        receive larger penalties (they're "promising more than they deliver").

        Args:
            pair: Currency pair.
            agent_confidence: Raw agent confidence (0-1).
            regime: Current regime for regime-specific stats.

        Returns:
            Confidence penalty in [0, MAX_CONFIDENCE_PENALTY].
        """
        stats = self.get_pair_quality(pair, regime)

        if stats.sample_count < 5:
            return 0.0  # Not enough data

        # Penalty increases with poor quality and high confidence
        quality_deficit = max(0.0, 1.0 - stats.quality_score)

        # Higher confidence = bigger penalty when quality is poor
        # (high-confidence bad-execution signals are the most dangerous)
        confidence_factor = agent_confidence ** 2  # Quadratic: penalize high conf more

        penalty = quality_deficit * confidence_factor * MAX_CONFIDENCE_PENALTY

        # Additional penalty for very high slippage
        if stats.avg_slippage_pips > HIGH_SLIPPAGE_THRESHOLD:
            excess = stats.avg_slippage_pips - HIGH_SLIPPAGE_THRESHOLD
            penalty += min(0.05, excess * 0.02)

        # Additional penalty for low fill rate
        if stats.fill_rate < LOW_FILL_RATE_THRESHOLD:
            deficit = LOW_FILL_RATE_THRESHOLD - stats.fill_rate
            penalty += min(0.03, deficit * 0.10)

        return min(MAX_CONFIDENCE_PENALTY, max(0.0, penalty))

    def get_worst_pairs(self, n: int = 5) -> List[PairQualityStats]:
        """Get pairs with worst execution quality."""
        all_stats = []
        for pair in self._fills:
            stats = self.get_pair_quality(pair)
            if stats.sample_count >= 5:
                all_stats.append(stats)

        all_stats.sort(key=lambda s: s.quality_score)
        return all_stats[:n]

    def get_status(self) -> Dict[str, Any]:
        """Get tracker status for observability."""
        return {
            "tracked_pairs": len(self._fills),
            "total_fills": self._total_fills,
            "total_rejections": self._total_rejections,
            "rejection_rate": round(
                self._total_rejections / max(self._total_fills, 1), 4
            ),
            "worst_pairs": [s.to_dict() for s in self.get_worst_pairs(3)],
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_state(self) -> None:
        """Persist execution quality data."""
        try:
            data = {
                "version": 1,
                "total_fills": self._total_fills,
                "total_rejections": self._total_rejections,
                "pairs": {},
                "last_updated": datetime.now(timezone.utc).isoformat(),
            }

            for pair, fills in self._fills.items():
                data["pairs"][pair] = [
                    {
                        "regime": f.regime,
                        "slippage_pips": round(f.slippage_pips, 4),
                        "fill_status": f.fill_status,
                        "execution_ms": round(f.execution_ms, 1),
                        "timestamp": f.timestamp,
                    }
                    for f in fills
                ]

            self.persistence_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                from src.scanner.automation.safe_json import safe_json_write
                safe_json_write(self.persistence_path, data)
            except ImportError:
                self.persistence_path.write_text(json.dumps(data, indent=2))
        except Exception as e:
            logger.debug(f"ExecQuality: Failed to save: {e}")

    def _load_state(self) -> None:
        """Load persisted execution quality data."""
        if not self.persistence_path.exists():
            return
        try:
            try:
                from src.scanner.automation.safe_json import safe_json_read
                data = safe_json_read(self.persistence_path, default={})
            except ImportError:
                data = json.loads(self.persistence_path.read_text(encoding="utf-8"))

            if not isinstance(data, dict):
                return

            self._total_fills = data.get("total_fills", 0)
            self._total_rejections = data.get("total_rejections", 0)

            for pair, fills in data.get("pairs", {}).items():
                for f in fills:
                    self._fills[pair].append(FillRecord(
                        pair=pair,
                        regime=f.get("regime", "NORMAL"),
                        slippage_pips=f.get("slippage_pips", 0.0),
                        fill_status=f.get("fill_status", "filled"),
                        execution_ms=f.get("execution_ms", 0.0),
                        timestamp=f.get("timestamp", ""),
                    ))

            logger.debug(f"ExecQuality: Loaded {self._total_fills} fill records")
        except Exception as e:
            logger.debug(f"ExecQuality: Failed to load: {e}")

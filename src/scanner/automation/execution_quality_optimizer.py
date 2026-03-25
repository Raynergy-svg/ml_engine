"""Execution quality cost profiling by model and regime.

US-109: Track execution cost (slippage, fill latency, partial fills)
per model+regime combination. Builds cost profiles and recommends
gate adjustments when execution quality degrades.

Approach:
1. Record execution metrics segmented by (pair, regime, model)
2. Build rolling cost profiles per regime+model
3. Compute cost_score [0-1] where 0=no cost, 1=maximum cost
4. Recommend confidence_threshold delta: raise threshold when costs high
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_LOG_PATH = _PROJECT_ROOT / "trained_data" / "execution_quality_optimizer_log.json"

# Cost profiling parameters
ROLLING_WINDOW = 50  # Executions per profile bucket
SLIPPAGE_WEIGHT = 0.50  # Weight in cost score
FILL_TIME_WEIGHT = 0.30  # Weight in cost score
FILL_RATE_WEIGHT = 0.20  # Weight in cost score

# Gate adjustment parameters
COST_THRESHOLD_HIGH = 0.60  # Cost score above this → raise confidence threshold
COST_THRESHOLD_LOW = 0.25   # Cost score below this → can lower threshold
MAX_THRESHOLD_DELTA = 0.05  # Maximum confidence threshold adjustment per step
SLIPPAGE_NORM = 3.0         # Normalize slippage: 3 pips = max expected
FILL_TIME_NORM = 2000.0     # Normalize fill time: 2000ms = max expected


class ExecutionQualityOptimizer:
    """Profiles execution costs by model+regime and recommends gate adjustments.

    Bridges execution_quality_tracker (raw metrics) with gate tuning
    (confidence thresholds). When execution costs rise in a regime,
    recommends tighter gates to avoid low-quality fills.

    Usage:
        optimizer = ExecutionQualityOptimizer()
        optimizer.record_execution("EUR_USD", "HIGH", "TCN", 1.2, 450, 0.95)
        profile = optimizer.get_cost_profile("HIGH", "TCN")
        delta = optimizer.recommend_gate_adjustment("HIGH")
    """

    def __init__(self, persistence_path: Optional[Path] = None):
        self.persistence_path = persistence_path or _LOG_PATH

        # Cost data: {regime: {model: [records]}}
        self._cost_data: Dict[str, Dict[str, List[Dict[str, float]]]] = defaultdict(
            lambda: defaultdict(list)
        )

        # Cached cost profiles
        self._profiles: Dict[str, Dict[str, Dict[str, float]]] = {}

        # Recommendation history
        self._recommendations: List[Dict[str, Any]] = []
        self._total_records: int = 0

        self._load_state()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record_execution(
        self,
        pair: str,
        regime: str,
        model: str,
        slippage_pips: float,
        fill_time_ms: float,
        fill_rate: float,
    ) -> None:
        """Record an execution event for cost profiling.

        Args:
            pair: Currency pair.
            regime: Market regime (LOW/NORMAL/HIGH/EXTREME).
            model: Model that generated the signal (TCN/Ridge/RF/ALL).
            slippage_pips: Absolute slippage in pips.
            fill_time_ms: Fill time in milliseconds.
            fill_rate: Fill rate [0.0, 1.0] where 1.0 = fully filled.
        """
        record = {
            "pair": pair,
            "slippage": abs(slippage_pips),
            "fill_time": max(0.0, fill_time_ms),
            "fill_rate": float(np.clip(fill_rate, 0.0, 1.0)),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        self._cost_data[regime][model].append(record)

        # Trim to rolling window
        if len(self._cost_data[regime][model]) > ROLLING_WINDOW * 2:
            self._cost_data[regime][model] = self._cost_data[regime][model][
                -ROLLING_WINDOW:
            ]

        self._total_records += 1

        # Invalidate cached profile
        cache_key = f"{regime}_{model}"
        self._profiles.pop(cache_key, None)

    def get_cost_profile(
        self, regime: str, model: str
    ) -> Dict[str, float]:
        """Get cost profile for a regime+model combination.

        Returns:
            Dict with avg_slippage, avg_fill_time, avg_fill_rate, cost_score.
        """
        cache_key = f"{regime}_{model}"
        if cache_key in self._profiles:
            return self._profiles[cache_key]

        records = self._cost_data.get(regime, {}).get(model, [])
        if not records:
            profile = {
                "avg_slippage": 0.0,
                "avg_fill_time": 0.0,
                "avg_fill_rate": 1.0,
                "cost_score": 0.0,
                "n_records": 0,
            }
            self._profiles[cache_key] = profile
            return profile

        recent = records[-ROLLING_WINDOW:]

        avg_slippage = float(np.mean([r["slippage"] for r in recent]))
        avg_fill_time = float(np.mean([r["fill_time"] for r in recent]))
        avg_fill_rate = float(np.mean([r["fill_rate"] for r in recent]))

        # Compute cost score [0, 1]
        slip_score = min(avg_slippage / SLIPPAGE_NORM, 1.0)
        time_score = min(avg_fill_time / FILL_TIME_NORM, 1.0)
        rate_score = 1.0 - avg_fill_rate  # Lower fill rate = higher cost

        cost_score = (
            SLIPPAGE_WEIGHT * slip_score
            + FILL_TIME_WEIGHT * time_score
            + FILL_RATE_WEIGHT * rate_score
        )

        profile = {
            "avg_slippage": round(avg_slippage, 4),
            "avg_fill_time": round(avg_fill_time, 1),
            "avg_fill_rate": round(avg_fill_rate, 4),
            "cost_score": round(float(np.clip(cost_score, 0.0, 1.0)), 4),
            "n_records": len(recent),
        }

        self._profiles[cache_key] = profile
        return profile

    def recommend_gate_adjustment(self, regime: str) -> Dict[str, Any]:
        """Recommend confidence threshold adjustment based on execution costs.

        When costs are high → raise confidence threshold (be more selective).
        When costs are low → allow slight lowering (more opportunities).

        Args:
            regime: Market regime.

        Returns:
            Dict with recommended_delta, reason, cost_scores per model.
        """
        models = list(self._cost_data.get(regime, {}).keys())
        if not models:
            return {
                "recommended_delta": 0.0,
                "reason": "no_data",
                "model_costs": {},
            }

        model_costs = {}
        for model in models:
            profile = self.get_cost_profile(regime, model)
            model_costs[model] = profile["cost_score"]

        # Aggregate cost: weighted average across models
        avg_cost = float(np.mean(list(model_costs.values())))

        # Compute recommended delta
        if avg_cost > COST_THRESHOLD_HIGH:
            # High cost → raise threshold
            intensity = (avg_cost - COST_THRESHOLD_HIGH) / (1.0 - COST_THRESHOLD_HIGH)
            delta = MAX_THRESHOLD_DELTA * intensity
            reason = "high_execution_cost"
        elif avg_cost < COST_THRESHOLD_LOW:
            # Low cost → can lower threshold slightly
            intensity = (COST_THRESHOLD_LOW - avg_cost) / COST_THRESHOLD_LOW
            delta = -MAX_THRESHOLD_DELTA * 0.5 * intensity  # More conservative lowering
            reason = "low_execution_cost"
        else:
            delta = 0.0
            reason = "normal_execution_cost"

        delta = round(float(np.clip(delta, -MAX_THRESHOLD_DELTA, MAX_THRESHOLD_DELTA)), 4)

        recommendation = {
            "recommended_delta": delta,
            "reason": reason,
            "avg_cost_score": round(avg_cost, 4),
            "model_costs": {m: round(c, 4) for m, c in model_costs.items()},
            "regime": regime,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        self._recommendations.append(recommendation)
        if len(self._recommendations) > 200:
            self._recommendations = self._recommendations[-100:]

        return recommendation

    def get_status(self) -> Dict[str, Any]:
        """Get optimizer status for observability."""
        regime_models = {}
        for regime, models in self._cost_data.items():
            regime_models[regime] = {
                m: len(records) for m, records in models.items()
            }

        return {
            "total_records": self._total_records,
            "regime_models": regime_models,
            "recent_recommendations": self._recommendations[-5:],
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_state(self) -> None:
        """Persist cost profiles and recommendations."""
        try:
            # Convert defaultdict to regular dict for JSON serialization
            cost_data_plain = {}
            for regime, models in self._cost_data.items():
                cost_data_plain[regime] = {}
                for model, records in models.items():
                    cost_data_plain[regime][model] = records[-ROLLING_WINDOW:]

            data = {
                "version": 1,
                "total_records": self._total_records,
                "cost_data": cost_data_plain,
                "recommendations": self._recommendations[-100:],
                "last_updated": datetime.now(timezone.utc).isoformat(),
            }
            self.persistence_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                from src.scanner.automation.safe_json import safe_json_write
                safe_json_write(self.persistence_path, data)
            except ImportError:
                self.persistence_path.write_text(json.dumps(data, indent=2))
        except Exception as e:
            logger.warning(f"ExecutionQualityOptimizer: Failed to save: {e}")

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
                self._total_records = data.get("total_records", 0)
                self._recommendations = data.get("recommendations", [])
                raw_cost = data.get("cost_data", {})
                for regime, models in raw_cost.items():
                    if not isinstance(models, dict):
                        continue
                    for model, records in models.items():
                        if not isinstance(records, list):
                            continue
                        valid = [
                            r for r in records
                            if isinstance(r, dict)
                            and all(k in r for k in ("slippage", "fill_time", "fill_rate"))
                        ]
                        if len(valid) < len(records):
                            logger.warning(
                                f"ExecutionQualityOptimizer: Skipped {len(records) - len(valid)} "
                                f"malformed records for {regime}/{model}"
                            )
                        self._cost_data[regime][model] = valid
        except Exception as e:
            logger.warning(f"ExecutionQualityOptimizer: Failed to load: {e}")

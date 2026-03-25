"""Adaptive gate threshold optimizer.

US-110: Auto-tune gate thresholds (confidence, momentum, R:R) based
on rolling win rate vs target. Tracks win rate per regime and adjusts
thresholds up (tighter) when above target, down (looser) when below,
within safety bounds.

Approach:
1. Record trade outcomes with their gate values per regime
2. Compute rolling win rate per regime
3. Compare to target win rate (default 55%)
4. Adjust thresholds: above target → tighten (fewer but better trades),
   below target → loosen (more opportunities to recover sample)
5. All adjustments bounded by safety limits
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
_LOG_PATH = _PROJECT_ROOT / "trained_data" / "threshold_optimizer_log.json"

# Rolling window for win rate computation
ROLLING_WINDOW = 40
MIN_SAMPLES = 15  # Minimum trades before optimization activates

# Target win rate
TARGET_WIN_RATE = 0.55

# Safety bounds for each threshold
BOUNDS = {
    "confidence": (0.40, 0.85),
    "momentum": (0.10, 0.60),
    "rr_ratio": (1.0, 2.5),
}

# Adjustment parameters
ADJUSTMENT_LR = 0.02  # How much to move per optimization step
TIGHTEN_MULTIPLIER = 1.0  # Full speed when tightening (above target)
LOOSEN_MULTIPLIER = 0.5   # Half speed when loosening (below target — conservative)

# Default starting thresholds (matches config.py defaults)
DEFAULT_THRESHOLDS = {
    "confidence": 0.55,
    "momentum": 0.30,
    "rr_ratio": 1.2,
}


class ThresholdOptimizer:
    """Auto-tunes gate thresholds based on rolling win rate.

    Connects trade outcome tracking with gate configuration. When
    win rate exceeds target, tightens thresholds to be more selective
    (higher quality trades). When below target, loosens slightly to
    capture more opportunities.

    Usage:
        optimizer = ThresholdOptimizer()
        optimizer.update_outcome("HIGH", {"confidence": 0.62, "momentum": 0.35, "rr_ratio": 1.5}, won=True)
        thresholds = optimizer.optimize_thresholds("HIGH")
    """

    def __init__(self, persistence_path: Optional[Path] = None):
        self.persistence_path = persistence_path or _LOG_PATH

        # Outcome history: {regime: [records]}
        self._outcomes: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

        # Current optimized thresholds: {regime: {threshold: value}}
        self._thresholds: Dict[str, Dict[str, float]] = {}

        # Optimization log
        self._optimization_log: List[Dict[str, Any]] = []
        self._total_outcomes: int = 0

        self._load_state()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update_outcome(
        self,
        regime: str,
        gate_values: Dict[str, float],
        won: bool,
    ) -> None:
        """Record a trade outcome with its gate values.

        Args:
            regime: Market regime when trade was taken.
            gate_values: Dict with confidence, momentum, rr_ratio at entry.
            won: Whether the trade was a winner.
        """
        record = {
            "confidence": gate_values.get("confidence", 0.0),
            "momentum": gate_values.get("momentum", 0.0),
            "rr_ratio": gate_values.get("rr_ratio", 0.0),
            "won": won,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        self._outcomes[regime].append(record)

        # Trim to rolling window
        if len(self._outcomes[regime]) > ROLLING_WINDOW * 2:
            self._outcomes[regime] = self._outcomes[regime][-ROLLING_WINDOW:]

        self._total_outcomes += 1

    def optimize_thresholds(self, regime: str) -> Dict[str, float]:
        """Compute optimized thresholds for a regime.

        Returns:
            Dict with optimized confidence, momentum, rr_ratio thresholds.
        """
        outcomes = self._outcomes.get(regime, [])

        # Not enough data — return current or defaults
        if len(outcomes) < MIN_SAMPLES:
            return self._get_current_thresholds(regime)

        recent = outcomes[-ROLLING_WINDOW:]
        win_rate = sum(1 for r in recent if r["won"]) / len(recent)

        current = self._get_current_thresholds(regime)
        new_thresholds = dict(current)

        # Compute adjustment direction and magnitude
        delta_wr = win_rate - TARGET_WIN_RATE

        if delta_wr > 0:
            # Above target → tighten (raise thresholds)
            multiplier = TIGHTEN_MULTIPLIER
            direction = 1.0
        else:
            # Below target → loosen (lower thresholds)
            multiplier = LOOSEN_MULTIPLIER
            direction = -1.0

        intensity = abs(delta_wr) * multiplier

        for key in ["confidence", "momentum", "rr_ratio"]:
            lo, hi = BOUNDS[key]
            # Normalize step by threshold range so each moves proportionally
            step = ADJUSTMENT_LR * intensity * direction * (hi - lo)
            new_val = current[key] + step
            new_thresholds[key] = round(float(np.clip(new_val, lo, hi)), 4)

        # Store updated thresholds
        self._thresholds[regime] = new_thresholds

        # Log optimization
        log_entry = {
            "regime": regime,
            "win_rate": round(win_rate, 4),
            "target": TARGET_WIN_RATE,
            "delta_wr": round(delta_wr, 4),
            "old_thresholds": {k: round(v, 4) for k, v in current.items()},
            "new_thresholds": {k: round(v, 4) for k, v in new_thresholds.items()},
            "n_samples": len(recent),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self._optimization_log.append(log_entry)
        if len(self._optimization_log) > 200:
            self._optimization_log = self._optimization_log[-100:]

        # Phase 26 (US-161): Check for threshold drift velocity
        self._check_drift_velocity(regime, current, new_thresholds)

        return new_thresholds

    def _check_drift_velocity(
        self,
        regime: str,
        old_thresholds: Dict[str, float],
        new_thresholds: Dict[str, float],
    ) -> None:
        """Phase 26 (US-161): Monitor threshold drift velocity.

        Tracks cumulative change over recent optimizations and alerts
        when velocity exceeds 0.01/cycle for 3+ consecutive changes.
        """
        _VELOCITY_ALERT = 0.01  # Max acceptable change per cycle
        _MIN_CONSECUTIVE = 3

        if not hasattr(self, "_drift_history"):
            self._drift_history: Dict[str, List[Dict[str, Any]]] = {}

        if regime not in self._drift_history:
            self._drift_history[regime] = []

        for gate_name in ["confidence", "momentum", "rr_ratio"]:
            _old = old_thresholds.get(gate_name, 0.0)
            _new = new_thresholds.get(gate_name, 0.0)
            _delta = _new - _old

            _history_key = f"{regime}_{gate_name}"
            if _history_key not in self._drift_history:
                self._drift_history[_history_key] = []

            self._drift_history[_history_key].append({
                "delta": round(_delta, 6),
                "new_value": round(_new, 4),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })
            # Keep last 20 entries
            if len(self._drift_history[_history_key]) > 20:
                self._drift_history[_history_key] = self._drift_history[_history_key][-20:]

            recent = self._drift_history[_history_key][-_MIN_CONSECUTIVE:]
            if len(recent) >= _MIN_CONSECUTIVE:
                _cum_delta = sum(r["delta"] for r in recent)
                _velocity = abs(_cum_delta) / len(recent)
                _direction = "loosening" if _cum_delta < 0 else "tightening"

                if _velocity > _VELOCITY_ALERT:
                    logger.warning(
                        "US-161: Threshold drift alert — %s/%s %s "
                        "velocity=%.4f/cycle (cumulative=%.4f over %d changes)",
                        regime, gate_name, _direction,
                        _velocity, _cum_delta, len(recent),
                    )
                    try:
                        from src.scanner.automation.observation_log import ObservationLog
                        ObservationLog().log_observation(
                            pair="SYSTEM",
                            category="threshold_drift_alert",
                            description=(
                                f"US-161: {regime}/{gate_name} {_direction} "
                                f"at velocity {_velocity:.4f}/cycle"
                            ),
                            metadata={
                                "regime": regime,
                                "gate": gate_name,
                                "direction": _direction,
                                "velocity": round(_velocity, 6),
                                "cumulative_delta": round(_cum_delta, 6),
                                "consecutive_changes": len(recent),
                            },
                        )
                    except Exception:
                        pass

    def get_win_rate(self, regime: str) -> Optional[float]:
        """Get current rolling win rate for a regime."""
        outcomes = self._outcomes.get(regime, [])
        if len(outcomes) < MIN_SAMPLES:
            return None
        recent = outcomes[-ROLLING_WINDOW:]
        return round(sum(1 for r in recent if r["won"]) / len(recent), 4)

    def get_regime_report(self) -> Dict[str, Any]:
        """Get threshold status for all regimes."""
        report = {}
        for regime in self._outcomes:
            win_rate = self.get_win_rate(regime)
            report[regime] = {
                "win_rate": win_rate,
                "n_outcomes": len(self._outcomes[regime]),
                "thresholds": self._get_current_thresholds(regime),
                "optimized": win_rate is not None,
            }
        return report

    def get_status(self) -> Dict[str, Any]:
        """Get optimizer status for observability."""
        return {
            "total_outcomes": self._total_outcomes,
            "regimes": list(self._outcomes.keys()),
            "regime_report": self.get_regime_report(),
            "recent_optimizations": self._optimization_log[-5:],
        }

    # ------------------------------------------------------------------
    # Phase 22 (US-136): Journal replay for warm-start
    # ------------------------------------------------------------------

    def replay_from_journal(self, journal_path: Optional[Path] = None) -> int:
        """Warm-start optimizer from historical trade journal data.

        Reads trade_journal_rl.json entries, extracts regime + gate values + won,
        and calls update_outcome() for each historical trade. This allows the
        optimizer to learn from past trades even when initialized with empty state.

        Args:
            journal_path: Path to trade journal. Defaults to trained_data/trade_journal_rl.json.

        Returns:
            Number of trades replayed.
        """
        if journal_path is None:
            journal_path = _PROJECT_ROOT / "trained_data" / "trade_journal_rl.json"

        if not journal_path.exists():
            logger.debug("ThresholdOptimizer: No journal found for replay")
            return 0

        try:
            try:
                from src.scanner.automation.safe_json import safe_json_read
                entries = safe_json_read(journal_path, default=[])
            except ImportError:
                entries = json.loads(journal_path.read_text(encoding="utf-8"))

            if not isinstance(entries, list):
                logger.warning("ThresholdOptimizer: Journal is not a list, skipping replay")
                return 0

            replayed = 0
            for entry in entries:
                if not isinstance(entry, dict):
                    continue

                # Extract outcome
                outcome = entry.get("outcome")
                if not isinstance(outcome, dict) or "trade_won" not in outcome:
                    continue

                # Extract regime
                regime_data = entry.get("regime", {})
                regime = regime_data.get("volatility_regime", "UNKNOWN") if isinstance(regime_data, dict) else "UNKNOWN"
                if not regime or regime == "UNKNOWN":
                    # Try to infer from top-level metadata if available
                    regime = entry.get("volatility_regime", "UNKNOWN")

                # Extract gate values
                confidence = float(entry.get("confidence", 0.0))
                # Compute R:R from SL/TP distances
                sl_price = float(entry.get("sl_price", 0.0))
                tp_price = float(entry.get("tp_price", 0.0))
                entry_price = float(entry.get("entry_price", 0.0))
                sl_dist = abs(sl_price - entry_price) if sl_price and entry_price else 0
                tp_dist = abs(tp_price - entry_price) if tp_price and entry_price else 0
                rr_ratio = (tp_dist / sl_dist) if sl_dist > 0 else 1.2

                # Momentum: derive from gates or default
                gates = entry.get("gates", {})
                momentum_val = 0.35 if gates.get("momentum_passed", False) else 0.15

                gate_values = {
                    "confidence": confidence,
                    "momentum": momentum_val,
                    "rr_ratio": round(rr_ratio, 4),
                }

                won = bool(outcome.get("trade_won", False))
                self.update_outcome(regime, gate_values, won)
                replayed += 1

            if replayed > 0:
                logger.info(
                    f"ThresholdOptimizer: Replayed {replayed} trades from journal "
                    f"(regimes: {list(self._outcomes.keys())})"
                )

            return replayed

        except Exception as e:
            logger.warning(f"ThresholdOptimizer: Journal replay failed: {e}")
            return 0

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _get_current_thresholds(self, regime: str) -> Dict[str, float]:
        """Get current thresholds for regime, with defaults."""
        if regime in self._thresholds:
            return dict(self._thresholds[regime])
        return dict(DEFAULT_THRESHOLDS)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_state(self) -> None:
        """Persist optimizer state."""
        try:
            outcomes_plain = {
                regime: records[-ROLLING_WINDOW:]
                for regime, records in self._outcomes.items()
            }

            data = {
                "version": 1,
                "total_outcomes": self._total_outcomes,
                "outcomes": outcomes_plain,
                "thresholds": self._thresholds,
                "optimization_log": self._optimization_log[-100:],
                "last_updated": datetime.now(timezone.utc).isoformat(),
            }
            self.persistence_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                from src.scanner.automation.safe_json import safe_json_write
                safe_json_write(self.persistence_path, data)
            except ImportError:
                self.persistence_path.write_text(json.dumps(data, indent=2))
        except Exception as e:
            logger.warning(f"ThresholdOptimizer: Failed to save: {e}")

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
                self._total_outcomes = data.get("total_outcomes", 0)
                self._thresholds = data.get("thresholds", {})
                self._optimization_log = data.get("optimization_log", [])
                raw_outcomes = data.get("outcomes", {})
                for regime, records in raw_outcomes.items():
                    if not isinstance(records, list):
                        continue
                    valid = [
                        r for r in records
                        if isinstance(r, dict) and "won" in r
                    ]
                    if len(valid) < len(records):
                        logger.warning(
                            f"ThresholdOptimizer: Skipped {len(records) - len(valid)} "
                            f"malformed records for regime {regime}"
                        )
                    self._outcomes[regime] = valid
        except Exception as e:
            logger.warning(f"ThresholdOptimizer: Failed to load: {e}")

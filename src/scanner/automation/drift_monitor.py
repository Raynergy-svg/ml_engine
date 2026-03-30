"""Drift monitor with auto-correction feedback.

US-125: Wire ReplayValidator drift detection into periodic checks that
feed results back to threshold_optimizer and config_tuner for auto-correction.
If drift exceeds critical threshold, reduce position sizing automatically.

Approach:
1. Periodically run replay scans on cached historical candle data
2. Compare current win rate to baseline → detect drift
3. Mild drift → recommend config adjustments (feed to config_tuner)
4. Critical drift (>20%) → force position sizing reduction + alert
5. Track drift trends across pairs for system-wide awareness
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Lazy import DriftProjector (non-blocking, optional)
_PROJECTOR = None


def _get_projector():
    """Lazy-load DriftProjector to avoid import overhead if not needed."""
    global _PROJECTOR
    if _PROJECTOR is None:
        try:
            from src.scanner.automation.drift_projector import DriftProjector

            _PROJECTOR = DriftProjector()
        except ImportError:
            logger.debug("DriftProjector not available; forward projections disabled")
            _PROJECTOR = False  # Mark as unavailable, don't retry
    return _PROJECTOR if _PROJECTOR is not False else None

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_LOG_PATH = _PROJECT_ROOT / "trained_data" / "drift_monitor_log.json"

# Drift thresholds
MILD_DRIFT = 0.10      # 10% accuracy drop → recommend adjustments
CRITICAL_DRIFT = 0.20  # 20% accuracy drop → force sizing reduction
BASELINE_ACCURACY = 0.55  # Default expected accuracy

# Position sizing reduction on critical drift
CRITICAL_SIZE_MULTIPLIER = 0.60  # Reduce to 60% of normal sizing


class DriftMonitor:
    """Monitors model drift via ReplayValidator and auto-corrects.

    Bridges the ReplayValidator (US-120) with the config_tuner (US-010)
    and threshold_optimizer (US-110) to close the drift → correction loop.
    When drift exceeds thresholds, adjustments are applied automatically.

    Usage:
        monitor = DriftMonitor(replay_validator, config_tuner, threshold_optimizer)
        # Every 20 cycles:
        report = monitor.run_drift_check("EUR_USD", recent_candles)
        if report["action_taken"]:
            logger.info("Drift correction applied: %s", report["actions"])
    """

    def __init__(
        self,
        replay_validator: Any = None,
        config_tuner: Any = None,
        threshold_optimizer: Any = None,
        persistence_path: Optional[Path] = None,
    ):
        self._replay_validator = replay_validator
        self._config_tuner = config_tuner
        self._threshold_optimizer = threshold_optimizer
        self.persistence_path = persistence_path or _LOG_PATH

        # State
        self._drift_history: Dict[str, List[Dict[str, Any]]] = {}  # pair → drift records
        self._corrections_applied: int = 0
        self._critical_alerts: int = 0
        self._total_checks: int = 0
        self._active_size_reductions: Dict[str, float] = {}  # pair → multiplier

        self._load_state()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_drift_check(
        self,
        pair: str,
        historical_candles: List[Dict[str, Any]],
        scan_config: Optional[Dict[str, Any]] = None,
        baseline_accuracy: float = BASELINE_ACCURACY,
    ) -> Dict[str, Any]:
        """Run drift detection on a pair and apply corrections if needed.

        Args:
            pair: Currency pair.
            historical_candles: Recent candle data for replay.
            scan_config: Current scan configuration.
            baseline_accuracy: Expected win rate baseline.

        Returns:
            Dict with drift_score, actions_taken, and recommendations.
        """
        if self._replay_validator is None:
            return {"error": "no_replay_validator", "action_taken": False}

        self._total_checks += 1
        report: Dict[str, Any] = {
            "pair": pair,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "action_taken": False,
            "actions": [],
            "drift_score": 0.0,
            "is_drifting": False,
            "severity": "none",
        }

        try:
            # Run replay scan
            replay_result = self._replay_validator.replay_scan(
                pair=pair,
                historical_candles=historical_candles,
                scan_config=scan_config,
            )

            # Detect drift
            drift = self._replay_validator.detect_drift(
                replay_result=replay_result,
                baseline_accuracy=baseline_accuracy,
            )

            report["drift_score"] = drift.get("drift_score", 0.0)
            report["is_drifting"] = drift.get("is_drifting", False)
            report["current_accuracy"] = drift.get("current_accuracy", 0.0)
            report["n_replay_trades"] = drift.get("n_trades", 0)

            drift_score = abs(drift.get("drift_score", 0.0))

            # Classify severity
            if drift_score >= CRITICAL_DRIFT:
                report["severity"] = "critical"
                self._apply_critical_correction(pair, drift_score, report)
            elif drift_score >= MILD_DRIFT:
                report["severity"] = "mild"
                self._apply_mild_correction(pair, drift_score, report)
            else:
                report["severity"] = "none"
                # Clear any active size reduction if drift has recovered
                if pair in self._active_size_reductions:
                    del self._active_size_reductions[pair]
                    report["actions"].append("size_reduction_cleared")

            # Track drift history per pair
            if pair not in self._drift_history:
                self._drift_history[pair] = []
            self._drift_history[pair].append({
                "drift_score": report["drift_score"],
                "severity": report["severity"],
                "timestamp": report["timestamp"],
                "current_accuracy": report.get("current_accuracy"),
            })
            # Keep last 20 per pair
            if len(self._drift_history[pair]) > 20:
                self._drift_history[pair] = self._drift_history[pair][-10:]

            # Generate forward projection (non-blocking)
            try:
                projector = _get_projector()
                if projector is not None:
                    curve = projector.project(
                        pair=pair,
                        current_accuracy=report.get("current_accuracy", baseline_accuracy),
                        baseline_accuracy=baseline_accuracy,
                    )
                    report["projection"] = {
                        "hours_until_critical": curve.hours_until_critical(),
                        "recommended_action": curve.recommended_action,
                        "action_urgency": curve.action_urgency,
                        "t6h_accuracy": (
                            curve.projection_points[0].predicted_accuracy
                            if curve.projection_points
                            else None
                        ),
                        "t24h_accuracy": (
                            curve.projection_points[2].predicted_accuracy
                            if len(curve.projection_points) > 2
                            else None
                        ),
                        "risk_factors": curve.risk_factors,
                    }
            except Exception as proj_err:
                logger.debug(f"DriftMonitor: Projection failed for {pair}: {proj_err}")

        except Exception as e:
            report["error"] = str(e)[:200]
            logger.debug(f"DriftMonitor: Check failed for {pair}: {e}")

        return report

    def get_size_multiplier(self, pair: str) -> float:
        """Get position sizing multiplier for a pair based on drift state.

        Returns:
            Multiplier in [0.60, 1.0]. Lower = more drift detected.
        """
        return self._active_size_reductions.get(pair, 1.0)

    def get_status(self) -> Dict[str, Any]:
        """Get monitor status for observability."""
        return {
            "total_checks": self._total_checks,
            "corrections_applied": self._corrections_applied,
            "critical_alerts": self._critical_alerts,
            "active_size_reductions": dict(self._active_size_reductions),
            "pairs_monitored": len(self._drift_history),
            "recent_drift": {
                pair: records[-1] if records else None
                for pair, records in self._drift_history.items()
            },
        }

    # ------------------------------------------------------------------
    # Internal correction logic
    # ------------------------------------------------------------------

    def _apply_mild_correction(
        self, pair: str, drift_score: float, report: Dict[str, Any],
    ) -> None:
        """Apply mild drift corrections: recommend config adjustments."""
        report["action_taken"] = True
        self._corrections_applied += 1

        # Recommend tightening confidence threshold
        if self._threshold_optimizer is not None:
            try:
                # Drift detected → tighten gates slightly
                self._threshold_optimizer.apply_manual_adjustment(
                    param="confidence",
                    delta=0.01,
                    reason=f"mild_drift_{pair}_{drift_score:.3f}",
                )
                report["actions"].append("confidence_threshold_tightened_+0.01")
            except Exception:
                pass

        # Feed recommendation to config_tuner
        if self._config_tuner is not None:
            try:
                self._config_tuner.propose_adjustment(
                    param="reduce_exposure",
                    delta=-0.05,
                    reason=f"mild_drift_detected_{pair}",
                )
                report["actions"].append("config_tuner_notified")
            except Exception:
                pass

        logger.info(
            "DriftMonitor: Mild drift on %s (score=%.3f) — %d corrections applied",
            pair, drift_score, len(report["actions"]),
        )

    def _apply_critical_correction(
        self, pair: str, drift_score: float, report: Dict[str, Any],
    ) -> None:
        """Apply critical drift corrections: force sizing reduction + alert."""
        report["action_taken"] = True
        self._corrections_applied += 1
        self._critical_alerts += 1

        # Force position sizing reduction
        self._active_size_reductions[pair] = CRITICAL_SIZE_MULTIPLIER
        report["actions"].append(
            f"position_sizing_reduced_to_{CRITICAL_SIZE_MULTIPLIER:.0%}"
        )

        # Tighten thresholds more aggressively
        if self._threshold_optimizer is not None:
            try:
                self._threshold_optimizer.apply_manual_adjustment(
                    param="confidence",
                    delta=0.03,
                    reason=f"critical_drift_{pair}_{drift_score:.3f}",
                )
                report["actions"].append("confidence_threshold_tightened_+0.03")
            except Exception:
                pass

        # Alert via config_tuner
        if self._config_tuner is not None:
            try:
                self._config_tuner.propose_adjustment(
                    param="critical_drift_alert",
                    delta=-0.10,
                    reason=f"CRITICAL_drift_{pair}_score_{drift_score:.3f}",
                )
                report["actions"].append("critical_alert_raised")
            except Exception:
                pass

        logger.warning(
            "DriftMonitor: CRITICAL drift on %s (score=%.3f) — "
            "sizing reduced to %.0f%%, %d actions taken",
            pair, drift_score, CRITICAL_SIZE_MULTIPLIER * 100, len(report["actions"]),
        )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_state(self) -> None:
        """Persist monitor state."""
        try:
            data = {
                "version": 1,
                "total_checks": self._total_checks,
                "corrections_applied": self._corrections_applied,
                "critical_alerts": self._critical_alerts,
                "active_size_reductions": self._active_size_reductions,
                "drift_history": {
                    pair: records[-10:]
                    for pair, records in self._drift_history.items()
                },
                "last_updated": datetime.now(timezone.utc).isoformat(),
            }
            self.persistence_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                from src.scanner.automation.safe_json import safe_json_write
                safe_json_write(self.persistence_path, data)
            except ImportError:
                self.persistence_path.write_text(json.dumps(data, indent=2))
        except Exception as e:
            logger.debug(f"DriftMonitor: Failed to save: {e}")

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
                self._total_checks = data.get("total_checks", 0)
                self._corrections_applied = data.get("corrections_applied", 0)
                self._critical_alerts = data.get("critical_alerts", 0)
                self._active_size_reductions = data.get("active_size_reductions", {})
                self._drift_history = data.get("drift_history", {})
        except Exception as e:
            logger.debug(f"DriftMonitor: Failed to load: {e}")

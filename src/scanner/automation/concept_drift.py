"""Anomaly detection and concept drift monitoring.

US-083: Market regimes shift and model assumptions become stale
(concept drift). Implements rolling z-score monitoring on prediction
errors, agent disagreement, and microstructure anomalies. When drift
is detected, trigger recalibration alerts and reduce confidence until
models re-adapt.

Detection method: rolling z-score on 3 signal streams:
1. prediction_error — abs(predicted_confidence - actual_outcome)
2. agent_disagreement — spread of agent votes (high = disagreement)
3. micro_anomaly — microstructure deviation from baseline

Rolling window: 100 observations
Drift threshold: z > 2.5 on any stream

Severity levels:
- LOW: 1 stream drifting → 0.85 confidence penalty
- MEDIUM: 2 streams drifting → 0.70 confidence penalty
- HIGH: 3 streams drifting → 0.50 confidence penalty
"""

from __future__ import annotations

import json
import logging
import math
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_DRIFT_LOG_PATH = _PROJECT_ROOT / "trained_data" / "concept_drift_log.json"

# Detection parameters
ROLLING_WINDOW = 100
DRIFT_Z_THRESHOLD = 2.5
WARMUP_OBSERVATIONS = 20  # Need this many before detection activates

# Severity → confidence multiplier
SEVERITY_PENALTIES: Dict[str, float] = {
    "NONE": 1.0,
    "LOW": 0.85,
    "MEDIUM": 0.70,
    "HIGH": 0.50,
}


@dataclass
class DriftResult:
    """Result of concept drift detection."""

    drift_detected: bool = False
    severity: str = "NONE"  # NONE, LOW, MEDIUM, HIGH
    confidence_penalty: float = 1.0  # Multiplier (< 1.0 means reduce)
    drifting_streams: List[str] = field(default_factory=list)
    z_scores: Dict[str, float] = field(default_factory=dict)
    observations_count: int = 0
    reason: str = ""

    def to_dict(self) -> dict:
        return {
            "drift_detected": self.drift_detected,
            "severity": self.severity,
            "confidence_penalty": round(self.confidence_penalty, 4),
            "drifting_streams": self.drifting_streams,
            "z_scores": {k: round(v, 4) for k, v in self.z_scores.items()},
            "observations_count": self.observations_count,
            "reason": self.reason,
        }


class ConceptDriftDetector:
    """Rolling z-score concept drift detector.

    Monitors 3 signal streams for distributional shift. When the
    z-score of recent observations exceeds the threshold, drift is
    flagged and a confidence penalty is applied.

    Thread-safe: uses deques with fixed maxlen for rolling windows.
    """

    def __init__(
        self,
        window_size: int = ROLLING_WINDOW,
        z_threshold: float = DRIFT_Z_THRESHOLD,
        warmup: int = WARMUP_OBSERVATIONS,
        persistence_path: Optional[Path] = None,
    ):
        self.window_size = window_size
        self.z_threshold = z_threshold
        self.warmup = warmup
        self.persistence_path = persistence_path or _DRIFT_LOG_PATH

        # Rolling windows for each stream
        self._prediction_errors: deque = deque(maxlen=window_size)
        self._agent_disagreements: deque = deque(maxlen=window_size)
        self._micro_anomalies: deque = deque(maxlen=window_size)

        # Baseline stats (computed from first N observations)
        self._baselines: Dict[str, Dict[str, float]] = {
            "prediction_error": {"mean": 0.0, "std": 1.0, "initialized": False},
            "agent_disagreement": {"mean": 0.0, "std": 1.0, "initialized": False},
            "micro_anomaly": {"mean": 0.0, "std": 1.0, "initialized": False},
        }

        # Drift history
        self._drift_events: List[Dict[str, Any]] = []
        self._total_observations: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(
        self,
        prediction_error: float = 0.0,
        agent_disagreement: float = 0.0,
        micro_anomaly_score: float = 0.0,
    ) -> None:
        """Add new observations to the rolling windows.

        Args:
            prediction_error: |predicted_confidence - actual_outcome|
            agent_disagreement: Spread of agent votes (0-1)
            micro_anomaly_score: Microstructure deviation score (0-1)
        """
        self._prediction_errors.append(prediction_error)
        self._agent_disagreements.append(agent_disagreement)
        self._micro_anomalies.append(micro_anomaly_score)
        self._total_observations += 1

        # Initialize baselines after warmup
        if self._total_observations == self.warmup:
            self._initialize_baselines()

    def detect(self) -> DriftResult:
        """Check for concept drift across all streams.

        Returns:
            DriftResult with drift status and severity
        """
        result = DriftResult(observations_count=self._total_observations)

        if self._total_observations < self.warmup:
            result.reason = (
                f"Warmup phase ({self._total_observations}/{self.warmup})"
            )
            return result

        # Compute z-scores for each stream
        drifting = []

        z_pred = self._compute_z_score(
            self._prediction_errors, self._baselines["prediction_error"]
        )
        result.z_scores["prediction_error"] = z_pred
        if abs(z_pred) > self.z_threshold:
            drifting.append("prediction_error")

        z_agent = self._compute_z_score(
            self._agent_disagreements, self._baselines["agent_disagreement"]
        )
        result.z_scores["agent_disagreement"] = z_agent
        if abs(z_agent) > self.z_threshold:
            drifting.append("agent_disagreement")

        z_micro = self._compute_z_score(
            self._micro_anomalies, self._baselines["micro_anomaly"]
        )
        result.z_scores["micro_anomaly"] = z_micro
        if abs(z_micro) > self.z_threshold:
            drifting.append("micro_anomaly")

        # Determine severity
        n_drifting = len(drifting)
        if n_drifting == 0:
            result.severity = "NONE"
            result.drift_detected = False
            result.confidence_penalty = 1.0
            result.reason = "No drift detected"
        elif n_drifting == 1:
            result.severity = "LOW"
            result.drift_detected = True
            result.confidence_penalty = SEVERITY_PENALTIES["LOW"]
            result.drifting_streams = drifting
            result.reason = f"Drift in {drifting[0]} (z={result.z_scores[drifting[0]]:.2f})"
        elif n_drifting == 2:
            result.severity = "MEDIUM"
            result.drift_detected = True
            result.confidence_penalty = SEVERITY_PENALTIES["MEDIUM"]
            result.drifting_streams = drifting
            result.reason = f"Drift in {', '.join(drifting)}"
        else:
            result.severity = "HIGH"
            result.drift_detected = True
            result.confidence_penalty = SEVERITY_PENALTIES["HIGH"]
            result.drifting_streams = drifting
            result.reason = "All streams drifting — severe concept drift"

        # Log drift events
        if result.drift_detected:
            self._record_drift_event(result)
            logger.warning(
                f"ConceptDrift: {result.severity} — {result.reason} "
                f"(penalty={result.confidence_penalty:.2f})"
            )

        return result

    def get_drift_history(self) -> List[Dict[str, Any]]:
        """Return recent drift events."""
        return self._drift_events[-20:]

    # ------------------------------------------------------------------
    # Internal methods
    # ------------------------------------------------------------------

    def _compute_z_score(
        self,
        window: deque,
        baseline: Dict[str, float],
    ) -> float:
        """Compute z-score of recent window mean vs baseline.

        Uses the mean of the last 10 observations compared against
        the baseline distribution.
        """
        if len(window) < 10 or not baseline.get("initialized", False):
            return 0.0

        recent = list(window)[-10:]
        recent_mean = float(np.mean(recent))

        baseline_mean = baseline["mean"]
        baseline_std = baseline["std"]

        if baseline_std < 1e-10:
            return 0.0

        z = (recent_mean - baseline_mean) / baseline_std
        return z

    def _initialize_baselines(self) -> None:
        """Set baselines from the warmup period observations."""
        if len(self._prediction_errors) >= self.warmup:
            arr = np.array(list(self._prediction_errors))
            self._baselines["prediction_error"] = {
                "mean": float(arr.mean()),
                "std": max(float(arr.std()), 0.01),
                "initialized": True,
            }

        if len(self._agent_disagreements) >= self.warmup:
            arr = np.array(list(self._agent_disagreements))
            self._baselines["agent_disagreement"] = {
                "mean": float(arr.mean()),
                "std": max(float(arr.std()), 0.01),
                "initialized": True,
            }

        if len(self._micro_anomalies) >= self.warmup:
            arr = np.array(list(self._micro_anomalies))
            self._baselines["micro_anomaly"] = {
                "mean": float(arr.mean()),
                "std": max(float(arr.std()), 0.01),
                "initialized": True,
            }

        logger.info(
            f"ConceptDrift: Baselines initialized from {self.warmup} observations — "
            f"pred_err={self._baselines['prediction_error']['mean']:.3f}±"
            f"{self._baselines['prediction_error']['std']:.3f}, "
            f"agent_dis={self._baselines['agent_disagreement']['mean']:.3f}±"
            f"{self._baselines['agent_disagreement']['std']:.3f}"
        )

    def _record_drift_event(self, result: DriftResult) -> None:
        """Record drift event for history and persistence."""
        event = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "severity": result.severity,
            "drifting_streams": result.drifting_streams,
            "z_scores": result.z_scores,
            "confidence_penalty": result.confidence_penalty,
            "observation_count": self._total_observations,
        }
        self._drift_events.append(event)

        # Keep only last 100 events in memory
        if len(self._drift_events) > 100:
            self._drift_events = self._drift_events[-100:]

        # Persist
        self._save_drift_log()

    def _save_drift_log(self) -> None:
        """Persist drift history."""
        try:
            data = {
                "version": 1,
                "total_observations": self._total_observations,
                "baselines": self._baselines,
                "recent_events": self._drift_events[-50:],
                "last_updated": datetime.now(timezone.utc).isoformat(),
            }
            self.persistence_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                from src.scanner.automation.safe_json import safe_json_write
                safe_json_write(self.persistence_path, data)
            except ImportError:
                self.persistence_path.write_text(json.dumps(data, indent=2))
        except Exception as e:
            logger.debug(f"ConceptDrift: Failed to save drift log: {e}")

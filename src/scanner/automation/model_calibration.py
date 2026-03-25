"""Model confidence calibration tracker.

US-112: Measure how well each model's confidence scores predict actual
outcomes. Builds calibration curves (Platt scaling inspired) and detects
flocking (all models agree but are systematically wrong).

Approach:
1. Record (model, predicted_confidence, actual_outcome) per prediction
2. Bin predictions into confidence buckets (0.0-0.1, 0.1-0.2, ...)
3. Compute calibration error: |predicted - actual| per bin
4. Detect overconfidence: when predicted > actual systematically
5. Detect flocking: all models agree with high confidence but outcome is wrong
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_LOG_PATH = _PROJECT_ROOT / "trained_data" / "model_calibration_log.json"

# Calibration parameters
N_BINS = 10  # Number of confidence bins (0.0-0.1, ..., 0.9-1.0)
MIN_SAMPLES_PER_BIN = 5  # Minimum samples for a bin to be valid
ROLLING_WINDOW = 100  # Maximum predictions per model

# Flocking detection
FLOCKING_CONFIDENCE_THRESHOLD = 0.65  # All models above this = potential flock
FLOCKING_AGREEMENT_THRESHOLD = 0.90  # Proportion of models that must agree
FLOCKING_WINDOW = 20  # Recent predictions to check for flocking


class ModelCalibrationTracker:
    """Tracks model confidence calibration and detects flocking.

    Bridges model predictions with outcome reality. Reveals when models
    are overconfident, underconfident, or when the ensemble herds into
    systematically wrong high-confidence predictions.

    Usage:
        tracker = ModelCalibrationTracker()
        tracker.record_prediction("TCN", 0.72, won=True)
        tracker.record_prediction("Ridge", 0.68, won=True)
        report = tracker.get_calibration_report("TCN")
        flock = tracker.detect_flocking({"TCN": 0.72, "Ridge": 0.68, "RF": 0.70})
    """

    def __init__(self, persistence_path: Optional[Path] = None):
        self.persistence_path = persistence_path or _LOG_PATH

        # Prediction history: {model: [records]}
        self._predictions: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

        # Flocking history
        self._flock_events: List[Dict[str, Any]] = []
        self._total_predictions: int = 0

        self._load_state()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record_prediction(
        self,
        model: str,
        predicted_confidence: float,
        won: bool,
    ) -> None:
        """Record a prediction and its outcome for calibration tracking.

        Args:
            model: Model name (TCN/Ridge/RF/ensemble).
            predicted_confidence: Model's confidence [0.0, 1.0].
            won: Whether the trade was a winner.
        """
        record = {
            "confidence": float(np.clip(predicted_confidence, 0.0, 1.0)),
            "won": won,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        self._predictions[model].append(record)

        # Trim
        if len(self._predictions[model]) > ROLLING_WINDOW * 2:
            self._predictions[model] = self._predictions[model][-ROLLING_WINDOW:]

        self._total_predictions += 1

    def get_calibration_report(self, model: str) -> Dict[str, Any]:
        """Get calibration report for a specific model.

        Returns:
            Dict with calibration_error, overconfidence_ratio,
            reliability_diagram data, and per-bin statistics.
        """
        predictions = self._predictions.get(model, [])
        if not predictions:
            return {
                "model": model,
                "calibration_error": None,
                "overconfidence_ratio": None,
                "bins": [],
                "n_predictions": 0,
            }

        recent = predictions[-ROLLING_WINDOW:]

        # Bin predictions
        bins = self._compute_bins(recent)

        # Expected Calibration Error (ECE)
        total_samples = sum(b["count"] for b in bins if b["count"] >= MIN_SAMPLES_PER_BIN)
        ece = 0.0
        overconfident_bins = 0
        valid_bins = 0

        for b in bins:
            if b["count"] < MIN_SAMPLES_PER_BIN:
                continue
            valid_bins += 1
            weight = b["count"] / max(total_samples, 1)
            ece += weight * abs(b["avg_confidence"] - b["actual_win_rate"])

            if b["avg_confidence"] > b["actual_win_rate"]:
                overconfident_bins += 1

        overconfidence_ratio = (
            overconfident_bins / valid_bins if valid_bins > 0 else None
        )

        return {
            "model": model,
            "calibration_error": round(ece, 4) if valid_bins > 0 else None,
            "overconfidence_ratio": (
                round(overconfidence_ratio, 4) if overconfidence_ratio is not None else None
            ),
            "bins": bins,
            "n_predictions": len(recent),
            "valid_bins": valid_bins,
        }

    def detect_flocking(
        self,
        model_predictions: Dict[str, float],
    ) -> Dict[str, Any]:
        """Detect flocking — all models agree with high confidence.

        Flocking is dangerous because it bypasses ensemble diversity.
        When all models produce the same high-confidence prediction,
        there's no dissenting signal to catch systematic errors.

        Args:
            model_predictions: {model_name: confidence} for current prediction.

        Returns:
            Dict with flocking_score, is_flocking, details.
        """
        if len(model_predictions) < 2:
            return {"flocking_score": 0.0, "is_flocking": False}

        confidences = list(model_predictions.values())
        high_conf = [c for c in confidences if c >= FLOCKING_CONFIDENCE_THRESHOLD]
        agreement_ratio = len(high_conf) / len(confidences)

        # Check direction agreement (all same sign relative to 0.5)
        # Neutral zone: confidences in [0.45, 0.55] are excluded from
        # direction agreement calculation to avoid false flocking on
        # uncertain predictions.
        directional = [c for c in confidences if c < 0.45 or c > 0.55]
        if not directional:
            # All models are in the neutral zone — no directional agreement
            return {"flocking_score": 0.0, "is_flocking": False}
        directions = [1 if c > 0.5 else -1 for c in directional]
        direction_agreement = abs(sum(directions)) / len(directions)

        # Flocking score: combination of confidence agreement and direction agreement
        flocking_score = agreement_ratio * direction_agreement

        # Average confidence of the flock
        avg_confidence = float(np.mean(confidences))

        is_flocking = (
            agreement_ratio >= FLOCKING_AGREEMENT_THRESHOLD
            and avg_confidence >= FLOCKING_CONFIDENCE_THRESHOLD
        )

        result = {
            "flocking_score": round(flocking_score, 4),
            "is_flocking": is_flocking,
            "agreement_ratio": round(agreement_ratio, 4),
            "avg_confidence": round(avg_confidence, 4),
            "n_models": len(model_predictions),
            "n_high_confidence": len(high_conf),
        }

        if is_flocking:
            # Check historical flocking accuracy
            flock_accuracy = self._get_flock_accuracy()
            result["historical_flock_accuracy"] = flock_accuracy

            self._flock_events.append({
                **result,
                "model_predictions": {
                    k: round(v, 4) for k, v in model_predictions.items()
                },
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })
            if len(self._flock_events) > 100:
                self._flock_events = self._flock_events[-50:]

        return result

    def record_flock_outcome(self, won: bool) -> None:
        """Record whether the most recent flock event was correct."""
        if self._flock_events:
            self._flock_events[-1]["won"] = won

    def get_status(self) -> Dict[str, Any]:
        """Get tracker status for observability."""
        return {
            "total_predictions": self._total_predictions,
            "models": {
                model: len(preds) for model, preds in self._predictions.items()
            },
            "flock_events": len(self._flock_events),
            "recent_flock_events": self._flock_events[-3:],
        }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _compute_bins(
        self, predictions: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Compute calibration bins from predictions."""
        bins = []
        for i in range(N_BINS):
            lo = i / N_BINS
            hi = (i + 1) / N_BINS

            in_bin = [
                p for p in predictions
                if lo <= p["confidence"] < hi or (i == N_BINS - 1 and p["confidence"] == hi)
            ]

            if not in_bin:
                bins.append({
                    "range": f"{lo:.1f}-{hi:.1f}",
                    "count": 0,
                    "avg_confidence": (lo + hi) / 2,
                    "actual_win_rate": 0.0,
                    "calibration_gap": 0.0,
                })
                continue

            avg_conf = float(np.mean([p["confidence"] for p in in_bin]))
            actual_wr = sum(1 for p in in_bin if p["won"]) / len(in_bin)
            gap = avg_conf - actual_wr

            bins.append({
                "range": f"{lo:.1f}-{hi:.1f}",
                "count": len(in_bin),
                "avg_confidence": round(avg_conf, 4),
                "actual_win_rate": round(actual_wr, 4),
                "calibration_gap": round(gap, 4),
            })

        return bins

    def _get_flock_accuracy(self) -> Optional[float]:
        """Get historical accuracy of flocking events."""
        resolved = [e for e in self._flock_events if "won" in e]
        if len(resolved) < 3:
            return None
        wins = sum(1 for e in resolved if e["won"])
        return round(wins / len(resolved), 4)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_state(self) -> None:
        """Persist calibration data."""
        try:
            predictions_plain = {
                model: records[-ROLLING_WINDOW:]
                for model, records in self._predictions.items()
            }

            data = {
                "version": 1,
                "total_predictions": self._total_predictions,
                "predictions": predictions_plain,
                "flock_events": self._flock_events[-50:],
                "last_updated": datetime.now(timezone.utc).isoformat(),
            }
            self.persistence_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                from src.scanner.automation.safe_json import safe_json_write
                safe_json_write(self.persistence_path, data)
            except ImportError:
                self.persistence_path.write_text(json.dumps(data, indent=2))
        except Exception as e:
            logger.warning(f"ModelCalibrationTracker: Failed to save: {e}")

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
                self._total_predictions = data.get("total_predictions", 0)
                self._flock_events = data.get("flock_events", [])
                raw_preds = data.get("predictions", {})
                for model, records in raw_preds.items():
                    if not isinstance(records, list):
                        continue
                    valid = [
                        r for r in records
                        if isinstance(r, dict)
                        and "confidence" in r and "won" in r
                    ]
                    if len(valid) < len(records):
                        logger.warning(
                            f"ModelCalibrationTracker: Skipped {len(records) - len(valid)} "
                            f"malformed records for model {model}"
                        )
                    self._predictions[model] = valid
        except Exception as e:
            logger.warning(f"ModelCalibrationTracker: Failed to load: {e}")

"""Forward drift projection for predictive self-modeling.

US-Tier6-PSM: Instead of detecting drift after it happens, project
future accuracy degradation based on:
1. Historical drift velocity (how fast accuracy decayed in past episodes)
2. Current feature distribution shift (KL divergence from training baseline)
3. Regime transition patterns (certain regime changes precede drift by ~12-24h)
4. Model age curves (performance degrades on a characteristic curve post-training)

Output: a ProjectionCurve with confidence accuracy estimates at t+6h, t+12h, t+24h, t+48h

Design principles:
- Pure statistical model, no ML dependencies (numpy only)
- Graceful degradation: if insufficient history, return conservative projection
- Non-blocking: never delays a scan cycle
- Persists projections to trained_data/drift_projections.json for dashboard use
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_DRIFT_LOG_PATH = _PROJECT_ROOT / "trained_data" / "drift_monitor_log.json"
_PROJECTION_CACHE_PATH = _PROJECT_ROOT / "trained_data" / "drift_projections.json"
_FEATURE_BASELINE_PATH = _PROJECT_ROOT / "trained_data" / "feature_baseline.json"
_REGIME_PATH = _PROJECT_ROOT / "trained_data" / "regime_transitions.json"


@dataclass
class DriftProjectionPoint:
    """A single point on the projection curve."""

    hours_ahead: int
    predicted_accuracy: float  # 0.0 - 1.0
    confidence_interval_low: float
    confidence_interval_high: float
    degradation_driver: str  # "regime_shift", "feature_drift", "model_age", "historical_pattern", "stable"

    def to_dict(self) -> dict:
        """Convert to dict for JSON serialization."""
        return asdict(self)


@dataclass
class DriftProjectionCurve:
    """Complete projection curve for a single pair."""

    pair: str
    generated_at: str  # ISO timestamp
    current_accuracy: float
    baseline_accuracy: float  # training baseline
    projection_points: List[DriftProjectionPoint] = field(default_factory=list)
    recommended_action: str = "monitor"  # "monitor", "prepare_retrain", "reduce_sizing", "immediate_retrain"
    action_urgency: str = "low"  # "low", "medium", "high", "critical"
    risk_factors: List[str] = field(default_factory=list)  # what's driving the projection

    def hours_until_critical(self, critical_threshold: float = 0.48) -> Optional[int]:
        """Return hours until projected accuracy falls below critical_threshold.

        Returns:
            Hours ahead, or None if accuracy never reaches critical threshold.
        """
        for point in self.projection_points:
            if point.predicted_accuracy < critical_threshold:
                return point.hours_ahead
        return None

    def to_dict(self) -> dict:
        """Convert to dict for JSON serialization."""
        return {
            "pair": self.pair,
            "generated_at": self.generated_at,
            "current_accuracy": round(self.current_accuracy, 4),
            "baseline_accuracy": round(self.baseline_accuracy, 4),
            "projection_points": [p.to_dict() for p in self.projection_points],
            "recommended_action": self.recommended_action,
            "action_urgency": self.action_urgency,
            "risk_factors": self.risk_factors,
            "hours_until_critical": self.hours_until_critical(),
        }


class DriftProjector:
    """Projects future model accuracy using historical drift patterns.

    Algorithm:
    1. Load drift history for the pair from drift_monitor_log.json
    2. Fit a simple degradation velocity model (linear regression on recent accuracy trend)
    3. Detect regime transition signals (cross-reference regime_tracker state)
    4. Compute KL divergence of recent feature distribution vs baseline
    5. Blend into a weighted projection curve
    6. Map projected accuracy to recommended action
    """

    CRITICAL_ACCURACY = 0.48  # Below this = force sizing reduction
    WARNING_ACCURACY = 0.52  # Below this = prepare for retrain
    MIN_HISTORY_POINTS = 5  # Minimum drift records before projecting

    # Projection time horizons (hours)
    PROJECTION_HORIZONS = [6, 12, 24, 48]

    # Weights for blending signals (velocity, regime_risk, feature_drift)
    # Sum to 1.0: how much each signal contributes to final projection
    VELOCITY_WEIGHT = 0.50
    REGIME_WEIGHT = 0.25
    FEATURE_WEIGHT = 0.15
    MODEL_AGE_WEIGHT = 0.10

    def __init__(
        self,
        drift_log_path: Optional[Path] = None,
        projection_cache_path: Optional[Path] = None,
        feature_baseline_path: Optional[Path] = None,
        regime_path: Optional[Path] = None,
    ):
        """Initialize DriftProjector.

        Args:
            drift_log_path: Path to drift_monitor_log.json
            projection_cache_path: Path to persist drift_projections.json
            feature_baseline_path: Path to feature_baseline.json
            regime_path: Path to regime_transitions.json
        """
        self.drift_log_path = drift_log_path or _DRIFT_LOG_PATH
        self.projection_cache_path = projection_cache_path or _PROJECTION_CACHE_PATH
        self.feature_baseline_path = feature_baseline_path or _FEATURE_BASELINE_PATH
        self.regime_path = regime_path or _REGIME_PATH

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def project(
        self,
        pair: str,
        current_accuracy: float = 0.55,
        baseline_accuracy: float = 0.55,
        current_features: Optional[np.ndarray] = None,
    ) -> DriftProjectionCurve:
        """Generate a forward drift projection for the given pair.

        Args:
            pair: Currency pair (e.g., "EUR_USD")
            current_accuracy: Current measured accuracy (0.0-1.0)
            baseline_accuracy: Training baseline accuracy
            current_features: Optional current feature vector for KL divergence

        Returns:
            DriftProjectionCurve with projections for t+6h, t+12h, t+24h, t+48h
        """
        # Load drift history
        history = self._load_drift_history(pair)

        # Estimate velocity (accuracy per hour)
        velocity = self._estimate_velocity(history, current_accuracy)

        # Detect regime risk
        regime_risk = self._detect_regime_risk_factor(pair)

        # Compute feature drift
        feature_drift = self._compute_feature_drift_score(current_features)

        # Estimate model age penalty
        model_age_penalty = self._estimate_model_age_penalty()

        # Build projection curve
        curve = self._build_projection_curve(
            pair=pair,
            current_accuracy=current_accuracy,
            baseline_accuracy=baseline_accuracy,
            velocity=velocity,
            regime_risk=regime_risk,
            feature_drift=feature_drift,
            model_age_penalty=model_age_penalty,
        )

        # Map to action
        recommended_action, urgency = self._map_to_action(curve)
        curve.recommended_action = recommended_action
        curve.action_urgency = urgency

        return curve

    def project_all_pairs(
        self,
        pairs: List[str],
        current_accuracies: Optional[Dict[str, float]] = None,
    ) -> Dict[str, DriftProjectionCurve]:
        """Project all active pairs and return dict keyed by pair.

        Args:
            pairs: List of currency pairs
            current_accuracies: Optional dict of pair -> accuracy

        Returns:
            Dict[pair, DriftProjectionCurve]
        """
        if current_accuracies is None:
            current_accuracies = {}

        projections = {}
        for pair in pairs:
            try:
                accuracy = current_accuracies.get(pair, 0.55)
                curve = self.project(pair, current_accuracy=accuracy)
                projections[pair] = curve
            except Exception as e:
                logger.warning(f"DriftProjector: Failed to project {pair}: {e}")
                # Return conservative projection on error
                projections[pair] = DriftProjectionCurve(
                    pair=pair,
                    generated_at=datetime.now(timezone.utc).isoformat(),
                    current_accuracy=accuracy,
                    baseline_accuracy=0.55,
                    recommended_action="monitor",
                    action_urgency="low",
                    risk_factors=["projection_error"],
                )

        return projections

    # ------------------------------------------------------------------
    # Internal methods
    # ------------------------------------------------------------------

    def _load_drift_history(self, pair: str) -> List[Dict[str, Any]]:
        """Load historical drift records for a pair.

        Returns:
            List of drift records, sorted by timestamp. Each record has:
            - timestamp (ISO string)
            - drift_score (float)
            - current_accuracy (float, optional)
            - severity (str, optional)
        """
        if not self.drift_log_path.exists():
            return []

        try:
            try:
                from src.scanner.automation.safe_json import safe_json_read

                data = safe_json_read(self.drift_log_path, default={})
            except ImportError:
                data = json.loads(self.drift_log_path.read_text(encoding="utf-8"))

            if not isinstance(data, dict):
                return []

            drift_history = data.get("drift_history", {})
            if not isinstance(drift_history, dict):
                return []

            pair_records = drift_history.get(pair, [])
            if not isinstance(pair_records, list):
                return []

            # Sort by timestamp (oldest first)
            try:
                pair_records = sorted(
                    pair_records,
                    key=lambda r: r.get("timestamp", ""),
                )
            except Exception:
                pass

            return pair_records

        except Exception as e:
            logger.debug(f"DriftProjector: Failed to load drift history for {pair}: {e}")
            return []

    def _estimate_velocity(
        self,
        history: List[Dict[str, Any]],
        current_accuracy: float,
    ) -> float:
        """Estimate accuracy degradation velocity (accuracy/hour).

        Uses linear regression on recent drift records to estimate the
        rate at which accuracy is changing.

        Args:
            history: List of historical drift records
            current_accuracy: Current accuracy (last point)

        Returns:
            Velocity in accuracy points/hour. Negative = degrading. 0 if insufficient data.
        """
        if len(history) < self.MIN_HISTORY_POINTS:
            return 0.0

        try:
            # Extract timestamps and accuracies
            points = []
            for record in history[-10:]:  # Use last 10 records
                if "timestamp" not in record:
                    continue
                try:
                    ts = datetime.fromisoformat(record["timestamp"].replace("Z", "+00:00"))
                except (ValueError, AttributeError):
                    continue

                # Try to extract accuracy
                accuracy = record.get("current_accuracy")
                if accuracy is None:
                    # Infer from drift_score if available
                    drift = record.get("drift_score", 0.0)
                    accuracy = 0.55 - abs(drift)  # Rough estimate

                points.append((ts, accuracy))

            if len(points) < self.MIN_HISTORY_POINTS:
                return 0.0

            # Sort by timestamp
            points.sort(key=lambda p: p[0])

            # Add current point
            now = datetime.now(timezone.utc)
            points.append((now, current_accuracy))

            # Fit line: hours -> accuracy
            hours_since_start = np.array(
                [(p[0] - points[0][0]).total_seconds() / 3600.0 for p in points],
                dtype=np.float64,
            )
            accuracies = np.array([p[1] for p in points], dtype=np.float64)

            # Linear regression: y = mx + b
            # Slope m is velocity (accuracy/hour)
            if np.std(hours_since_start) < 1e-6:
                return 0.0

            coeffs = np.polyfit(hours_since_start, accuracies, 1)
            velocity = float(coeffs[0])  # Slope

            return velocity

        except Exception as e:
            logger.debug(f"DriftProjector: Velocity estimation failed: {e}")
            return 0.0

    def _detect_regime_risk_factor(self, pair: str) -> float:
        """Return 0.0-1.0 risk score based on recent regime transitions.

        Checks if pair is in a high-risk regime or approaching one.

        Returns:
            Risk score 0.0-1.0. Higher = more risk of drift.
        """
        if not self.regime_path.exists():
            return 0.0

        try:
            try:
                from src.scanner.automation.safe_json import safe_json_read

                data = safe_json_read(self.regime_path, default={})
            except ImportError:
                data = json.loads(self.regime_path.read_text(encoding="utf-8"))

            if not isinstance(data, dict):
                return 0.0

            # Look for pair-specific regime info or use global
            regime_info = data.get(pair, data.get("_global", {}))
            if not isinstance(regime_info, dict):
                return 0.0

            # Extract current regime
            current_regime = regime_info.get("current_regime", "NORMAL")

            # Map regime to risk
            regime_risk_map = {
                "LOW": 0.05,
                "NORMAL": 0.10,
                "HIGH": 0.40,
                "EXTREME": 0.80,
            }

            base_risk = regime_risk_map.get(current_regime, 0.15)

            # Check for recent transitions (transition_count)
            recent_transitions = regime_info.get("recent_transitions", 0)
            transition_risk = min(0.3, recent_transitions * 0.15)

            return min(1.0, base_risk + transition_risk)

        except Exception as e:
            logger.debug(f"DriftProjector: Regime risk detection failed: {e}")
            return 0.0

    def _compute_feature_drift_score(
        self,
        current_features: Optional[np.ndarray],
    ) -> float:
        """Return 0.0-1.0 KL-divergence-based feature drift score.

        Compares current feature distribution to baseline.

        Args:
            current_features: Optional current feature vector

        Returns:
            Drift score 0.0-1.0. 0 if features unavailable.
        """
        if current_features is None:
            return 0.0

        if not self.feature_baseline_path.exists():
            return 0.0

        try:
            try:
                from src.scanner.automation.safe_json import safe_json_read

                baseline_data = safe_json_read(self.feature_baseline_path, default={})
            except ImportError:
                baseline_data = json.loads(
                    self.feature_baseline_path.read_text(encoding="utf-8")
                )

            if not isinstance(baseline_data, dict):
                return 0.0

            # Extract baseline mean and std
            baseline_mean = baseline_data.get("mean")
            baseline_std = baseline_data.get("std")

            if baseline_mean is None or baseline_std is None:
                return 0.0

            baseline_mean = np.array(baseline_mean, dtype=np.float64)
            baseline_std = np.array(baseline_std, dtype=np.float64)

            if current_features.shape != baseline_mean.shape:
                return 0.0

            # Compute normalized KL divergence
            # Simple metric: sum of squared z-scores / feature count
            z_scores = np.abs((current_features - baseline_mean) / (baseline_std + 1e-8))
            kl_approx = np.mean(z_scores ** 2)

            # Normalize to 0-1 (assume large z_score divergence is 2.0)
            drift_score = min(1.0, kl_approx / 4.0)

            return float(drift_score)

        except Exception as e:
            logger.debug(f"DriftProjector: Feature drift computation failed: {e}")
            return 0.0

    def _estimate_model_age_penalty(self) -> float:
        """Estimate accuracy penalty based on model age.

        Models degrade over time even without obvious drift.
        Returns penalty multiplier (0-1): how much to reduce accuracy per hour.

        Returns:
            Hourly degradation rate. Default 0.0001 (0.01% per hour).
        """
        # Default: models degrade ~0.01% per hour naturally
        # This is a conservative estimate; can be tuned based on empirical data
        return 0.0001

    def _build_projection_curve(
        self,
        pair: str,
        current_accuracy: float,
        baseline_accuracy: float,
        velocity: float,
        regime_risk: float,
        feature_drift: float,
        model_age_penalty: float,
    ) -> DriftProjectionCurve:
        """Combine signals into projection curve with confidence intervals.

        Args:
            pair: Currency pair
            current_accuracy: Current accuracy
            baseline_accuracy: Training baseline
            velocity: Velocity (accuracy/hour)
            regime_risk: Regime risk score (0-1)
            feature_drift: Feature drift score (0-1)
            model_age_penalty: Model age penalty (accuracy/hour)

        Returns:
            DriftProjectionCurve
        """
        # Blend velocity signal with other risk factors
        # Higher regime_risk and feature_drift increase the severity of velocity
        risk_blend = (
            self.VELOCITY_WEIGHT * velocity
            + self.REGIME_WEIGHT * (regime_risk * -0.05)  # regime risk increases negative velocity
            + self.FEATURE_WEIGHT * (feature_drift * -0.02)  # feature drift slightly increases negative velocity
            + self.MODEL_AGE_WEIGHT * (-model_age_penalty)
        )

        # Clamp to reasonable range
        combined_velocity = np.clip(risk_blend, -0.10, 0.05)  # Max -10% per hour, max +5% per hour

        # Generate projection points
        projection_points = []
        risk_factors = []

        for hours_ahead in self.PROJECTION_HORIZONS:
            # Project accuracy at this horizon
            projected_accuracy = current_accuracy + (combined_velocity * hours_ahead)
            projected_accuracy = np.clip(projected_accuracy, 0.0, 1.0)

            # Confidence interval: narrower for near-term, wider for far-term
            # Use historical volatility as guide; default to ±0.05
            ci_width = 0.05 + (hours_ahead / 48.0) * 0.10  # Grows with horizon
            ci_low = max(0.0, projected_accuracy - ci_width)
            ci_high = min(1.0, projected_accuracy + ci_width)

            # Determine degradation driver
            drivers = []
            if abs(velocity) > 0.005:  # >0.5% per hour
                drivers.append("historical_pattern")
            if regime_risk > 0.3:
                drivers.append("regime_shift")
            if feature_drift > 0.3:
                drivers.append("feature_drift")
            if abs(combined_velocity) > 0.0:
                drivers.append("model_age")

            driver = drivers[0] if drivers else "stable"

            point = DriftProjectionPoint(
                hours_ahead=hours_ahead,
                predicted_accuracy=float(projected_accuracy),
                confidence_interval_low=float(ci_low),
                confidence_interval_high=float(ci_high),
                degradation_driver=driver,
            )
            projection_points.append(point)

        # Collect risk factors for reporting
        if velocity < -0.005:
            risk_factors.append(f"negative_velocity_{velocity:.4f}_per_hour")
        if regime_risk > 0.3:
            risk_factors.append(f"regime_risk_{regime_risk:.2f}")
        if feature_drift > 0.3:
            risk_factors.append(f"feature_drift_{feature_drift:.2f}")

        curve = DriftProjectionCurve(
            pair=pair,
            generated_at=datetime.now(timezone.utc).isoformat(),
            current_accuracy=current_accuracy,
            baseline_accuracy=baseline_accuracy,
            projection_points=projection_points,
            risk_factors=risk_factors if risk_factors else ["stable"],
        )

        return curve

    def _map_to_action(
        self,
        curve: DriftProjectionCurve,
    ) -> Tuple[str, str]:
        """Map projection to (recommended_action, urgency).

        Args:
            curve: DriftProjectionCurve

        Returns:
            Tuple of (action, urgency) where action is one of:
            - "monitor": stable, no action needed
            - "prepare_retrain": mild degradation, start planning retrain
            - "reduce_sizing": moderate degradation, reduce position sizes
            - "immediate_retrain": severe degradation, retrain immediately
        """
        hours_until_critical = curve.hours_until_critical(self.CRITICAL_ACCURACY)
        hours_until_warning = curve.hours_until_critical(self.WARNING_ACCURACY)

        # Current accuracy assessment
        if curve.current_accuracy < self.CRITICAL_ACCURACY:
            return "immediate_retrain", "critical"

        if curve.current_accuracy < self.WARNING_ACCURACY:
            return "reduce_sizing", "high"

        # Projected degradation assessment
        if hours_until_critical is not None:
            if hours_until_critical <= 6:
                return "immediate_retrain", "critical"
            elif hours_until_critical <= 12:
                return "reduce_sizing", "high"
            elif hours_until_critical <= 24:
                return "reduce_sizing", "medium"
            elif hours_until_critical <= 48:
                return "prepare_retrain", "medium"
            else:
                return "prepare_retrain", "low"

        if hours_until_warning is not None:
            if hours_until_warning <= 12:
                return "prepare_retrain", "medium"
            else:
                return "prepare_retrain", "low"

        # Stable projection
        return "monitor", "low"

    def save_projections(self, projections: Dict[str, DriftProjectionCurve]) -> None:
        """Persist projection cache atomically.

        Args:
            projections: Dict[pair, DriftProjectionCurve]
        """
        try:
            data = {
                "version": 1,
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "projections": {
                    pair: curve.to_dict() for pair, curve in projections.items()
                },
            }

            try:
                from src.scanner.automation.safe_json import safe_json_write

                safe_json_write(self.projection_cache_path, data)
            except ImportError:
                self.projection_cache_path.parent.mkdir(parents=True, exist_ok=True)
                tmp_path = self.projection_cache_path.with_suffix(".json.tmp")
                tmp_path.write_text(json.dumps(data, indent=2, sort_keys=True))
                os.rename(str(tmp_path), str(self.projection_cache_path))

            logger.debug(
                f"DriftProjector: Saved {len(projections)} projections to {self.projection_cache_path}"
            )

        except Exception as e:
            logger.warning(f"DriftProjector: Failed to save projections: {e}")

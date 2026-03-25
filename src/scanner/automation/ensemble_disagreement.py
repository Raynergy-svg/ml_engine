"""Ensemble disagreement as a first-class meta-signal.

US-085: Use TCN/Ridge/RF ensemble disagreement beyond simple uncertainty.
Compute prediction variance, spread, and entropy across members. High
disagreement = reduce position size. Rising disagreement = regime
transition warning. Based on 2025 GDE research: ensemble disagreement
equals generalization error.

Metrics:
1. prediction_variance — variance of member confidences
2. prediction_spread — (max - min) / mean of confidences
3. directional_agreement — do all members agree on direction?
4. rolling_trend — is disagreement rising (regime shift signal)?

Position multiplier: 1.0 (consensus) → 0.5 (high disagreement)
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
_DISAGREEMENT_PATH = _PROJECT_ROOT / "trained_data" / "ensemble_disagreement.json"

ROLLING_WINDOW = 50


@dataclass
class DisagreementResult:
    """Result of ensemble disagreement analysis."""

    prediction_variance: float = 0.0
    prediction_spread: float = 0.0
    directional_agreement: bool = True
    confidence_multiplier: float = 1.0
    is_regime_shift_signal: bool = False
    disagreement_trend: str = "stable"  # rising, falling, stable
    member_confidences: Dict[str, float] = field(default_factory=dict)
    reason: str = ""

    def to_dict(self) -> dict:
        return {
            "prediction_variance": round(self.prediction_variance, 6),
            "prediction_spread": round(self.prediction_spread, 4),
            "directional_agreement": self.directional_agreement,
            "confidence_multiplier": round(self.confidence_multiplier, 4),
            "is_regime_shift_signal": self.is_regime_shift_signal,
            "disagreement_trend": self.disagreement_trend,
            "member_confidences": {
                k: round(v, 4) for k, v in self.member_confidences.items()
            },
            "reason": self.reason,
        }


class EnsembleDisagreementSignal:
    """Extract trading signals from ensemble member disagreement.

    Tracks rolling disagreement history to detect regime transitions
    (rising disagreement) and compute position size multipliers
    (high disagreement = smaller positions).
    """

    def __init__(
        self,
        window_size: int = ROLLING_WINDOW,
        high_disagreement_threshold: float = 0.15,
        regime_shift_z_threshold: float = 1.5,
        persistence_path: Optional[Path] = None,
    ):
        self.window_size = window_size
        self.high_threshold = high_disagreement_threshold
        self.regime_shift_z = regime_shift_z_threshold
        self.persistence_path = persistence_path or _DISAGREEMENT_PATH

        # Rolling history
        self._variance_history: deque = deque(maxlen=window_size)
        self._spread_history: deque = deque(maxlen=window_size)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute_disagreement(
        self,
        tcn_confidence: float,
        ridge_confidence: float,
        rf_confidence: float = 0.0,
        tcn_direction: str = "",
        ridge_direction: str = "",
        rf_direction: str = "",
    ) -> DisagreementResult:
        """Compute disagreement metrics from ensemble member outputs.

        Args:
            tcn_confidence: TCN model confidence (0-100 scale)
            ridge_confidence: Ridge model confidence (0-100 scale)
            rf_confidence: Random forest confidence (0-100 scale, 0 if unavailable)
            tcn_direction: TCN predicted direction
            ridge_direction: Ridge predicted direction
            rf_direction: RF predicted direction

        Returns:
            DisagreementResult with metrics and position multiplier
        """
        result = DisagreementResult()

        # Normalize confidences to 0-1 scale
        members = {}
        confs = []
        dirs = []

        if tcn_confidence > 0:
            norm_tcn = min(tcn_confidence / 100.0, 1.0) if tcn_confidence > 1 else tcn_confidence
            members["tcn"] = norm_tcn
            confs.append(norm_tcn)
            if tcn_direction:
                dirs.append(tcn_direction.upper())

        if ridge_confidence > 0:
            norm_ridge = min(ridge_confidence / 100.0, 1.0) if ridge_confidence > 1 else ridge_confidence
            members["ridge"] = norm_ridge
            confs.append(norm_ridge)
            if ridge_direction:
                dirs.append(ridge_direction.upper())

        if rf_confidence > 0:
            norm_rf = min(rf_confidence / 100.0, 1.0) if rf_confidence > 1 else rf_confidence
            members["rf"] = norm_rf
            confs.append(norm_rf)
            if rf_direction:
                dirs.append(rf_direction.upper())

        result.member_confidences = members

        if len(confs) < 2:
            result.reason = "Need at least 2 ensemble members"
            return result

        confs_arr = np.array(confs)

        # Metric 1: Prediction variance
        result.prediction_variance = float(np.var(confs_arr))

        # Metric 2: Prediction spread (max - min) / mean
        mean_conf = float(np.mean(confs_arr))
        spread = float(np.max(confs_arr) - np.min(confs_arr))
        result.prediction_spread = spread / max(mean_conf, 0.01)

        # Metric 3: Directional agreement
        if dirs:
            unique_dirs = set(d for d in dirs if d in ("LONG", "SHORT"))
            result.directional_agreement = len(unique_dirs) <= 1

        # Update rolling history
        self._variance_history.append(result.prediction_variance)
        self._spread_history.append(result.prediction_spread)

        # Metric 4: Detect rising disagreement (regime shift signal)
        result.disagreement_trend, result.is_regime_shift_signal = (
            self._detect_trend()
        )

        # Compute position multiplier
        result.confidence_multiplier = self._compute_multiplier(
            result.prediction_variance,
            result.prediction_spread,
            result.directional_agreement,
        )

        result.reason = (
            f"var={result.prediction_variance:.4f}, "
            f"spread={result.prediction_spread:.2f}, "
            f"agree={result.directional_agreement}, "
            f"trend={result.disagreement_trend}"
        )

        if result.is_regime_shift_signal:
            logger.warning(
                f"EnsembleDisagreement: Regime shift signal — "
                f"disagreement {result.disagreement_trend} "
                f"(mult={result.confidence_multiplier:.2f})"
            )

        return result

    def get_position_multiplier(self) -> float:
        """Get the latest position size multiplier based on disagreement."""
        if not self._variance_history:
            return 1.0
        latest_var = self._variance_history[-1]
        if latest_var > self.high_threshold:
            return 0.5
        elif latest_var > self.high_threshold * 0.5:
            return 0.75
        return 1.0

    # ------------------------------------------------------------------
    # Internal methods
    # ------------------------------------------------------------------

    def _compute_multiplier(
        self,
        variance: float,
        spread: float,
        directional_agreement: bool,
    ) -> float:
        """Compute confidence multiplier from disagreement metrics.

        Returns:
            Multiplier 0.5-1.0
        """
        mult = 1.0

        # Variance penalty
        if variance > self.high_threshold:
            mult *= 0.6
        elif variance > self.high_threshold * 0.5:
            mult *= 0.8

        # Spread penalty
        if spread > 0.5:
            mult *= 0.7
        elif spread > 0.3:
            mult *= 0.85

        # Directional disagreement penalty
        if not directional_agreement:
            mult *= 0.7

        return max(0.5, min(1.0, mult))

    def _detect_trend(self) -> Tuple[str, bool]:
        """Detect if disagreement is rising (regime shift signal).

        Returns:
            (trend_direction, is_regime_shift)
        """
        if len(self._variance_history) < 10:
            return "stable", False

        recent = list(self._variance_history)
        recent_mean = float(np.mean(recent[-10:]))
        historical_mean = float(np.mean(recent[:-10])) if len(recent) > 10 else recent_mean
        historical_std = float(np.std(recent[:-10])) if len(recent) > 10 else 0.01

        if historical_std < 1e-10:
            historical_std = 0.01

        z_score = (recent_mean - historical_mean) / historical_std

        if z_score > self.regime_shift_z:
            return "rising", True
        elif z_score < -self.regime_shift_z:
            return "falling", False
        return "stable", False

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_snapshot(self) -> None:
        """Persist disagreement history."""
        try:
            data = {
                "version": 1,
                "variance_history": list(self._variance_history)[-50:],
                "spread_history": list(self._spread_history)[-50:],
                "last_updated": datetime.now(timezone.utc).isoformat(),
            }
            self.persistence_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                from src.scanner.automation.safe_json import safe_json_write
                safe_json_write(self.persistence_path, data)
            except ImportError:
                self.persistence_path.write_text(json.dumps(data, indent=2))
        except Exception as e:
            logger.debug(f"EnsembleDisagreement: Failed to save: {e}")

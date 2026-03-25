"""Override Prediction Model v1 — logistic regression on historical overrides.

PRD v2.2 §13 Phase 3: "Override prediction model v1 (logistic regression
on historical overrides)"

Predicts the probability that an override will result in a loss, given:
- Buddy's confidence at the time of override
- Weighted vote score at the time
- Emotional state (encoded)
- Cognitive load (encoded)
- Override type (one-hot)
- Market regime (one-hot)

Zero-dependency implementation using gradient descent on logistic regression.
No sklearn/numpy required — pure Python with math stdlib.

Usage:
    predictor = OverridePredictor()
    predictor.fit(override_events)  # Train on historical data
    risk = predictor.predict_loss_probability(current_context)
    # risk = 0.78 → "This override has a 78% chance of losing"
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Feature encoding maps
EMOTIONAL_STATE_RISK = {
    "neutral": 0.0,
    "calm": -0.1,
    "energized": -0.05,
    "focused": -0.1,
    "anxious": 0.4,
    "stressed": 0.5,
    "frustrated": 0.6,
    "angry": 0.7,
    "fearful": 0.5,
    "panic": 0.8,
    "revenge": 0.9,
    "impulsive": 0.7,
    "overwhelmed": 0.6,
    "desperate": 0.8,
    "fatigued": 0.3,
    "euphoric": 0.4,  # Overconfidence risk
}

COGNITIVE_LOAD_RISK = {
    "low": -0.1,
    "normal": 0.0,
    "moderate": 0.1,
    "high": 0.4,
    "overloaded": 0.6,
    "exhausted": 0.7,
    "overwhelmed": 0.7,
    "saturated": 0.5,
}

OVERRIDE_TYPE_RISK = {
    "took_rejected": 0.3,      # Highest risk — Buddy said no
    "skipped_recommended": 0.1, # Moderate — missed opportunity, not loss
    "closed_early": 0.2,       # Sometimes smart (cut losses)
    "modified_sl_tp": 0.15,    # Depends on direction
}

REGIME_RISK = {
    "LOW": -0.1,
    "NORMAL": 0.0,
    "HIGH": 0.2,
    "VOLATILE": 0.4,
    "EXTREME": 0.6,
}


@dataclass
class OverridePrediction:
    """Prediction result for an override event."""

    loss_probability: float    # 0.0 to 1.0
    risk_level: str            # "low", "moderate", "high", "critical"
    top_risk_factors: List[str]  # Ranked list of contributing factors
    recommendation: str        # Action recommendation
    feature_contributions: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "loss_probability": round(self.loss_probability, 3),
            "risk_level": self.risk_level,
            "top_risk_factors": self.top_risk_factors,
            "recommendation": self.recommendation,
            "feature_contributions": {
                k: round(v, 3) for k, v in self.feature_contributions.items()
            },
        }


class OverridePredictor:
    """Logistic regression model for override outcome prediction.

    Uses gradient descent to learn weights from historical override events.
    Zero-dependency — no sklearn, numpy, or scipy required.

    Features (8 dimensions):
        0: buddy_confidence (0-1)
        1: weighted_vote (0-1)
        2: emotional_state_risk (encoded -0.1 to 0.9)
        3: cognitive_load_risk (encoded -0.1 to 0.7)
        4: override_type_risk (encoded 0.0 to 0.3)
        5: regime_risk (encoded -0.1 to 0.6)
        6: confidence_vote_interaction (conf * vote, captures consensus strength)
        7: emotional_cognitive_interaction (emotional * cognitive, captures compounded impairment)

    Target: 1.0 = loss, 0.0 = win
    """

    N_FEATURES = 8
    FEATURE_NAMES = [
        "buddy_confidence",
        "weighted_vote",
        "emotional_state_risk",
        "cognitive_load_risk",
        "override_type_risk",
        "regime_risk",
        "confidence_vote_interaction",
        "emotional_cognitive_interaction",
    ]

    def __init__(
        self,
        model_path: Optional[Path] = None,
        learning_rate: float = 0.05,
        n_epochs: int = 200,
        regularization: float = 0.01,
    ):
        self.model_path = model_path or Path(".aura/models/override_predictor.json")
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.regularization = regularization  # L2 regularization

        # Model parameters (weights + bias)
        self._weights: List[float] = [0.0] * self.N_FEATURES
        self._bias: float = 0.0
        self._trained: bool = False
        self._train_samples: int = 0
        self._train_accuracy: float = 0.0
        self._feature_means: List[float] = [0.0] * self.N_FEATURES
        self._feature_stds: List[float] = [1.0] * self.N_FEATURES

        # Try to load saved model
        self._load_model()

    # --- Feature Engineering ---

    def _encode_features(self, event: Dict[str, Any]) -> List[float]:
        """Encode an override event into a feature vector.

        Args:
            event: OverrideEvent.to_dict() payload

        Returns:
            List of 8 float features
        """
        confidence = float(event.get("confidence_at_time", 0.5))
        weighted_vote = float(event.get("weighted_vote_at_time", 0.5))

        emotional = (event.get("emotional_state", "") or "").lower()
        emotional_risk = EMOTIONAL_STATE_RISK.get(emotional, 0.0)

        cognitive = (event.get("cognitive_load", "") or "").lower()
        cognitive_risk = COGNITIVE_LOAD_RISK.get(cognitive, 0.0)

        override_type = event.get("override_type", "")
        type_risk = OVERRIDE_TYPE_RISK.get(override_type, 0.15)

        regime = (event.get("regime", "") or "NORMAL").upper()
        regime_risk = REGIME_RISK.get(regime, 0.0)

        # Interaction features
        conf_vote = confidence * weighted_vote
        emotional_cognitive = emotional_risk * cognitive_risk

        return [
            confidence,
            weighted_vote,
            emotional_risk,
            cognitive_risk,
            type_risk,
            regime_risk,
            conf_vote,
            emotional_cognitive,
        ]

    def _encode_target(self, event: Dict[str, Any]) -> Optional[float]:
        """Encode the outcome as a binary target.

        Returns:
            1.0 for loss, 0.0 for win, None if no outcome.
        """
        outcome = event.get("outcome")
        if outcome == "loss":
            return 1.0
        elif outcome == "win":
            return 0.0
        return None

    # --- Logistic Regression Core ---

    @staticmethod
    def _sigmoid(z: float) -> float:
        """Numerically stable sigmoid function."""
        if z >= 0:
            return 1.0 / (1.0 + math.exp(-z))
        else:
            exp_z = math.exp(z)
            return exp_z / (1.0 + exp_z)

    def _predict_raw(self, features: List[float]) -> float:
        """Compute raw prediction (sigmoid of linear combination)."""
        z = self._bias
        for w, x in zip(self._weights, features):
            z += w * x
        return self._sigmoid(z)

    def _normalize_features(
        self, X: List[List[float]]
    ) -> List[List[float]]:
        """Z-score normalize features using stored means/stds."""
        normalized = []
        for sample in X:
            row = []
            for i, val in enumerate(sample):
                std = self._feature_stds[i] if self._feature_stds[i] > 1e-8 else 1.0
                row.append((val - self._feature_means[i]) / std)
            normalized.append(row)
        return normalized

    def _compute_means_stds(self, X: List[List[float]]) -> None:
        """Compute feature means and standard deviations for normalization."""
        n = len(X)
        if n == 0:
            return

        means = [0.0] * self.N_FEATURES
        for sample in X:
            for i, val in enumerate(sample):
                means[i] += val
        self._feature_means = [m / n for m in means]

        stds = [0.0] * self.N_FEATURES
        for sample in X:
            for i, val in enumerate(sample):
                stds[i] += (val - self._feature_means[i]) ** 2
        self._feature_stds = [math.sqrt(s / n) if s > 0 else 1.0 for s in stds]

    # --- Training ---

    def fit(self, events: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Train the logistic regression model on historical override events.

        Uses mini-batch gradient descent with L2 regularization.

        Args:
            events: List of OverrideEvent.to_dict() payloads with outcomes.

        Returns:
            Training metrics dict.
        """
        # Encode features and targets
        X: List[List[float]] = []
        y: List[float] = []

        for event in events:
            target = self._encode_target(event)
            if target is not None:
                features = self._encode_features(event)
                X.append(features)
                y.append(target)

        n = len(X)
        if n < 5:
            logger.warning(
                "OverridePredictor: insufficient training data (%d < 5)", n
            )
            return {"error": "insufficient_data", "samples": n}

        # Compute normalization parameters
        self._compute_means_stds(X)
        X_norm = self._normalize_features(X)

        # Initialize weights
        self._weights = [0.0] * self.N_FEATURES
        self._bias = 0.0

        # Gradient descent
        losses: List[float] = []
        for epoch in range(self.n_epochs):
            total_loss = 0.0

            for i in range(n):
                # Forward pass
                pred = self._predict_raw(X_norm[i])

                # Binary cross-entropy loss
                eps = 1e-15
                loss = -(
                    y[i] * math.log(pred + eps)
                    + (1 - y[i]) * math.log(1 - pred + eps)
                )
                total_loss += loss

                # Gradient (pred - target) * feature
                error = pred - y[i]

                # Update weights with L2 regularization
                for j in range(self.N_FEATURES):
                    gradient = error * X_norm[i][j] + self.regularization * self._weights[j]
                    self._weights[j] -= self.learning_rate * gradient

                # Update bias (no regularization on bias)
                self._bias -= self.learning_rate * error

            avg_loss = total_loss / n
            losses.append(avg_loss)

            # Early stopping if converged
            if epoch > 20 and abs(losses[-1] - losses[-2]) < 1e-6:
                break

        # Compute training accuracy
        correct = 0
        for i in range(n):
            pred = self._predict_raw(X_norm[i])
            predicted_class = 1.0 if pred >= 0.5 else 0.0
            if predicted_class == y[i]:
                correct += 1

        self._train_accuracy = correct / n
        self._train_samples = n
        self._trained = True

        # Persist model
        self._save_model()

        metrics = {
            "samples": n,
            "epochs": len(losses),
            "final_loss": round(losses[-1], 4) if losses else 0.0,
            "accuracy": round(self._train_accuracy, 3),
            "loss_rate_in_data": round(sum(y) / n, 3),
            "weights": {
                name: round(w, 4)
                for name, w in zip(self.FEATURE_NAMES, self._weights)
            },
            "bias": round(self._bias, 4),
        }
        logger.info(
            "OverridePredictor trained: %d samples, %.1f%% accuracy, loss=%.4f",
            n, self._train_accuracy * 100, losses[-1] if losses else 0,
        )
        return metrics

    # --- Prediction ---

    def predict_loss_probability(
        self, context: Dict[str, Any]
    ) -> OverridePrediction:
        """Predict the probability that an override will result in a loss.

        Args:
            context: Current override context (same schema as OverrideEvent).

        Returns:
            OverridePrediction with loss probability, risk level, and factors.
        """
        features = self._encode_features(context)

        if self._trained:
            # Normalize using training statistics
            # US-208: OOD detection — flag features beyond 3σ from training mean
            normalized = []
            _ood_features = []
            for i, val in enumerate(features):
                std = self._feature_stds[i] if self._feature_stds[i] > 1e-8 else 1.0
                z = (val - self._feature_means[i]) / std
                normalized.append(z)
                if abs(z) > 3.0:
                    fname = self.FEATURE_NAMES[i] if i < len(self.FEATURE_NAMES) else f"feat_{i}"
                    _ood_features.append((fname, val, z))
            if _ood_features:
                logger.warning(
                    "US-208: OOD input in override predictor — %d feature(s) beyond 3σ: %s",
                    len(_ood_features),
                    [(f, f"val={v:.3f}, z={z:.1f}σ") for f, v, z in _ood_features],
                )
            loss_prob = self._predict_raw(normalized)
            # US-208: Cap confidence towards 0.5 when extrapolating
            if _ood_features:
                loss_prob = loss_prob * 0.6 + 0.5 * 0.4
        else:
            # Fallback: heuristic-based prediction using feature risk scores
            risk_sum = sum(features[2:6])  # emotional + cognitive + type + regime
            loss_prob = self._sigmoid(risk_sum * 2.0 - 0.5)

        # Compute feature contributions
        contributions = {}
        for i, (name, feat_val) in enumerate(
            zip(self.FEATURE_NAMES, features)
        ):
            if self._trained:
                # Weight * normalized feature value
                std = self._feature_stds[i] if self._feature_stds[i] > 1e-8 else 1.0
                contributions[name] = self._weights[i] * (
                    (feat_val - self._feature_means[i]) / std
                )
            else:
                contributions[name] = feat_val

        # Determine risk level
        if loss_prob >= 0.80:
            risk_level = "critical"
            recommendation = (
                "STRONGLY ADVISE AGAINST this override. "
                f"{loss_prob:.0%} predicted loss probability."
            )
        elif loss_prob >= 0.65:
            risk_level = "high"
            recommendation = (
                "This override is likely to lose. Consider following "
                f"Buddy's recommendation. ({loss_prob:.0%} loss probability)"
            )
        elif loss_prob >= 0.50:
            risk_level = "moderate"
            recommendation = (
                "Uncertain outcome — proceed with caution and reduced size. "
                f"({loss_prob:.0%} loss probability)"
            )
        else:
            risk_level = "low"
            recommendation = (
                f"Override has reasonable odds. ({loss_prob:.0%} loss probability)"
            )

        # Top risk factors (sorted by absolute contribution)
        sorted_factors = sorted(
            contributions.items(), key=lambda x: abs(x[1]), reverse=True
        )
        top_factors = [
            f"{name}: {val:+.2f}" for name, val in sorted_factors[:3]
            if abs(val) > 0.01
        ]

        return OverridePrediction(
            loss_probability=loss_prob,
            risk_level=risk_level,
            top_risk_factors=top_factors,
            recommendation=recommendation,
            feature_contributions=contributions,
        )

    def predict_batch(
        self, events: List[Dict[str, Any]]
    ) -> List[OverridePrediction]:
        """Predict loss probability for multiple override contexts."""
        return [self.predict_loss_probability(e) for e in events]

    # --- Model Status ---

    def get_model_info(self) -> Dict[str, Any]:
        """Return model metadata for display."""
        return {
            "trained": self._trained,
            "train_samples": self._train_samples,
            "train_accuracy": round(self._train_accuracy, 3),
            "n_features": self.N_FEATURES,
            "feature_names": self.FEATURE_NAMES,
            "weights": {
                name: round(w, 4)
                for name, w in zip(self.FEATURE_NAMES, self._weights)
            } if self._trained else {},
            "model_path": str(self.model_path),
        }

    # --- Persistence ---

    def _save_model(self) -> None:
        """Persist model weights to JSON."""
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            data = {
                "weights": self._weights,
                "bias": self._bias,
                "feature_means": self._feature_means,
                "feature_stds": self._feature_stds,
                "trained": self._trained,
                "train_samples": self._train_samples,
                "train_accuracy": self._train_accuracy,
                "saved_at": datetime.now(timezone.utc).isoformat(),
                "n_features": self.N_FEATURES,
                "feature_names": self.FEATURE_NAMES,
            }
            self.model_path.write_text(json.dumps(data, indent=2))
            logger.debug("Override predictor saved to %s", self.model_path)
        except Exception as e:
            logger.error("Failed to save override predictor: %s", e)

    def _load_model(self) -> None:
        """Load model weights from JSON."""
        if not self.model_path.exists():
            return
        try:
            data = json.loads(self.model_path.read_text())
            if data.get("n_features") != self.N_FEATURES:
                logger.warning("Override predictor: feature count mismatch, retraining needed")
                return
            self._weights = data["weights"]
            self._bias = data["bias"]
            self._feature_means = data["feature_means"]
            self._feature_stds = data["feature_stds"]
            self._trained = data.get("trained", False)
            self._train_samples = data.get("train_samples", 0)
            self._train_accuracy = data.get("train_accuracy", 0.0)
            logger.debug(
                "Override predictor loaded: %d samples, %.1f%% accuracy",
                self._train_samples, self._train_accuracy * 100,
            )
        except Exception as e:
            logger.warning("Failed to load override predictor: %s", e)

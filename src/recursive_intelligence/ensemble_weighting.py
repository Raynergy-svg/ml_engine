"""Differentiable ensemble weighting layer (Tier 6).

Learns a nonlinear mapping from regime context → ensemble member weights.
Uses a lightweight numpy MLP (~200 params) to avoid PyTorch dependency in the
inference path.

Integration:
    weighter = EnsembleWeighter(n_models=4, n_context=6)
    weights = weighter.predict(context_features)
    # weights is a softmax distribution over ensemble members

    # After trade outcome:
    weighter.record(context_features, model_predictions, realized_outcome)
    # Retrains every retrain_interval samples
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.scanner.automation.safe_json import safe_json_write

logger = logging.getLogger(__name__)

# Persistence
_STATE_FILE = "trained_data/ensemble_weighter_state.json"
_RETRAIN_INTERVAL = 30  # Retrain every N recorded outcomes
_MIN_SAMPLES = 20       # Minimum samples before using learned weights
_BUFFER_MAX = 500       # Max training buffer size (sliding window)


def _softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax."""
    e = np.exp(x - np.max(x))
    return e / (e.sum() + 1e-12)


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)


def _relu_deriv(x: np.ndarray) -> np.ndarray:
    return (x > 0.0).astype(float)


@dataclass
class WeighterConfig:
    """Configuration for the ensemble weighting MLP."""
    n_models: int = 4          # Number of ensemble members to weight
    n_context: int = 6         # Dimension of regime context features
    hidden_dim: int = 16       # Hidden layer size
    learning_rate: float = 0.005
    retrain_interval: int = _RETRAIN_INTERVAL
    min_samples: int = _MIN_SAMPLES
    buffer_max: int = _BUFFER_MAX
    epochs_per_retrain: int = 50
    weight_decay: float = 1e-4  # L2 regularization


class NumpyMLP:
    """Tiny 2-layer MLP in pure numpy for inference-path speed.

    Architecture:
        input (n_context + n_models) → Linear(hidden) → ReLU → Linear(n_models) → Softmax
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, seed: int = 42):
        rng = np.random.RandomState(seed)
        # Xavier initialization
        scale1 = np.sqrt(2.0 / (input_dim + hidden_dim))
        scale2 = np.sqrt(2.0 / (hidden_dim + output_dim))
        self.W1 = rng.randn(input_dim, hidden_dim).astype(np.float64) * scale1
        self.b1 = np.zeros(hidden_dim, dtype=np.float64)
        self.W2 = rng.randn(hidden_dim, output_dim).astype(np.float64) * scale2
        self.b2 = np.zeros(output_dim, dtype=np.float64)

    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Forward pass. Returns (softmax_output, hidden_activations)."""
        h = _relu(x @ self.W1 + self.b1)
        logits = h @ self.W2 + self.b2
        return _softmax(logits), h

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Return softmax weights for a single input."""
        out, _ = self.forward(x)
        return out

    def train_step(
        self,
        x: np.ndarray,
        target: np.ndarray,
        lr: float = 0.005,
        weight_decay: float = 1e-4,
    ) -> float:
        """Single gradient step (MSE loss). Returns loss value."""
        # Forward
        h_pre = x @ self.W1 + self.b1
        h = _relu(h_pre)
        logits = h @ self.W2 + self.b2
        pred = _softmax(logits)

        # Loss: MSE between predicted weights and oracle weights
        diff = pred - target
        loss = float(np.mean(diff ** 2))

        # Backward through softmax + MSE (simplified Jacobian)
        # d_logits ≈ diff * pred * (1 - pred) for diagonal Jacobian approx
        d_logits = diff * pred * (1.0 - pred + 1e-8)

        # Gradient for W2, b2
        dW2 = np.outer(h, d_logits)
        db2 = d_logits

        # Backprop through ReLU
        d_h = d_logits @ self.W2.T * _relu_deriv(h_pre)

        # Gradient for W1, b1
        dW1 = np.outer(x, d_h)
        db1 = d_h

        # Update with L2 regularization
        self.W1 -= lr * (dW1 + weight_decay * self.W1)
        self.b1 -= lr * db1
        self.W2 -= lr * (dW2 + weight_decay * self.W2)
        self.b2 -= lr * db2

        return loss

    def to_dict(self) -> Dict[str, Any]:
        return {
            "W1": self.W1.tolist(),
            "b1": self.b1.tolist(),
            "W2": self.W2.tolist(),
            "b2": self.b2.tolist(),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any], input_dim: int, hidden_dim: int, output_dim: int) -> "NumpyMLP":
        mlp = cls(input_dim, hidden_dim, output_dim)
        mlp.W1 = np.array(d["W1"], dtype=np.float64)
        mlp.b1 = np.array(d["b1"], dtype=np.float64)
        mlp.W2 = np.array(d["W2"], dtype=np.float64)
        mlp.b2 = np.array(d["b2"], dtype=np.float64)
        return mlp


class EnsembleWeighter:
    """Learns regime-conditioned ensemble weights via a small MLP.

    Usage:
        weighter = EnsembleWeighter(n_models=4, n_context=6)

        # At prediction time:
        context = [vol_pct, trend_str, session_id, spread_norm, recent_acc, regime_id]
        model_preds = [xgb_pred, lgbm_pred, ridge_pred, rf_pred]
        weights = weighter.get_weights(context, model_preds)
        final_pred = np.dot(weights, model_preds)

        # After trade closes:
        weighter.record(context, model_preds, realized_outcome=0.02)
    """

    def __init__(self, config: Optional[WeighterConfig] = None):
        self.config = config or WeighterConfig()
        input_dim = self.config.n_context + self.config.n_models
        self._mlp = NumpyMLP(
            input_dim=input_dim,
            hidden_dim=self.config.hidden_dim,
            output_dim=self.config.n_models,
        )
        # Training buffer: list of (context_features, model_preds, realized_outcome)
        self._buffer: List[Tuple[np.ndarray, np.ndarray, float]] = []
        self._samples_since_retrain: int = 0
        self._total_retrains: int = 0
        self._last_train_loss: float = 0.0
        self._is_trained: bool = False

        # Try to load persisted state
        self._load_state()

    def get_weights(
        self,
        context_features: List[float],
        model_predictions: List[float],
    ) -> np.ndarray:
        """Get learned ensemble weights conditioned on context.

        If insufficient training data, returns uniform weights (fallback).

        Args:
            context_features: Regime context [vol_pct, trend_str, session, spread, acc, regime]
            model_predictions: Raw predictions from each ensemble member

        Returns:
            np.ndarray of shape (n_models,) summing to 1.0
        """
        n = self.config.n_models
        if not self._is_trained or len(self._buffer) < self.config.min_samples:
            return np.ones(n) / n  # Uniform fallback

        try:
            x = np.array(list(context_features) + list(model_predictions), dtype=np.float64)
            weights = self._mlp.predict(x)
            return weights
        except Exception as e:
            logger.debug("EnsembleWeighter: prediction failed: %s", e)
            return np.ones(n) / n

    def record(
        self,
        context_features: List[float],
        model_predictions: List[float],
        realized_outcome: float,
    ) -> None:
        """Record a trade outcome for training.

        The oracle target is computed as: how well did each model predict the outcome?
        Models whose predictions were closer to the realized outcome get higher weight.
        """
        ctx = np.array(context_features, dtype=np.float64)
        preds = np.array(model_predictions, dtype=np.float64)

        self._buffer.append((ctx, preds, float(realized_outcome)))

        # Sliding window
        if len(self._buffer) > self.config.buffer_max:
            self._buffer = self._buffer[-self.config.buffer_max:]

        self._samples_since_retrain += 1

        # Auto-retrain at interval
        if (
            self._samples_since_retrain >= self.config.retrain_interval
            and len(self._buffer) >= self.config.min_samples
        ):
            self._retrain()

    def _retrain(self) -> None:
        """Retrain the MLP on the current buffer."""
        if len(self._buffer) < self.config.min_samples:
            return

        total_loss = 0.0
        n_samples = len(self._buffer)

        for epoch in range(self.config.epochs_per_retrain):
            epoch_loss = 0.0
            # Shuffle indices
            indices = np.random.permutation(n_samples)
            for idx in indices:
                ctx, preds, outcome = self._buffer[idx]
                x = np.concatenate([ctx, preds])

                # Oracle target: inverse-error weighting
                # Models closer to the outcome get higher weight
                errors = np.abs(preds - outcome) + 1e-6
                inv_errors = 1.0 / errors
                oracle_weights = inv_errors / inv_errors.sum()

                loss = self._mlp.train_step(
                    x, oracle_weights,
                    lr=self.config.learning_rate,
                    weight_decay=self.config.weight_decay,
                )
                epoch_loss += loss

            total_loss = epoch_loss / n_samples

        self._last_train_loss = total_loss
        self._samples_since_retrain = 0
        self._total_retrains += 1
        self._is_trained = True

        logger.info(
            "Tier 6: EnsembleWeighter retrained (samples=%d, loss=%.6f, retrains=%d)",
            n_samples, total_loss, self._total_retrains,
        )

        self._save_state()

    def _save_state(self) -> None:
        """Persist MLP weights and buffer to disk."""
        try:
            state = {
                "mlp": self._mlp.to_dict(),
                "buffer": [
                    {
                        "ctx": ctx.tolist(),
                        "preds": preds.tolist(),
                        "outcome": outcome,
                    }
                    for ctx, preds, outcome in self._buffer[-self.config.buffer_max:]
                ],
                "total_retrains": self._total_retrains,
                "last_train_loss": self._last_train_loss,
                "is_trained": self._is_trained,
                "config": {
                    "n_models": self.config.n_models,
                    "n_context": self.config.n_context,
                    "hidden_dim": self.config.hidden_dim,
                },
            }
            path = Path(_STATE_FILE)
            if not safe_json_write(path, state):
                logger.warning("EnsembleWeighter: state save failed")
                return
            logger.debug("EnsembleWeighter: state saved (%d buffer entries)", len(self._buffer))
        except Exception as e:
            logger.warning("EnsembleWeighter: save_state failed: %s", e)

    def _load_state(self) -> None:
        """Load persisted state if available."""
        path = Path(_STATE_FILE)
        if not path.exists():
            return
        try:
            state = json.loads(path.read_text())
            cfg = state.get("config", {})
            input_dim = cfg.get("n_context", self.config.n_context) + cfg.get("n_models", self.config.n_models)
            self._mlp = NumpyMLP.from_dict(
                state["mlp"],
                input_dim=input_dim,
                hidden_dim=cfg.get("hidden_dim", self.config.hidden_dim),
                output_dim=cfg.get("n_models", self.config.n_models),
            )
            self._buffer = [
                (
                    np.array(entry["ctx"], dtype=np.float64),
                    np.array(entry["preds"], dtype=np.float64),
                    float(entry["outcome"]),
                )
                for entry in state.get("buffer", [])
            ]
            self._total_retrains = state.get("total_retrains", 0)
            self._last_train_loss = state.get("last_train_loss", 0.0)
            self._is_trained = state.get("is_trained", False)
            logger.info(
                "EnsembleWeighter: loaded state (buffer=%d, retrains=%d, trained=%s)",
                len(self._buffer), self._total_retrains, self._is_trained,
            )
        except Exception as e:
            logger.warning("EnsembleWeighter: load_state failed, starting fresh: %s", e)

    @property
    def stats(self) -> Dict[str, Any]:
        return {
            "buffer_size": len(self._buffer),
            "total_retrains": self._total_retrains,
            "last_train_loss": self._last_train_loss,
            "is_trained": self._is_trained,
            "samples_since_retrain": self._samples_since_retrain,
            "n_params": (
                self._mlp.W1.size + self._mlp.b1.size +
                self._mlp.W2.size + self._mlp.b2.size
            ),
        }

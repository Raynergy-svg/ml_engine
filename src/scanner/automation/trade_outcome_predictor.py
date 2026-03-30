"""Trade outcome prediction from post-entry features.

US-082: Once a trade is open, new information becomes available (MAE,
MFE, momentum since entry, distance to SL/TP, time in trade). Train a
GradientBoosting classifier on historical trade features to predict
whether an open trade will win or lose. Use predictions to suggest
early exits or position adjustments.

Features (7 post-entry):
1. dist_to_sl_pct — current price distance to SL as % of SL distance
2. dist_to_tp_pct — current price distance to TP as % of TP distance
3. mae_pips — Maximum Adverse Excursion (worst drawdown since entry)
4. mfe_pips — Maximum Favorable Excursion (best profit since entry)
5. bars_in_trade — how long the trade has been open (in bars)
6. momentum_since_entry — price momentum since entry
7. current_vol_vs_entry_vol — volatility ratio (current/entry)

Actions:
- HOLD: predicted win probability > 0.55
- TIGHTEN_SL: predicted win probability 0.40-0.55 (reduce risk)
- TAKE_PARTIAL: MFE is high but win probability dropping
- EXIT_EARLY: predicted win probability < 0.30

Minimum 50 historical trades required for training.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_JOURNAL_PATH = _PROJECT_ROOT / "trained_data" / "trade_journal_rl.json"
_MODEL_PATH = _PROJECT_ROOT / "trained_data" / "trade_outcome_model.json"

MIN_TRADES_FOR_TRAINING = 50

FEATURE_NAMES = [
    "dist_to_sl_pct",
    "dist_to_tp_pct",
    "mae_pips",
    "mfe_pips",
    "bars_in_trade",
    "momentum_since_entry",
    "current_vol_vs_entry_vol",
]


@dataclass
class PredictionResult:
    """Prediction for an open trade."""

    win_probability: float = 0.5
    suggested_action: str = "HOLD"
    confidence: float = 0.0
    feature_importances: Dict[str, float] = field(default_factory=dict)
    reason: str = ""
    is_trained: bool = False

    def to_dict(self) -> dict:
        return {
            "win_probability": round(self.win_probability, 4),
            "suggested_action": self.suggested_action,
            "confidence": round(self.confidence, 4),
            "feature_importances": {
                k: round(v, 4) for k, v in self.feature_importances.items()
            },
            "reason": self.reason,
            "is_trained": self.is_trained,
        }


class TradeOutcomePredictor:
    """Predicts open trade outcomes from post-entry features.

    Uses GradientBoosting (sklearn) when available, falls back to
    logistic regression on key features. Suggests trade management
    actions based on predicted win probability.
    """

    def __init__(
        self,
        journal_path: Optional[Path] = None,
        model_path: Optional[Path] = None,
        min_trades: int = MIN_TRADES_FOR_TRAINING,
    ):
        self.journal_path = journal_path or _JOURNAL_PATH
        self.model_path = model_path or _MODEL_PATH
        self.min_trades = min_trades

        # Model state
        self._is_trained: bool = False
        self._model_params: Dict[str, Any] = {}
        self._feature_importances: Dict[str, float] = {}
        self._train_accuracy: float = 0.0

        # Try to load persisted model
        self._load_model()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict(self, features: Dict[str, float]) -> PredictionResult:
        """Predict outcome for an open trade.

        Args:
            features: Dict with keys matching FEATURE_NAMES

        Returns:
            PredictionResult with win probability and suggested action
        """
        result = PredictionResult()

        if not self._is_trained:
            result.reason = "Model not trained (insufficient historical trades)"
            return result

        try:
            # Extract feature vector in correct order
            x = np.array([features.get(f, 0.0) for f in FEATURE_NAMES])

            # Predict using stored model parameters
            win_prob = self._predict_from_params(x)
            win_prob = max(0.01, min(0.99, win_prob))

            result.win_probability = win_prob
            result.is_trained = True
            result.confidence = abs(win_prob - 0.5) * 2  # 0-1 scale of certainty
            result.feature_importances = dict(self._feature_importances)

            # Determine action
            mfe = features.get("mfe_pips", 0.0)
            dist_to_tp = features.get("dist_to_tp_pct", 1.0)

            if win_prob >= 0.55:
                result.suggested_action = "HOLD"
                result.reason = f"Win probability {win_prob:.0%} — maintain position"
            elif win_prob >= 0.40:
                result.suggested_action = "TIGHTEN_SL"
                result.reason = f"Win probability {win_prob:.0%} — reduce risk"
            elif mfe > 10 and dist_to_tp < 0.5:
                result.suggested_action = "TAKE_PARTIAL"
                result.reason = (
                    f"Win probability {win_prob:.0%} but MFE={mfe:.1f} — "
                    f"lock in partial profit"
                )
            else:
                result.suggested_action = "EXIT_EARLY"
                result.reason = f"Win probability {win_prob:.0%} — consider exit"

        except Exception as e:
            result.reason = f"Prediction error: {e}"
            logger.debug(f"TradeOutcomePredictor prediction error: {e}")

        return result

    def fit_from_journal(self) -> bool:
        """Train model from trade journal.

        Returns:
            True if training succeeded
        """
        trades = self._load_trades()
        if not trades:
            logger.debug("TradeOutcomePredictor: No trades in journal")
            return False

        X, y = self._extract_features(trades)
        if len(X) < self.min_trades:
            logger.info(
                f"TradeOutcomePredictor: Only {len(X)} trades "
                f"(need {self.min_trades}), skipping training"
            )
            return False

        return self._fit(np.array(X), np.array(y))

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def _fit(self, X: np.ndarray, y: np.ndarray) -> bool:
        """Train the outcome prediction model.

        Tries GradientBoosting first, falls back to logistic regression.
        """
        try:
            from sklearn.ensemble import GradientBoostingClassifier
            from sklearn.metrics import accuracy_score
            return self._fit_sklearn(X, y)
        except ImportError:
            return self._fit_manual(X, y)

    def _fit_sklearn(self, X: np.ndarray, y: np.ndarray) -> bool:
        """Train with sklearn GradientBoosting."""
        try:
            from sklearn.ensemble import GradientBoostingClassifier
            from sklearn.metrics import accuracy_score

            model = GradientBoostingClassifier(
                n_estimators=50,
                max_depth=3,
                learning_rate=0.1,
                min_samples_leaf=5,
                random_state=42,
            )
            model.fit(X, y)

            y_pred = model.predict(X)
            acc = accuracy_score(y, y_pred)

            # Store feature importances
            importances = model.feature_importances_
            self._feature_importances = {
                FEATURE_NAMES[i]: float(importances[i])
                for i in range(len(FEATURE_NAMES))
            }

            # Store model as simplified logistic approximation for persistence
            # (We can't easily serialize the full GBM, so extract key thresholds)
            y_prob = model.predict_proba(X)[:, 1]

            # Fit a simple logistic on top for portable persistence
            from sklearn.linear_model import LogisticRegression
            lr = LogisticRegression(penalty=None, solver="lbfgs", max_iter=500)
            lr.fit(X, y)

            self._model_params = {
                "type": "logistic_approx",
                "coefs": [float(c) for c in lr.coef_[0]],
                "intercept": float(lr.intercept_[0]),
                "gbm_accuracy": float(acc),
                "n_trades": int(len(y)),
                "win_rate": float(y.mean()),
            }
            self._train_accuracy = acc
            self._is_trained = True

            self._save_model()

            logger.info(
                f"TradeOutcomePredictor: Trained on {len(y)} trades — "
                f"GBM accuracy={acc:.2%}, win_rate={y.mean():.2%}"
            )
            return True

        except Exception as e:
            logger.warning(f"TradeOutcomePredictor: sklearn fit failed: {e}")
            return self._fit_manual(X, y)

    def _fit_manual(self, X: np.ndarray, y: np.ndarray) -> bool:
        """Manual logistic regression fallback."""
        try:
            n_features = X.shape[1]
            n = len(y)

            # Standardize features
            means = X.mean(axis=0)
            stds = X.std(axis=0)
            stds[stds < 1e-10] = 1.0
            X_std = (X - means) / stds

            # Simple gradient descent logistic regression
            w = np.zeros(n_features)
            b = 0.0
            lr = 0.01

            for _ in range(500):
                z = X_std @ w + b
                z = np.clip(z, -500, 500)
                pred = 1.0 / (1.0 + np.exp(-z))
                error = pred - y
                w -= lr * (X_std.T @ error) / n
                b -= lr * error.mean()

            # Store
            # Convert back to original scale: z = sum(w_i * (x_i - mean_i) / std_i) + b
            # = sum((w_i/std_i) * x_i) - sum(w_i * mean_i / std_i) + b
            coefs_orig = w / stds
            intercept_orig = b - float((w * means / stds).sum())

            z_final = X @ coefs_orig + intercept_orig
            z_final = np.clip(z_final, -500, 500)
            probs = 1.0 / (1.0 + np.exp(-z_final))
            acc = float(((probs >= 0.5) == y).mean())

            self._model_params = {
                "type": "logistic_manual",
                "coefs": [float(c) for c in coefs_orig],
                "intercept": float(intercept_orig),
                "gbm_accuracy": float(acc),
                "n_trades": int(n),
                "win_rate": float(y.mean()),
            }
            self._train_accuracy = acc
            self._is_trained = True

            # Feature importance approximation: |coefficient| * std
            abs_coefs = np.abs(coefs_orig)
            total = abs_coefs.sum()
            if total > 0:
                self._feature_importances = {
                    FEATURE_NAMES[i]: float(abs_coefs[i] / total)
                    for i in range(n_features)
                }

            self._save_model()

            logger.info(
                f"TradeOutcomePredictor (manual): Trained on {n} trades — "
                f"accuracy={acc:.2%}"
            )
            return True

        except Exception as e:
            logger.warning(f"TradeOutcomePredictor: Manual fit failed: {e}")
            return False

    def _predict_from_params(self, x: np.ndarray) -> float:
        """Predict using stored logistic parameters."""
        coefs = np.array(self._model_params.get("coefs", []))
        intercept = self._model_params.get("intercept", 0.0)

        if len(coefs) != len(x):
            return 0.5

        z = float(np.dot(coefs, x) + intercept)
        z = max(-500.0, min(500.0, z))
        return 1.0 / (1.0 + math.exp(-z))

    # ------------------------------------------------------------------
    # Data extraction
    # ------------------------------------------------------------------

    def _extract_features(
        self,
        trades: List[Dict[str, Any]],
    ) -> Tuple[List[List[float]], List[int]]:
        """Extract feature vectors from trade journal."""
        X: List[List[float]] = []
        y: List[int] = []

        for t in trades:
            outcome = (t.get("outcome") or "").upper() if isinstance(t.get("outcome"), str) else ""
            # Handle dict-style outcome from trade_journal_rl.json
            if isinstance(t.get("outcome"), dict):
                outcome = "WIN" if t["outcome"].get("pnl", 0) > 0 else "LOSS"
            if outcome in ("WIN", "WON", "TP_HIT"):
                label = 1
            elif outcome in ("LOSS", "LOST", "SL_HIT", "STOPPED"):
                label = 0
            else:
                continue

            # Extract features (use defaults when missing)
            sl_pips = float(t.get("sl_pips", 10.0))
            tp_pips = float(t.get("tp_pips", 20.0))
            entry = float(t.get("entry_price", 0.0))
            close_price = float(t.get("close_price", entry))

            # Compute synthetic post-entry features from available data
            if sl_pips <= 0:
                continue

            pnl = float(t.get("pnl_pips", 0.0))
            mae = float(t.get("mae_pips", abs(min(pnl, 0))))
            mfe = float(t.get("mfe_pips", abs(max(pnl, 0))))

            dist_to_sl_pct = mae / sl_pips if sl_pips > 0 else 0.5
            dist_to_tp_pct = mfe / tp_pips if tp_pips > 0 else 0.5
            bars = float(t.get("bars_in_trade", 10))
            momentum = float(t.get("momentum_since_entry", pnl / max(sl_pips, 1.0)))
            vol_ratio = float(t.get("vol_ratio", 1.0))

            features = [
                min(dist_to_sl_pct, 2.0),
                min(dist_to_tp_pct, 2.0),
                mae,
                mfe,
                min(bars, 200),
                max(-2.0, min(2.0, momentum)),
                max(0.2, min(3.0, vol_ratio)),
            ]

            X.append(features)
            y.append(label)

        return X, y

    def _load_trades(self) -> List[Dict[str, Any]]:
        """Load trades from journal."""
        if not self.journal_path.exists():
            return []
        try:
            try:
                from src.scanner.automation.safe_json import safe_json_read
                data = safe_json_read(self.journal_path, default=[])
            except ImportError:
                data = json.loads(self.journal_path.read_text())
            return data if isinstance(data, list) else []
        except Exception as e:
            logger.debug(f"TradeOutcomePredictor: Failed to load journal: {e}")
            return []

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save_model(self) -> None:
        """Persist model parameters."""
        try:
            data = {
                "version": 1,
                "params": self._model_params,
                "feature_importances": {
                    k: round(v, 6) for k, v in self._feature_importances.items()
                },
                "train_accuracy": round(self._train_accuracy, 4),
                "last_updated": datetime.now(timezone.utc).isoformat(),
            }
            self.model_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                from src.scanner.automation.safe_json import safe_json_write
                safe_json_write(self.model_path, data)
            except ImportError:
                self.model_path.write_text(json.dumps(data, indent=2))
        except Exception as e:
            logger.debug(f"TradeOutcomePredictor: Failed to save model: {e}")

    def _load_model(self) -> None:
        """Load persisted model parameters."""
        if not self.model_path.exists():
            return
        try:
            try:
                from src.scanner.automation.safe_json import safe_json_read
                data = safe_json_read(self.model_path, default={})
            except ImportError:
                data = json.loads(self.model_path.read_text())

            params = data.get("params", {})
            if params and params.get("n_trades", 0) >= self.min_trades:
                self._model_params = params
                self._feature_importances = data.get("feature_importances", {})
                self._train_accuracy = data.get("train_accuracy", 0.0)
                self._is_trained = True
                logger.info(
                    f"TradeOutcomePredictor: Loaded model "
                    f"(n={params.get('n_trades', 0)}, "
                    f"acc={self._train_accuracy:.2%})"
                )
        except Exception as e:
            logger.debug(f"TradeOutcomePredictor: Failed to load model: {e}")

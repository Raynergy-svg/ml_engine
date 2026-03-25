"""Adaptive confidence calibration via Platt scaling.

US-079: Raw model confidence scores are poorly calibrated — a 0.70
prediction doesn't truly mean 70% actual win rate. Apply Platt scaling
(logistic regression on model outputs vs actual outcomes) to transform
raw scores into calibrated probabilities.

Pipeline:
1. Load historical trades from trade_journal_rl.json
2. Extract (raw_confidence, binary_outcome) pairs
3. Fit sklearn LogisticRegression as sigmoid calibrator
4. At scan time: calibrate(raw_score) → calibrated probability
5. Periodically refit as new trades accumulate

Minimum 30 closed trades required before calibration activates.
Falls back to raw confidence if insufficient data or calibration fails.
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
_CALIBRATION_PATH = _PROJECT_ROOT / "trained_data" / "confidence_calibration.json"

# Minimum trades before calibration activates
MIN_TRADES_FOR_CALIBRATION = 30

# Refit interval: retrain after this many new trades since last fit
REFIT_INTERVAL = 20


@dataclass
class CalibrationParams:
    """Persisted sigmoid calibration parameters.

    Platt scaling fits: P(win | score) = 1 / (1 + exp(A*score + B))
    where A and B are learned from logistic regression.
    """

    coef: float = 0.0  # A (logistic regression coefficient)
    intercept: float = 0.0  # B (logistic regression intercept)
    n_trades_used: int = 0
    train_accuracy: float = 0.0
    train_log_loss: float = 0.0
    raw_mean_confidence: float = 0.0
    calibrated_mean: float = 0.0
    fit_timestamp: str = ""

    def to_dict(self) -> dict:
        return {
            "coef": round(self.coef, 6),
            "intercept": round(self.intercept, 6),
            "n_trades_used": self.n_trades_used,
            "train_accuracy": round(self.train_accuracy, 4),
            "train_log_loss": round(self.train_log_loss, 4),
            "raw_mean_confidence": round(self.raw_mean_confidence, 4),
            "calibrated_mean": round(self.calibrated_mean, 4),
            "fit_timestamp": self.fit_timestamp,
        }


@dataclass
class CalibrationResult:
    """Result of calibrating a single confidence score."""

    raw_score: float = 0.0
    calibrated_score: float = 0.0
    is_calibrated: bool = False  # False = fell back to raw
    reason: str = ""

    def to_dict(self) -> dict:
        return {
            "raw_score": round(self.raw_score, 4),
            "calibrated_score": round(self.calibrated_score, 4),
            "is_calibrated": self.is_calibrated,
            "reason": self.reason,
        }


class ConfidenceCalibrator:
    """Platt scaling confidence calibrator.

    Uses logistic regression to map raw model confidence scores to
    calibrated probabilities based on historical trade outcomes.

    Key properties:
    - Minimum 30 trades before activation (safety)
    - Sigmoid transform preserves monotonicity (higher raw → higher cal)
    - Auto-refit every 20 new trades
    - Falls back to raw score on any failure
    """

    def __init__(
        self,
        journal_path: Optional[Path] = None,
        calibration_path: Optional[Path] = None,
        min_trades: int = MIN_TRADES_FOR_CALIBRATION,
        refit_interval: int = REFIT_INTERVAL,
    ):
        self.journal_path = journal_path or _JOURNAL_PATH
        self.calibration_path = calibration_path or _CALIBRATION_PATH
        self.min_trades = min_trades
        self.refit_interval = refit_interval

        # Calibration state
        self._params: Optional[CalibrationParams] = None
        self._is_fitted: bool = False
        self._trades_since_fit: int = 0

        # Attempt to load persisted calibration
        self._load_calibration()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def calibrate(self, raw_score: float) -> CalibrationResult:
        """Calibrate a raw confidence score to a true probability.

        Args:
            raw_score: Raw model confidence (0.0-1.0)

        Returns:
            CalibrationResult with calibrated score and metadata
        """
        result = CalibrationResult(raw_score=raw_score)

        if not self._is_fitted or self._params is None:
            result.calibrated_score = raw_score
            result.is_calibrated = False
            result.reason = "Not fitted (insufficient trades or no calibration loaded)"
            return result

        try:
            # Platt scaling: P = sigmoid(A*x + B)
            logit = self._params.coef * raw_score + self._params.intercept
            calibrated = 1.0 / (1.0 + math.exp(-logit))

            # Clamp to reasonable range
            calibrated = max(0.01, min(0.99, calibrated))

            result.calibrated_score = calibrated
            result.is_calibrated = True
            result.reason = (
                f"Platt scaled (n={self._params.n_trades_used}, "
                f"acc={self._params.train_accuracy:.2%})"
            )
        except (OverflowError, ValueError) as e:
            # Fallback on math errors
            result.calibrated_score = raw_score
            result.is_calibrated = False
            result.reason = f"Calibration math error: {e}"
            logger.debug(f"Calibration math error for score {raw_score}: {e}")

        return result

    def fit_from_journal(self) -> bool:
        """Fit calibrator from the trade journal.

        Returns:
            True if successfully fitted, False otherwise
        """
        trades = self._load_trades()
        if not trades:
            logger.debug("ConfidenceCalibrator: No trades found in journal")
            return False

        # Extract (raw_confidence, outcome) pairs
        scores, outcomes = self._extract_training_data(trades)

        if len(scores) < self.min_trades:
            logger.info(
                f"ConfidenceCalibrator: Only {len(scores)} trades "
                f"(need {self.min_trades}), skipping calibration"
            )
            return False

        return self._fit(np.array(scores), np.array(outcomes))

    def should_refit(self) -> bool:
        """Check if recalibration is needed."""
        return self._trades_since_fit >= self.refit_interval

    def record_new_trade(self) -> None:
        """Notify calibrator that a new trade closed (for refit tracking)."""
        self._trades_since_fit += 1

    def get_params(self) -> Optional[Dict[str, Any]]:
        """Return current calibration parameters."""
        if self._params:
            return self._params.to_dict()
        return None

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def _fit(self, scores: np.ndarray, outcomes: np.ndarray) -> bool:
        """Fit Platt scaling logistic regression.

        Args:
            scores: Raw confidence scores (n,)
            outcomes: Binary outcomes, 1=WIN, 0=LOSS (n,)

        Returns:
            True if fit succeeded
        """
        try:
            from sklearn.linear_model import LogisticRegression
            from sklearn.metrics import accuracy_score, log_loss
        except ImportError:
            # Fallback: manual logistic regression via Newton-Raphson
            return self._fit_manual(scores, outcomes)

        try:
            X = scores.reshape(-1, 1)
            y = outcomes.astype(int)

            # Platt scaling uses logistic regression with no regularization
            model = LogisticRegression(
                penalty=None,
                solver="lbfgs",
                max_iter=1000,
                random_state=42,
            )
            model.fit(X, y)

            # Extract parameters
            coef = float(model.coef_[0][0])
            intercept = float(model.intercept_[0])

            # Evaluate
            y_pred = model.predict(X)
            y_prob = model.predict_proba(X)[:, 1]
            acc = accuracy_score(y, y_pred)
            ll = log_loss(y, y_prob)

            self._params = CalibrationParams(
                coef=coef,
                intercept=intercept,
                n_trades_used=len(scores),
                train_accuracy=acc,
                train_log_loss=ll,
                raw_mean_confidence=float(scores.mean()),
                calibrated_mean=float(y_prob.mean()),
                fit_timestamp=datetime.now(timezone.utc).isoformat(),
            )
            self._is_fitted = True
            self._trades_since_fit = 0

            # Persist
            self._save_calibration()

            logger.info(
                f"ConfidenceCalibrator: Fitted on {len(scores)} trades — "
                f"coef={coef:.4f}, intercept={intercept:.4f}, "
                f"accuracy={acc:.2%}, log_loss={ll:.4f}"
            )
            return True

        except Exception as e:
            logger.warning(f"ConfidenceCalibrator: sklearn fit failed: {e}")
            return self._fit_manual(scores, outcomes)

    def _fit_manual(self, scores: np.ndarray, outcomes: np.ndarray) -> bool:
        """Manual Platt scaling when sklearn unavailable.

        Uses simple binning approach: compute empirical win rate per
        confidence bin, then fit sigmoid via least-squares.
        """
        try:
            n = len(scores)
            if n < self.min_trades:
                return False

            # Bin scores into 5 buckets and compute empirical win rates
            bins = np.linspace(0.0, 1.0, 6)
            bin_indices = np.digitize(scores, bins) - 1
            bin_indices = np.clip(bin_indices, 0, 4)

            bin_centers = []
            bin_win_rates = []
            for b in range(5):
                mask = bin_indices == b
                if mask.sum() >= 3:  # Need at least 3 trades per bin
                    bin_centers.append(float(bins[b] + bins[b + 1]) / 2.0)
                    bin_win_rates.append(float(outcomes[mask].mean()))

            if len(bin_centers) < 2:
                return False

            # Simple linear regression on logit(win_rate) vs bin_center
            bc = np.array(bin_centers)
            # Clamp win rates to avoid log(0)
            bwr = np.array(bin_win_rates)
            bwr = np.clip(bwr, 0.05, 0.95)
            logit_wr = np.log(bwr / (1.0 - bwr))

            # Least squares: logit = A*x + B
            A_matrix = np.vstack([bc, np.ones(len(bc))]).T
            result = np.linalg.lstsq(A_matrix, logit_wr, rcond=None)
            coef, intercept = float(result[0][0]), float(result[0][1])

            # Compute calibrated predictions
            logits = coef * scores + intercept
            cal_probs = 1.0 / (1.0 + np.exp(-logits))
            acc = float(((cal_probs >= 0.5) == outcomes).mean())

            self._params = CalibrationParams(
                coef=coef,
                intercept=intercept,
                n_trades_used=n,
                train_accuracy=acc,
                train_log_loss=0.0,  # Skip log_loss for manual fit
                raw_mean_confidence=float(scores.mean()),
                calibrated_mean=float(cal_probs.mean()),
                fit_timestamp=datetime.now(timezone.utc).isoformat(),
            )
            self._is_fitted = True
            self._trades_since_fit = 0
            self._save_calibration()

            logger.info(
                f"ConfidenceCalibrator (manual): Fitted on {n} trades — "
                f"coef={coef:.4f}, intercept={intercept:.4f}, accuracy={acc:.2%}"
            )
            return True

        except Exception as e:
            logger.warning(f"ConfidenceCalibrator: Manual fit failed: {e}")
            return False

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

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
            logger.debug(f"ConfidenceCalibrator: Failed to load journal: {e}")
            return []

    def _extract_training_data(
        self,
        trades: List[Dict[str, Any]],
    ) -> Tuple[List[float], List[int]]:
        """Extract (confidence, outcome) pairs from trade journal.

        Only uses closed trades with known outcomes.
        """
        scores: List[float] = []
        outcomes: List[int] = []

        for t in trades:
            # Need confidence and outcome
            conf = t.get("confidence")
            outcome = t.get("outcome", "").upper()

            if conf is None or not outcome:
                continue

            # Determine binary outcome
            if outcome in ("WIN", "WON", "TP_HIT"):
                binary = 1
            elif outcome in ("LOSS", "LOST", "SL_HIT", "STOPPED"):
                binary = 0
            else:
                continue  # Skip open/pending trades

            try:
                raw_conf = float(conf)
                if 0.0 <= raw_conf <= 1.0:
                    scores.append(raw_conf)
                    outcomes.append(binary)
            except (ValueError, TypeError):
                continue

        return scores, outcomes

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save_calibration(self) -> None:
        """Persist calibration parameters to disk."""
        if not self._params:
            return
        try:
            data = {
                "version": 1,
                "params": self._params.to_dict(),
                "last_updated": datetime.now(timezone.utc).isoformat(),
            }
            self.calibration_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                from src.scanner.automation.safe_json import safe_json_write
                safe_json_write(self.calibration_path, data)
            except ImportError:
                self.calibration_path.write_text(json.dumps(data, indent=2))
            logger.debug("ConfidenceCalibrator: Saved calibration to disk")
        except Exception as e:
            logger.debug(f"ConfidenceCalibrator: Failed to save: {e}")

    def _load_calibration(self) -> None:
        """Load persisted calibration parameters."""
        if not self.calibration_path.exists():
            return
        try:
            try:
                from src.scanner.automation.safe_json import safe_json_read
                data = safe_json_read(self.calibration_path, default={})
            except ImportError:
                data = json.loads(self.calibration_path.read_text())

            params = data.get("params", {})
            if params and params.get("n_trades_used", 0) >= self.min_trades:
                self._params = CalibrationParams(
                    coef=params.get("coef", 0.0),
                    intercept=params.get("intercept", 0.0),
                    n_trades_used=params.get("n_trades_used", 0),
                    train_accuracy=params.get("train_accuracy", 0.0),
                    train_log_loss=params.get("train_log_loss", 0.0),
                    raw_mean_confidence=params.get("raw_mean_confidence", 0.0),
                    calibrated_mean=params.get("calibrated_mean", 0.0),
                    fit_timestamp=params.get("fit_timestamp", ""),
                )
                self._is_fitted = True
                logger.info(
                    f"ConfidenceCalibrator: Loaded calibration "
                    f"(n={self._params.n_trades_used}, "
                    f"coef={self._params.coef:.4f})"
                )
        except Exception as e:
            logger.debug(f"ConfidenceCalibrator: Failed to load calibration: {e}")

"""
Confidence Calibration Seeder — Phase 50 (US-312).

Seeds the confidence calibration system from historical trade data
using Platt scaling (logistic regression).

Given (confidence, won) pairs, fits P(win|conf) = sigmoid(a*conf + b).
Saves fitted coefficients for use by confidence_calibration.py.

Usage:
    seeder = CalibrationSeeder()
    result = seeder.seed_from_journal("trained_data/trade_journal_rl.json")
    seeder.save_params(result, "trained_data/models/calibration_params.json")
"""

import json
import logging
import math
import os
from dataclasses import dataclass, asdict
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class CalibrationParams:
    """Fitted Platt scaling parameters."""
    a: float               # Slope coefficient
    b: float               # Intercept
    n_samples: int         # Number of training samples
    n_holdout: int         # Number of holdout samples
    train_win_rate: float  # Win rate in training set
    holdout_win_rate: float  # Win rate in holdout set
    calibration_error: float  # Mean absolute calibration error on holdout
    confidence_range: Tuple[float, float] = (0.0, 1.0)
    is_valid: bool = True  # Whether calibration is reliable


@dataclass
class CalibrationSeederConfig:
    """Configuration for calibration seeding."""
    holdout_ratio: float = 0.20    # 20% holdout
    max_iterations: int = 100      # Gradient descent iterations
    learning_rate: float = 0.01    # Gradient descent step size
    min_samples: int = 50          # Minimum trades for calibration
    n_calibration_bins: int = 10   # Bins for calibration error measurement


class CalibrationSeeder:
    """Seeds confidence calibration from historical trades."""

    def __init__(self, config: Optional[CalibrationSeederConfig] = None):
        self.config = config or CalibrationSeederConfig()

    def seed_from_journal(self, journal_path: str) -> Optional[CalibrationParams]:
        """Load journal and fit Platt scaling."""
        pairs = self._extract_pairs(journal_path)
        if len(pairs) < self.config.min_samples:
            logger.warning(
                f"Phase 50: Only {len(pairs)} samples, need {self.config.min_samples} for calibration"
            )
            return None

        # Split into train and holdout
        split_idx = int(len(pairs) * (1 - self.config.holdout_ratio))
        train = pairs[:split_idx]
        holdout = pairs[split_idx:]

        logger.info(f"Phase 50: Calibration — {len(train)} train, {len(holdout)} holdout")

        # Fit Platt scaling on training set
        a, b = self._fit_platt(train)

        # Evaluate on holdout
        cal_error = self._calibration_error(holdout, a, b)
        train_wr = sum(1 for _, w in train if w) / len(train) if train else 0
        holdout_wr = sum(1 for _, w in holdout if w) / len(holdout) if holdout else 0

        confs = [c for c, _ in pairs]
        result = CalibrationParams(
            a=round(a, 6),
            b=round(b, 6),
            n_samples=len(train),
            n_holdout=len(holdout),
            train_win_rate=round(train_wr, 4),
            holdout_win_rate=round(holdout_wr, 4),
            calibration_error=round(cal_error, 4),
            confidence_range=(round(min(confs), 4), round(max(confs), 4)),
            is_valid=cal_error < 0.15 and len(holdout) >= 20,
        )

        logger.info(
            f"Phase 50: Calibration fitted — a={a:.4f}, b={b:.4f}, "
            f"cal_error={cal_error:.4f}, valid={result.is_valid}"
        )
        return result

    def _extract_pairs(self, journal_path: str) -> List[Tuple[float, bool]]:
        """Extract (confidence, won) pairs from journal."""
        try:
            with open(journal_path, "r") as f:
                raw = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.error(f"Phase 50: Failed to load journal: {e}")
            return []

        pairs = []
        for entry in raw:
            if not isinstance(entry, dict):
                continue
            conf = entry.get("confidence", entry.get("entry_confidence"))
            if not isinstance(conf, (int, float)) or conf <= 0:
                continue
            outcome = entry.get("outcome")
            if isinstance(outcome, dict):
                won = outcome.get("trade_won", False)
            elif isinstance(outcome, str):
                won = outcome.lower() == "win"
            else:
                continue
            pairs.append((float(conf), bool(won)))

        return pairs

    def _fit_platt(self, data: List[Tuple[float, bool]]) -> Tuple[float, float]:
        """Fit Platt scaling via gradient descent on log-loss."""
        a = 1.0  # Initial slope
        b = 0.0  # Initial intercept
        lr = self.config.learning_rate

        for _ in range(self.config.max_iterations):
            grad_a = 0.0
            grad_b = 0.0
            for conf, won in data:
                p = _sigmoid(a * conf + b)
                y = 1.0 if won else 0.0
                error = p - y
                grad_a += error * conf
                grad_b += error

            n = len(data)
            a -= lr * grad_a / n
            b -= lr * grad_b / n

        return a, b

    def _calibration_error(
        self, holdout: List[Tuple[float, bool]], a: float, b: float
    ) -> float:
        """Compute mean absolute calibration error on holdout set."""
        n_bins = self.config.n_calibration_bins
        if not holdout:
            return 1.0

        # Bin by predicted probability
        bins: dict = {i: [] for i in range(n_bins)}
        for conf, won in holdout:
            p = _sigmoid(a * conf + b)
            bin_idx = min(int(p * n_bins), n_bins - 1)
            bins[bin_idx].append((p, 1.0 if won else 0.0))

        total_error = 0.0
        n_nonempty = 0
        for bin_items in bins.values():
            if not bin_items:
                continue
            avg_pred = sum(p for p, _ in bin_items) / len(bin_items)
            avg_actual = sum(y for _, y in bin_items) / len(bin_items)
            total_error += abs(avg_pred - avg_actual)
            n_nonempty += 1

        return total_error / n_nonempty if n_nonempty > 0 else 1.0

    def calibrate_confidence(self, raw_confidence: float, params: CalibrationParams) -> float:
        """Apply Platt scaling to a raw confidence score."""
        if not params.is_valid:
            return raw_confidence
        return _sigmoid(params.a * raw_confidence + params.b)

    def save_params(self, params: CalibrationParams, path: str) -> bool:
        """Save calibration parameters atomically."""
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            data = asdict(params)
            # Convert tuple to list for JSON
            data["confidence_range"] = list(data["confidence_range"])
            tmp = path + ".tmp"
            with open(tmp, "w") as f:
                json.dump(data, f, indent=2, sort_keys=True)
            os.rename(tmp, path)
            logger.info(f"Phase 50: Saved calibration params to {path}")
            return True
        except Exception as e:
            logger.error(f"Phase 50: Failed to save calibration params: {e}")
            return False

    @staticmethod
    def load_params(path: str) -> Optional[CalibrationParams]:
        """Load calibration parameters from file."""
        try:
            with open(path, "r") as f:
                data = json.load(f)
            data["confidence_range"] = tuple(data.get("confidence_range", [0.0, 1.0]))
            return CalibrationParams(**data)
        except (FileNotFoundError, json.JSONDecodeError, TypeError) as e:
            logger.debug(f"Phase 50: No calibration params at {path}: {e}")
            return None


def _sigmoid(x: float) -> float:
    """Numerically stable sigmoid function."""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    else:
        ex = math.exp(x)
        return ex / (1.0 + ex)


def create_default_calibration_seeder() -> CalibrationSeeder:
    """Create a CalibrationSeeder with default config."""
    return CalibrationSeeder(CalibrationSeederConfig())

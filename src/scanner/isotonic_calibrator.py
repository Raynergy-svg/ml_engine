"""
Isotonic Regression Calibrator — Phase 51 (US-317).

Replaces failed Platt scaling (cal_error=0.36) with non-parametric
isotonic regression for mapping raw confidence → P(win).

Isotonic regression learns a piecewise-constant monotonic function
that corrects any systematic miscalibration without assuming a
parametric form (sigmoid). Pure Python implementation — no sklearn required.

Usage:
    cal = IsotonicCalibrator()
    result = cal.fit_from_journal("trained_data/trade_journal_rl.json")
    calibrated = cal.calibrate(0.72)  # Returns calibrated P(win)
"""

import json
import logging
import math
import os
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class IsotonicCalibrationResult:
    """Result of isotonic calibration fitting."""
    n_train: int = 0
    n_holdout: int = 0
    ece_before: float = 1.0       # Expected Calibration Error before
    ece_after: float = 1.0        # ECE after isotonic calibration
    n_breakpoints: int = 0        # Number of piecewise breakpoints
    is_valid: bool = False         # True if ECE improved and < 0.20
    holdout_win_rate: float = 0.0
    train_win_rate: float = 0.0


@dataclass
class IsotonicModel:
    """Persisted isotonic regression model."""
    breakpoints_x: List[float] = field(default_factory=list)  # Sorted confidence thresholds
    breakpoints_y: List[float] = field(default_factory=list)  # Corresponding P(win) values
    is_valid: bool = False
    ece: float = 1.0
    n_training_samples: int = 0
    version: int = 1

    def predict(self, confidence: float) -> float:
        """Map raw confidence → calibrated P(win) via piecewise linear interpolation."""
        if not self.breakpoints_x or not self.is_valid:
            return confidence  # Fallback: return raw

        x = self.breakpoints_x
        y = self.breakpoints_y

        # Clamp to range
        if confidence <= x[0]:
            return y[0]
        if confidence >= x[-1]:
            return y[-1]

        # Binary search for interpolation segment
        lo, hi = 0, len(x) - 1
        while lo < hi - 1:
            mid = (lo + hi) // 2
            if x[mid] <= confidence:
                lo = mid
            else:
                hi = mid

        # Linear interpolation between breakpoints
        if x[hi] == x[lo]:
            return y[lo]
        t = (confidence - x[lo]) / (x[hi] - x[lo])
        return y[lo] + t * (y[hi] - y[lo])


class IsotonicCalibrator:
    """Isotonic regression calibrator for confidence scores."""

    def __init__(self, n_bins: int = 15, min_samples_per_bin: int = 5):
        self.n_bins = n_bins
        self.min_samples_per_bin = min_samples_per_bin
        self.model: Optional[IsotonicModel] = None

    def fit_from_journal(
        self, journal_path: str, holdout_ratio: float = 0.20
    ) -> IsotonicCalibrationResult:
        """Load journal and fit isotonic regression.

        Args:
            journal_path: Path to trade_journal_rl.json
            holdout_ratio: Fraction held out for validation (0.20 = 20%)

        Returns:
            IsotonicCalibrationResult with fit diagnostics.
        """
        pairs = self._extract_confidence_outcome_pairs(journal_path)
        if len(pairs) < 30:
            logger.warning("Phase 51 US-317: Only %d pairs — need 30+ for isotonic fit", len(pairs))
            return IsotonicCalibrationResult(n_train=len(pairs))

        # Split train/holdout (chronological — last N% is holdout)
        split_idx = int(len(pairs) * (1.0 - holdout_ratio))
        train = pairs[:split_idx]
        holdout = pairs[split_idx:]

        # Measure ECE before calibration
        ece_before = self._compute_ece(
            [p[0] for p in holdout], [p[1] for p in holdout]
        )

        # Fit isotonic regression on training data
        model = self._fit_isotonic(train)

        # Measure ECE after calibration on holdout
        calibrated_holdout = [model.predict(p[0]) for p in holdout]
        ece_after = self._compute_ece(
            calibrated_holdout, [p[1] for p in holdout]
        )

        is_valid = ece_after < 0.20 and ece_after < ece_before
        model.is_valid = is_valid
        model.ece = ece_after

        self.model = model

        result = IsotonicCalibrationResult(
            n_train=len(train),
            n_holdout=len(holdout),
            ece_before=round(ece_before, 4),
            ece_after=round(ece_after, 4),
            n_breakpoints=len(model.breakpoints_x),
            is_valid=is_valid,
            holdout_win_rate=round(
                sum(1 for p in holdout if p[1]) / len(holdout) if holdout else 0, 4
            ),
            train_win_rate=round(
                sum(1 for p in train if p[1]) / len(train) if train else 0, 4
            ),
        )

        logger.info(
            "Phase 51 US-317: Isotonic fit — ECE %.4f→%.4f (%d breakpoints, valid=%s)",
            ece_before, ece_after, len(model.breakpoints_x), is_valid,
        )
        return result

    def calibrate(self, raw_confidence: float) -> float:
        """Calibrate a raw confidence score.

        Returns calibrated P(win) if model is valid, else raw confidence.
        """
        if self.model is None or not self.model.is_valid:
            return raw_confidence
        calibrated = self.model.predict(raw_confidence)
        return max(0.0, min(1.0, calibrated))

    def save_model(self, path: str) -> bool:
        """Save fitted model atomically."""
        if self.model is None:
            return False
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            data = asdict(self.model)
            tmp = path + ".tmp"
            with open(tmp, "w") as f:
                json.dump(data, f, indent=2, sort_keys=True)
            os.rename(tmp, path)
            logger.info("Phase 51: Saved isotonic model (%d breakpoints) to %s",
                        len(self.model.breakpoints_x), path)
            return True
        except Exception as e:
            logger.error("Phase 51: Failed to save isotonic model: %s", e)
            return False

    def load_model(self, path: str) -> bool:
        """Load fitted model from file."""
        try:
            with open(path, "r") as f:
                data = json.load(f)
            self.model = IsotonicModel(
                breakpoints_x=data.get("breakpoints_x", []),
                breakpoints_y=data.get("breakpoints_y", []),
                is_valid=data.get("is_valid", False),
                ece=data.get("ece", 1.0),
                n_training_samples=data.get("n_training_samples", 0),
                version=data.get("version", 1),
            )
            logger.info("Phase 51: Loaded isotonic model (%d breakpoints, valid=%s)",
                        len(self.model.breakpoints_x), self.model.is_valid)
            return True
        except (FileNotFoundError, json.JSONDecodeError, TypeError) as e:
            logger.debug("Phase 51: No isotonic model at %s: %s", path, e)
            return False

    # ── Private methods ──────────────────────────────────────────────

    def _extract_confidence_outcome_pairs(
        self, journal_path: str
    ) -> List[Tuple[float, bool]]:
        """Extract (confidence, won) pairs from journal."""
        try:
            with open(journal_path, "r") as f:
                trades = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.error("Phase 51: Failed to load journal: %s", e)
            return []

        if not isinstance(trades, list):
            return []

        pairs = []
        for t in trades:
            if not isinstance(t, dict):
                continue

            # Extract confidence
            conf = t.get("confidence") or t.get("entry_confidence") or 0.0
            if not isinstance(conf, (int, float)):
                continue
            conf = float(conf)
            if conf <= 0 or conf > 1.0:
                continue

            # Extract outcome
            outcome = t.get("outcome", {})
            if isinstance(outcome, dict):
                won = bool(outcome.get("trade_won", False))
            elif isinstance(outcome, str):
                won = outcome.lower() == "win"
            else:
                continue

            pairs.append((conf, won))

        return pairs

    def _fit_isotonic(self, pairs: List[Tuple[float, bool]]) -> IsotonicModel:
        """Fit isotonic regression using Pool Adjacent Violators Algorithm (PAVA).

        PAVA ensures the calibration function is monotonically non-decreasing:
        higher confidence should map to higher (or equal) P(win).
        """
        if not pairs:
            return IsotonicModel()

        # Sort by confidence
        sorted_pairs = sorted(pairs, key=lambda p: p[0])

        # Bin into groups for stability
        n = len(sorted_pairs)
        bin_size = max(self.min_samples_per_bin, n // self.n_bins)
        bins = []
        for i in range(0, n, bin_size):
            chunk = sorted_pairs[i:i + bin_size]
            avg_conf = sum(p[0] for p in chunk) / len(chunk)
            win_rate = sum(1 for p in chunk if p[1]) / len(chunk)
            bins.append((avg_conf, win_rate, len(chunk)))

        if not bins:
            return IsotonicModel()

        # PAVA: enforce monotonicity
        # Start with bin win rates, then merge adjacent violators
        x_vals = [b[0] for b in bins]
        y_vals = [b[1] for b in bins]
        weights = [b[2] for b in bins]

        y_iso = self._pava(y_vals, weights)

        model = IsotonicModel(
            breakpoints_x=x_vals,
            breakpoints_y=[round(y, 6) for y in y_iso],
            is_valid=False,  # Set by caller after ECE check
            n_training_samples=n,
        )
        return model

    @staticmethod
    def _pava(values: List[float], weights: List[float]) -> List[float]:
        """Pool Adjacent Violators Algorithm.

        Ensures output is monotonically non-decreasing.
        Uses weighted averaging when merging adjacent violators.
        """
        n = len(values)
        if n == 0:
            return []

        # Work on blocks: each block has (weighted_sum, weight_sum)
        blocks = [(values[i] * weights[i], weights[i]) for i in range(n)]
        result_blocks = [blocks[0]]

        for i in range(1, n):
            result_blocks.append(blocks[i])
            # Merge backward while monotonicity is violated
            while len(result_blocks) > 1:
                last_avg = result_blocks[-1][0] / result_blocks[-1][1]
                prev_avg = result_blocks[-2][0] / result_blocks[-2][1]
                if prev_avg <= last_avg:
                    break  # Monotonic — OK
                # Merge: combine last two blocks
                merged = (
                    result_blocks[-2][0] + result_blocks[-1][0],
                    result_blocks[-2][1] + result_blocks[-1][1],
                )
                result_blocks.pop()
                result_blocks.pop()
                result_blocks.append(merged)

        # Expand blocks back to per-bin values
        output = []
        block_idx = 0
        consumed = 0
        for i in range(n):
            block_sum, block_weight = result_blocks[block_idx]
            output.append(block_sum / block_weight)
            consumed += weights[i]
            if consumed >= result_blocks[block_idx][1] - 1e-9:
                block_idx += 1
                consumed = 0
                if block_idx >= len(result_blocks):
                    # Fill remaining with last value
                    while len(output) < n:
                        output.append(output[-1])
                    break

        return output[:n]

    def _compute_ece(
        self, predicted: List[float], actual: List[bool], n_bins: int = 10
    ) -> float:
        """Compute Expected Calibration Error.

        ECE = Σ (|bin| / N) * |accuracy(bin) - confidence(bin)|
        """
        if not predicted or not actual:
            return 1.0

        n = len(predicted)
        bin_edges = [i / n_bins for i in range(n_bins + 1)]
        ece = 0.0

        for b in range(n_bins):
            lo, hi = bin_edges[b], bin_edges[b + 1]
            mask = [(lo <= p < hi) if b < n_bins - 1 else (lo <= p <= hi)
                    for p in predicted]

            bin_count = sum(mask)
            if bin_count == 0:
                continue

            bin_acc = sum(1 for i, m in enumerate(mask) if m and actual[i]) / bin_count
            bin_conf = sum(predicted[i] for i, m in enumerate(mask) if m) / bin_count
            ece += (bin_count / n) * abs(bin_acc - bin_conf)

        return round(ece, 6)


def create_default_isotonic_calibrator() -> IsotonicCalibrator:
    """Create an IsotonicCalibrator with default config."""
    return IsotonicCalibrator(n_bins=15, min_samples_per_bin=5)

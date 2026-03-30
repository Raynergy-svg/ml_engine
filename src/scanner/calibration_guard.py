"""Calibration Guard — Phase 57 US-353.

Suppresses the overconfidence penalty (-0.03 on 0-1 scale) when the ModelCalibrationTracker
does not have sufficient data to produce a statistically valid signal.

Root cause: with only 6 total trades across 10 calibration bins
(MIN_SAMPLES_PER_BIN=5), valid_bins=0. The overconfidence_ratio is computed
as None or from 0 valid bins. Yet engine.py checks:
    if _oc_ratio is not None and _oc_ratio > 0.15:
        confidence = max(confidence - 3.0, 0.0)

When there are too few trades, _oc_ratio = None (guard passes), but the
computation may still return a non-None value from a single under-populated
bin. The guard adds an additional validity layer: n_predictions must be
sufficient for the calibration signal to be trusted.

Usage:
    guard = CalibrationGuard()
    calibration_report = tracker.get_calibration_report("TCN")
    if guard.should_apply_penalty(calibration_report):
        confidence = max(confidence - 3.0, 0.0)
    else:
        logger.debug("Overconfidence penalty suppressed: %s", guard.get_status(calibration_report))
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Minimum number of recorded predictions for calibration to be trusted
MIN_VALID_PREDICTIONS = 20

# Minimum number of bins with >= MIN_SAMPLES_PER_BIN entries
MIN_VALID_BINS = 2


class CalibrationGuard:
    """Guards the overconfidence penalty against statistically insufficient calibration data.

    The ModelCalibrationTracker requires MIN_SAMPLES_PER_BIN=5 per bin to
    consider a bin valid. With 10 bins and only a few trades, valid_bins is
    often 0 or 1. This guard requires both a minimum prediction count AND
    a minimum number of valid bins before the penalty is permitted.

    Usage:
        guard = CalibrationGuard()
        if guard.should_apply_penalty(calibration_report):
            confidence -= 3.0
    """

    def __init__(
        self,
        min_valid_predictions: int = MIN_VALID_PREDICTIONS,
        min_valid_bins: int = MIN_VALID_BINS,
    ):
        self.min_valid_predictions = int(min_valid_predictions)
        self.min_valid_bins = int(min_valid_bins)

        # Monitoring
        self._total_checks: int = 0
        self._suppression_count: int = 0

    # ── Public API ─────────────────────────────────────────────────────────

    def is_valid(self, calibration_report: Dict[str, Any]) -> bool:
        """Return True when the calibration report has sufficient statistical backing.

        Args:
            calibration_report: Dict from ModelCalibrationTracker.get_calibration_report().
                Expected keys: n_predictions (int), valid_bins (int),
                overconfidence_ratio (float|None).

        Returns:
            True if the calibration data is statistically sufficient.
        """
        if not isinstance(calibration_report, dict):
            return False

        n_predictions = int(calibration_report.get("n_predictions", 0))
        valid_bins = int(calibration_report.get("valid_bins", 0))
        overconfidence_ratio = calibration_report.get("overconfidence_ratio")

        # Primary guard: must have enough predictions
        if n_predictions < self.min_valid_predictions:
            return False

        # Secondary guard: must have enough valid bins
        if valid_bins < self.min_valid_bins:
            return False

        # Tertiary guard: ratio must not be None (no valid bins → None)
        if overconfidence_ratio is None:
            return False

        return True

    def should_apply_penalty(self, calibration_report: Dict[str, Any]) -> bool:
        """Return True when the overconfidence penalty is safe to apply.

        Only returns True when is_valid() is True AND the overconfidence_ratio
        exceeds the 0.15 threshold that engine.py uses.

        Args:
            calibration_report: Dict from ModelCalibrationTracker.get_calibration_report().

        Returns:
            True when penalty should apply (valid data AND overconfidence detected).
        """
        self._total_checks += 1

        if not self.is_valid(calibration_report):
            self._suppression_count += 1
            return False

        overconfidence_ratio = calibration_report.get("overconfidence_ratio", 0.0)
        return float(overconfidence_ratio) > 0.15

    def get_status(self, calibration_report: Dict[str, Any]) -> Dict[str, Any]:
        """Return a diagnostic dict explaining the guard's decision.

        Args:
            calibration_report: Dict from ModelCalibrationTracker.get_calibration_report().

        Returns:
            Dict with: is_valid, n_predictions, valid_bins, suppression_reason, will_apply_penalty.
        """
        if not isinstance(calibration_report, dict):
            return {
                "is_valid": False,
                "n_predictions": 0,
                "valid_bins": 0,
                "overconfidence_ratio": None,
                "suppression_reason": "invalid_report_format",
                "will_apply_penalty": False,
            }

        n_predictions = int(calibration_report.get("n_predictions", 0))
        valid_bins = int(calibration_report.get("valid_bins", 0))
        overconfidence_ratio = calibration_report.get("overconfidence_ratio")

        suppression_reason: Optional[str] = None
        if n_predictions < self.min_valid_predictions:
            suppression_reason = f"insufficient_predictions({n_predictions}<{self.min_valid_predictions})"
        elif valid_bins < self.min_valid_bins:
            suppression_reason = f"insufficient_valid_bins({valid_bins}<{self.min_valid_bins})"
        elif overconfidence_ratio is None:
            suppression_reason = "overconfidence_ratio_is_none"

        is_valid_flag = suppression_reason is None
        will_apply = is_valid_flag and float(overconfidence_ratio or 0.0) > 0.15

        return {
            "is_valid": is_valid_flag,
            "n_predictions": n_predictions,
            "valid_bins": valid_bins,
            "overconfidence_ratio": overconfidence_ratio,
            "suppression_reason": suppression_reason,
            "will_apply_penalty": will_apply,
        }

    def get_stats(self) -> Dict[str, Any]:
        """Return guard statistics."""
        suppression_rate = (
            self._suppression_count / self._total_checks
            if self._total_checks > 0 else 0.0
        )
        return {
            "total_checks": self._total_checks,
            "suppression_count": self._suppression_count,
            "suppression_rate": round(suppression_rate, 4),
            "min_valid_predictions": self.min_valid_predictions,
            "min_valid_bins": self.min_valid_bins,
        }

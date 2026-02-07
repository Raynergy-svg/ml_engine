"""Tier-2 calibration functions for confidence probability mapping.

This module provides functions for calibrating model confidence scores
to actual win probabilities using temperature scaling and bin interpolation.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def get_calibration_dict(meta: dict[str, Any]) -> dict[str, Any] | None:
    """Extract calibration dictionary from model metadata.

    Args:
        meta: Model metadata dictionary.

    Returns:
        Calibration dictionary if found, None otherwise.
    """
    tier2 = meta.get("tier2") or {}
    calib = (
        tier2.get("calibration")
        or tier2.get("calibration_v1")
        or meta.get("tier2_calibration")
    )
    return calib if isinstance(calib, dict) else None


def points_from_bins(calib: dict[str, Any]) -> list[tuple[float, float]]:
    """Extract calibration points from bin configuration.

    Args:
        calib: Calibration dictionary containing bins.

    Returns:
        List of (score, p_win) tuples sorted by score.
    """
    bins = calib.get("bins")
    if not isinstance(bins, list) or not bins:
        return []

    pts: list[tuple[float, float]] = []
    for b in bins:
        if not isinstance(b, dict):
            continue
        x = b.get("score")
        y = b.get("p_win")
        if x is None or y is None:
            continue
        try:
            xf = float(x)
            yf = float(y)
        except (ValueError, TypeError):
            continue
        if 0.0 <= xf <= 1.0:
            pts.append((xf, float(max(0.0, min(1.0, yf)))))
    pts.sort(key=lambda t: t[0])
    return pts


def interpolate_points(
    pts: list[tuple[float, float]], score: float
) -> float | None:
    """Interpolate calibrated probability from calibration points.

    Uses binary search and linear interpolation between points.

    Args:
        pts: Sorted list of (score, p_win) calibration points.
        score: Raw model score in [0, 1].

    Returns:
        Interpolated probability, or None if no points available.
    """
    if not pts:
        return None

    s = float(max(0.0, min(1.0, float(score))))
    x_min, y_min = pts[0]
    x_max, y_max = pts[-1]

    if s <= x_min:
        return float(y_min)
    if s >= x_max:
        return float(y_max)

    # Binary search for interpolation interval
    lo = 0
    hi = len(pts) - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if s <= pts[mid][0]:
            hi = mid
        else:
            lo = mid

    x0, y0 = pts[lo]
    x1, y1 = pts[hi]
    if x1 <= x0 + 1e-12:
        return float(y1)
    t = (s - x0) / (x1 - x0)
    return float((1.0 - t) * y0 + t * y1)


def clip_prob(p: float, *, eps: float = 1e-7) -> float:
    """Clip probability to valid range (eps, 1 - eps).

    Args:
        p: Input probability.
        eps: Minimum distance from 0 and 1.

    Returns:
        Clipped probability, or 0.5 if input is invalid.
    """
    try:
        pf = float(p)
    except (ValueError, TypeError):
        return 0.5
    if np.isnan(pf):
        return 0.5
    return float(max(eps, min(1.0 - eps, pf)))


def logit(p: float, *, eps: float = 1e-7) -> float:
    """Compute logit (log-odds) of probability.

    Args:
        p: Probability in (0, 1).
        eps: Clipping epsilon.

    Returns:
        Log-odds value.
    """
    pp = clip_prob(p, eps=eps)
    return float(np.log(pp / (1.0 - pp)))


def sigmoid(x: float) -> float:
    """Numerically stable sigmoid function.

    Args:
        x: Input value.

    Returns:
        Sigmoid(x) in (0, 1).
    """
    xf = float(x)
    if xf >= 0:
        ex = float(np.exp(-xf))
        return float(1.0 / (1.0 + ex))
    ex = float(np.exp(xf))
    return float(ex / (1.0 + ex))


def temperature_scale_prob(
    p: float, temperature: float, *, eps: float = 1e-7
) -> float:
    """Apply temperature scaling to probability.

    Args:
        p: Input probability.
        temperature: Temperature parameter (T > 1 smooths, T < 1 sharpens).
        eps: Clipping epsilon.

    Returns:
        Temperature-scaled probability.
    """
    try:
        temperature_f = float(temperature)
    except (ValueError, TypeError):
        return float(max(0.0, min(1.0, float(p))))
    if not np.isfinite(temperature_f) or temperature_f <= 0:
        return float(max(0.0, min(1.0, float(p))))
    z = logit(p, eps=eps)
    return float(sigmoid(z / temperature_f))


def negative_log_likelihood(
    y_true: np.ndarray, p_pred: np.ndarray, *, eps: float = 1e-15
) -> float:
    """Compute negative log-likelihood (binary cross-entropy).

    Args:
        y_true: True binary labels.
        p_pred: Predicted probabilities.
        eps: Clipping epsilon to avoid log(0).

    Returns:
        Mean NLL value.
    """
    p = np.clip(p_pred.astype(np.float64, copy=False), eps, 1.0 - eps)
    y = y_true.astype(np.float64, copy=False)
    return float(-np.mean((y * np.log(p)) + ((1.0 - y) * np.log(1.0 - p))))


def spearman_correlation(a: np.ndarray, b: np.ndarray) -> float:
    """Compute Spearman rank correlation coefficient.

    Args:
        a: First array.
        b: Second array.

    Returns:
        Spearman correlation, or NaN if computation fails.
    """
    aa = np.asarray(a, dtype=np.float64).reshape(-1)
    bb = np.asarray(b, dtype=np.float64).reshape(-1)
    if aa.size == 0 or aa.size != bb.size:
        return float("nan")
    ra = aa.argsort(kind="mergesort").argsort(kind="mergesort").astype(np.float64)
    rb = bb.argsort(kind="mergesort").argsort(kind="mergesort").astype(np.float64)
    try:
        return float(np.corrcoef(ra, rb)[0, 1])
    except (ValueError, IndexError):
        return float("nan")


def apply_calibration(meta: dict[str, Any], score: float) -> float | None:
    """Map a model score in [0,1] to calibrated P(win).

    Prefers temperature scaling when available; falls back to saved bin interpolation.

    Args:
        meta: Model metadata containing calibration info.
        score: Raw model score in [0, 1].

    Returns:
        Calibrated probability, or None if calibration unavailable.
    """
    try:
        calib = get_calibration_dict(meta)
        if calib is None:
            return None

        # Primary: temperature scaling
        ts = calib.get("temperature_scaling") if isinstance(calib, dict) else None
        if isinstance(ts, dict) and bool(ts.get("enabled", False)):
            T = ts.get("T")
            if T is not None:
                return temperature_scale_prob(float(score), float(T))

        # Fallback: bin interpolation
        pts = points_from_bins(calib)
        return interpolate_points(pts, score)
    except (KeyError, TypeError, ValueError) as e:
        logger.debug(f"Calibration application failed: {e}")
        return None


# Aliases for backward compatibility with main.py naming convention
_tier2_get_calibration_dict = get_calibration_dict
_tier2_points_from_bins = points_from_bins
_tier2_interpolate_points = interpolate_points
_tier2_clip_prob = clip_prob
_tier2_logit = logit
_tier2_sigmoid = sigmoid
_tier2_temperature_scale_prob = temperature_scale_prob
_tier2_nll = negative_log_likelihood
_tier2_spearman = spearman_correlation
_tier2_apply_calibration = apply_calibration

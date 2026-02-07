#!/usr/bin/env python3
"""Tier-2 probability calibration utilities for ML Engine."""
from __future__ import annotations

from typing import Any
import numpy as np


def _tier2_get_calibration_dict(meta: dict[str, Any]) -> dict[str, Any] | None:
    tier2 = meta.get("tier2") or {}
    calib = tier2.get("calibration") or tier2.get("calibration_v1") or meta.get("tier2_calibration")
    return calib if isinstance(calib, dict) else None


def _tier2_points_from_bins(calib: dict[str, Any]) -> list[tuple[float, float]]:
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
        except Exception:
            continue
        if 0.0 <= xf <= 1.0:
            pts.append((xf, float(max(0.0, min(1.0, yf)))))
    pts.sort(key=lambda t: t[0])
    return pts


def _tier2_interpolate_points(pts: list[tuple[float, float]], score: float) -> float | None:
    if not pts:
        return None

    s = float(max(0.0, min(1.0, float(score))))
    x_min, y_min = pts[0]
    x_max, y_max = pts[-1]

    if s <= x_min:
        return float(y_min)
    if s >= x_max:
        return float(y_max)

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


def _tier2_clip_prob(p: float, *, eps: float = 1e-7) -> float:
    try:
        pf = float(p)
    except Exception:
        return 0.5
    if np.isnan(pf):
        return 0.5
    return float(max(eps, min(1.0 - eps, pf)))


def _tier2_logit(p: float, *, eps: float = 1e-7) -> float:
    pp = _tier2_clip_prob(p, eps=eps)
    return float(np.log(pp / (1.0 - pp)))


def _tier2_sigmoid(x: float) -> float:
    xf = float(x)
    if xf >= 0:
        ex = float(np.exp(-xf))
        return float(1.0 / (1.0 + ex))
    ex = float(np.exp(xf))
    return float(ex / (1.0 + ex))


def _tier2_temperature_scale_prob(p: float, temperature: float, *, eps: float = 1e-7) -> float:
    try:
        temperature_f = float(temperature)
    except Exception:
        return float(max(0.0, min(1.0, float(p))))
    if not np.isfinite(temperature_f) or temperature_f <= 0:
        return float(max(0.0, min(1.0, float(p))))
    z = _tier2_logit(p, eps=eps)
    return float(_tier2_sigmoid(z / temperature_f))


def _tier2_nll(y_true: np.ndarray, p_pred: np.ndarray, *, eps: float = 1e-15) -> float:
    p = np.clip(p_pred.astype(np.float64, copy=False), eps, 1.0 - eps)
    y = y_true.astype(np.float64, copy=False)
    return float(-np.mean((y * np.log(p)) + ((1.0 - y) * np.log(1.0 - p))))


def _tier2_spearman(a: np.ndarray, b: np.ndarray) -> float:
    # Simple Spearman via rank correlation (stable enough for diagnostics).
    aa = np.asarray(a, dtype=np.float64).reshape(-1)
    bb = np.asarray(b, dtype=np.float64).reshape(-1)
    if aa.size == 0 or aa.size != bb.size:
        return float("nan")
    ra = aa.argsort(kind="mergesort").argsort(kind="mergesort").astype(np.float64)
    rb = bb.argsort(kind="mergesort").argsort(kind="mergesort").astype(np.float64)
    try:
        return float(np.corrcoef(ra, rb)[0, 1])
    except Exception:
        return float("nan")


def _tier2_apply_calibration(meta: dict[str, Any], score: float) -> float | None:
    """Map a model score in [0,1] to calibrated P(win).

    Prefers temperature scaling when available; falls back to saved bin interpolation.
    """
    try:
        calib = _tier2_get_calibration_dict(meta)
        if calib is None:
            return None

        # Primary: temperature scaling
        ts = calib.get("temperature_scaling") if isinstance(calib, dict) else None
        if isinstance(ts, dict) and bool(ts.get("enabled", False)):
            T = ts.get("T")
            if T is not None:
                return _tier2_temperature_scale_prob(float(score), float(T))

        # Fallback: bin interpolation
        pts = _tier2_points_from_bins(calib)
        return _tier2_interpolate_points(pts, score)
    except Exception:
        return None

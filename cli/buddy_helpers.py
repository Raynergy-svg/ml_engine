#!/usr/bin/env python3
"""Buddy helper functions for tier-2 calibration and trade execution utilities.

This module contains:
- Tier-2 calibration functions for probability mapping
- Buddy prediction output configuration
- FX execution guards and auto-close scheduling
- Model building utilities
"""
from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
from rich.console import Console

from cli.io_utils import BUDDY_META_FILENAME

console = Console()


# =============================================================================
# TIER-2 CALIBRATION HELPERS
# =============================================================================


def _tier2_get_calibration_dict(meta: dict[str, Any]) -> dict[str, Any] | None:
    """Extract calibration dictionary from model metadata."""
    tier2 = meta.get("tier2") or {}
    calib = tier2.get("calibration") or tier2.get("calibration_v1") or meta.get("tier2_calibration")
    return calib if isinstance(calib, dict) else None


def _tier2_points_from_bins(calib: dict[str, Any]) -> list[tuple[float, float]]:
    """Extract (score, p_win) points from calibration bins."""
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
    """Linearly interpolate between calibration points."""
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
    """Clip probability to (eps, 1-eps)."""
    try:
        pf = float(p)
    except Exception:
        return 0.5
    if np.isnan(pf):
        return 0.5
    return float(max(eps, min(1.0 - eps, pf)))


def _tier2_logit(p: float, *, eps: float = 1e-7) -> float:
    """Convert probability to logit."""
    pp = _tier2_clip_prob(p, eps=eps)
    return float(np.log(pp / (1.0 - pp)))


def _tier2_sigmoid(x: float) -> float:
    """Numerically stable sigmoid function."""
    xf = float(x)
    if xf >= 0:
        ex = float(np.exp(-xf))
        return float(1.0 / (1.0 + ex))
    ex = float(np.exp(xf))
    return float(ex / (1.0 + ex))


def _tier2_temperature_scale_prob(p: float, temperature: float, *, eps: float = 1e-7) -> float:
    """Apply temperature scaling to a probability."""
    try:
        temperature_f = float(temperature)
    except Exception:
        return float(max(0.0, min(1.0, float(p))))
    if not np.isfinite(temperature_f) or temperature_f <= 0:
        return float(max(0.0, min(1.0, float(p))))
    z = _tier2_logit(p, eps=eps)
    return float(_tier2_sigmoid(z / temperature_f))


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


# =============================================================================
# BUDDY HELPER FUNCTIONS
# =============================================================================


def _configure_predict_output(verbose: bool) -> None:
    """Reduce log/warning noise for interactive predict runs."""
    if verbose:
        return

    import warnings

    # Keep common third-party warnings from drowning out the prediction output.
    warnings.filterwarnings(
        "ignore",
        message=r"urllib3 v2 only supports OpenSSL 1\.1\.1\+",
    )
    try:
        import pandas as pd  # noqa: F401
        from pandas.errors import PerformanceWarning

        warnings.filterwarnings("ignore", category=PerformanceWarning)
    except Exception:
        pass

    # Keep CLI output clean by muting INFO logs from internal modules.
    logging.getLogger().setLevel(logging.WARNING)
    for name in (
        "utils",
        "reasoning_enhanced",
    ):
        logging.getLogger(name).setLevel(logging.WARNING)


def _buddy_live_enabled_from_meta() -> tuple[bool, float]:
    """Return (live_enabled, confidence_threshold) from last Buddy training."""
    meta_path = Path("trained_data") / "models" / BUDDY_META_FILENAME
    if not meta_path.exists():
        return False, 0.65
    try:
        meta = json.loads(meta_path.read_text())
        return bool(meta.get("live_enabled", False)), float(meta.get("live_confidence_threshold", 0.65))
    except Exception:
        return False, 0.65


def _fx_execution_guard_price_bound(_policy: Any, client: Any, *, instrument: str, units: int) -> float | None:
    """Compute price bound for execution guard."""
    from src.utils.fx_paper import pip_size

    try:
        q = client.get_price_quote(instrument=instrument)
    except Exception as e:
        console.print(f"[bold red]Blocked[/bold red]: could not fetch live quote: {e}")
        return None

    bid = float(q["bid"])
    ask = float(q["ask"])
    # Determine buffer in pips. Prefer policy-level override if available.
    buffer_pips = None
    try:
        buffer_pips = float(getattr(_policy.costs, "price_bound_buffer_pips"))
    except Exception:
        try:
            buffer_pips = float((_policy.get("costs") or {}).get("price_bound_buffer_pips", None))
        except Exception:
            buffer_pips = None

    # Default to a small buffer (1 pip) suitable for scalping.
    if buffer_pips is None:
        buffer_pips = 1.0

    buffer_price = float(buffer_pips) * pip_size(instrument)

    if int(units) > 0:
        return float(ask + buffer_price)
    return float(bid - buffer_price)


def _schedule_auto_close(client: Any, instrument: str, delay_s: float, *, verbose: bool = False) -> None:
    """Spawned in a daemon thread to close the instrument position after delay_s seconds.

    This is a best-effort helper for PRACTICE mode to ensure scalping-style trades
    are not left open beyond the desired timeframe.
    """
    def _worker():
        try:
            if verbose:
                console.print(f"[dim]Auto-close thread[/dim]: sleeping {delay_s:.1f}s before closing {instrument}")
            time.sleep(max(0.0, float(delay_s)))
            try:
                if hasattr(client, "close_trade") and hasattr(client, "_last_trade_id") and getattr(client, "_last_trade_id"):
                    tid = getattr(client, "_last_trade_id")
                    try:
                        res = client.close_trade(trade_id=tid)
                        console.print(f"[dim]Auto-close[/dim]: closed trade {tid} for {instrument}: {res}")
                    except Exception:
                        res = client.close_position(instrument=instrument)
                        console.print(f"[dim]Auto-close[/dim]: fallback closed position for {instrument}: {res}")
                else:
                    res = client.close_position(instrument=instrument)
                    console.print(f"[dim]Auto-close[/dim]: closed position for {instrument}: {res}")
            except Exception as e:
                console.print(f"[yellow]Auto-close failed[/yellow]: could not close {instrument}: {e}")
        except Exception:
            return

    t = threading.Thread(target=_worker, daemon=True, name=f"auto-close-{instrument}")
    t.start()


def _build_buddy_model_for_type(
    model_type: str,
    feature_dim: int,
    seq_len: int,
) -> Any:
    """Rebuild a Buddy model architecture from metadata."""
    from cli.tf_config import _configure_tf_metal
    _configure_tf_metal(verbose=False)
    import tensorflow as tf  # noqa: F401
    
    if model_type.lower() == "tcn":
        from src.models.tensorflow_models import build_tcn_direction_model
        return build_tcn_direction_model(
            seq_len=seq_len,
            feature_dim=feature_dim,
        )
    else:
        # Default to transformer
        from src.models.tensorflow_models import build_transformer_direction_model
        return build_transformer_direction_model(
            seq_len=seq_len,
            feature_dim=feature_dim,
        )

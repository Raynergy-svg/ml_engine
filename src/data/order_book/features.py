"""Numerical feature extraction from OANDA order/position book buckets.

Inputs are OANDA's raw bucket arrays:

    [
      {"price": "1.0500", "longCountPercent": "0.5", "shortCountPercent": "0.3"},
      {"price": "1.0501", "longCountPercent": "0.7", "shortCountPercent": "0.4"},
      ...
    ]

All percent fields are strings in OANDA's response — coerce defensively.
Each function returns a dict of float features; the caller decides whether
to feed them into the ``order_flow`` agent's heuristics or persist them for
future direction-transformer training.

Distance bands (``near``/``mid``/``far``) use percent of current price so the
same thresholds work across pairs with different absolute price scales.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List


_NEAR_BAND = 0.005   # 0.5% — ~50 pips for majors
_MID_BAND = 0.015    # 1.5% — ~150 pips
_TOP_K = 5           # bucket concentration window


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _band_sums(
    buckets: Iterable[Dict[str, Any]],
    current_price: float,
) -> Dict[str, float]:
    """Sum long/short counts in ``near``, ``mid``, ``far`` price bands.

    Bands are nested: ``near`` ⊂ ``mid`` ⊂ ``far`` (where ``far`` is all).
    """
    near_long = near_short = 0.0
    mid_long = mid_short = 0.0
    all_long = all_short = 0.0
    n_buckets = 0
    for b in buckets:
        bp = _safe_float(b.get("price"))
        if bp <= 0:
            continue
        n_buckets += 1
        lc = _safe_float(b.get("longCountPercent"))
        sc = _safe_float(b.get("shortCountPercent"))
        all_long += lc
        all_short += sc
        if current_price > 0:
            dist = abs(bp - current_price) / current_price
            if dist < _MID_BAND:
                mid_long += lc
                mid_short += sc
                if dist < _NEAR_BAND:
                    near_long += lc
                    near_short += sc
    return {
        "near_long": near_long,
        "near_short": near_short,
        "mid_long": mid_long,
        "mid_short": mid_short,
        "total_long": all_long,
        "total_short": all_short,
        "n_buckets": float(n_buckets),
    }


def _imbalance(long_pct: float, short_pct: float) -> float:
    """Normalized long-vs-short imbalance in [-1, 1]; 0 = balanced."""
    denom = long_pct + short_pct
    if denom <= 0:
        return 0.0
    return (long_pct - short_pct) / denom


def _top_k_concentration(
    buckets: Iterable[Dict[str, Any]],
    side: str,
    k: int = _TOP_K,
) -> float:
    """Fraction of total long/short concentrated in the top-``k`` buckets."""
    key = "longCountPercent" if side == "long" else "shortCountPercent"
    values: List[float] = []
    total = 0.0
    for b in buckets:
        v = _safe_float(b.get(key))
        if v <= 0:
            continue
        values.append(v)
        total += v
    if total <= 0:
        return 0.0
    values.sort(reverse=True)
    return sum(values[:k]) / total


def _common_features(
    buckets: List[Dict[str, Any]],
    current_price: float,
) -> Dict[str, float]:
    sums = _band_sums(buckets, current_price)
    near_imb = _imbalance(sums["near_long"], sums["near_short"])
    mid_imb = _imbalance(sums["mid_long"], sums["mid_short"])
    total_imb = _imbalance(sums["total_long"], sums["total_short"])
    long_ratio = (
        sums["total_long"] / (sums["total_long"] + sums["total_short"])
        if (sums["total_long"] + sums["total_short"]) > 0
        else 0.5
    )
    return {
        "long_ratio": long_ratio,
        "near_imbalance": near_imb,
        "mid_imbalance": mid_imb,
        "total_imbalance": total_imb,
        "near_long_pct": sums["near_long"],
        "near_short_pct": sums["near_short"],
        "top5_long_concentration": _top_k_concentration(buckets, "long"),
        "top5_short_concentration": _top_k_concentration(buckets, "short"),
        "n_buckets": sums["n_buckets"],
    }


def extract_position_features(
    buckets: List[Dict[str, Any]],
    current_price: float,
) -> Dict[str, float]:
    """Features for the position book (open positions).

    Crowded-trade indicators: ``long_ratio > 0.70`` or ``< 0.30`` are textbook
    contrarian signals — these features expose them numerically rather than as
    thresholded booleans.
    """
    feats = _common_features(buckets, current_price)
    return {f"pb_{k}": v for k, v in feats.items()}


def extract_order_features(
    buckets: List[Dict[str, Any]],
    current_price: float,
) -> Dict[str, float]:
    """Features for the order book (pending orders).

    ``near_imbalance`` captures the resistance/support pressure at the price
    levels closest to current — directly maps to the "heavy sell orders nearby"
    pattern the existing agent looks for, but as a graded signal.
    """
    feats = _common_features(buckets, current_price)
    return {f"ob_{k}": v for k, v in feats.items()}

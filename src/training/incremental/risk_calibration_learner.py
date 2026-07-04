"""Lightweight, single-pass (River-style) online learner for the risk /
execution / calibration layer.

Scope discipline (operator directive, 2026-07-04): this learner predicts
realized win-probability from RISK/EXECUTION features only, to recalibrate
confidence and derive a risk-decreasing-only position-size multiplier. It
must NEVER be given directional/price features (trend, momentum, price
levels) — this repo's empirical ceiling work established that price-only
intraday direction has no edge (docs/fx-edge-search-final-verdict-2026-06-18.md);
re-fitting a directional signal from journal outcomes would only overfit
noise, not add an edge. ``ALLOWED_FEATURE_KEYS`` is the enforced whitelist.

No new dependency: this hand-rolls the River `learn_one` / `predict_one`
idiom (constant-size state, O(1) update per sample, no full retrain) rather
than adding the `river` package, to avoid new supply-chain surface for a
research/infra-only task. The algorithm shape — not the specific library —
is what the operator asked to prefer over heavy deep-RL/FinRL.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# Whitelisted, risk/execution/calibration-only feature keys. Anything not in
# this set is (or risks being) a directional/price signal and MUST NOT be
# added here — see module docstring.
ALLOWED_FEATURE_KEYS = frozenset({
    "confidence",
    "regime_normal",
    "regime_low",
    "regime_high",
    "regime_extreme",
    "rr_ratio",
    "sl_pips",
    "tp_pips",
    "mae_pips",
    "mfe_pips",
    "lane_scanner",
    "lane_trend",
})


def build_features(entry: Dict[str, Any]) -> Dict[str, float]:
    """Extract the risk/calibration-only feature vector from a journal
    entry. Never raises; missing fields default to neutral values."""
    regime_field = entry.get("regime")
    regime = regime_field.get("volatility_regime") if isinstance(regime_field, dict) else None
    if not regime and isinstance(entry.get("regime_at_entry"), str):
        regime = entry.get("regime_at_entry")
    regime = str(regime or "NORMAL").upper()

    outcome = entry.get("outcome")
    if isinstance(outcome, dict):
        mae = outcome.get("mae_pips", entry.get("mae_pips"))
        mfe = outcome.get("mfe_pips", entry.get("mfe_pips"))
        rr = outcome.get("rr_ratio", entry.get("rr_ratio"))
    else:
        mae = entry.get("mae_pips")
        mfe = entry.get("mfe_pips")
        rr = entry.get("rr_ratio")

    def _f(v: Any) -> float:
        try:
            return float(v) if v is not None else 0.0
        except (TypeError, ValueError):
            return 0.0

    lane = str(entry.get("lane") or "scanner").lower()

    feats = {
        "confidence": _f(entry.get("confidence", 0.5)) or 0.5,
        "regime_normal": 1.0 if regime == "NORMAL" else 0.0,
        "regime_low": 1.0 if regime == "LOW" else 0.0,
        "regime_high": 1.0 if regime == "HIGH" else 0.0,
        "regime_extreme": 1.0 if regime == "EXTREME" else 0.0,
        "rr_ratio": _f(rr),
        "sl_pips": _f(entry.get("sl_pips")),
        "tp_pips": _f(entry.get("tp_pips")),
        "mae_pips": _f(mae),
        "mfe_pips": _f(mfe),
        "lane_scanner": 0.0 if lane == "trend" else 1.0,
        "lane_trend": 1.0 if lane == "trend" else 0.0,
    }
    if not (set(feats) <= ALLOWED_FEATURE_KEYS):
        # Not an assert: this boundary must hold even under -O/PYTHONOPTIMIZE,
        # since it's the only enforcement of "never a directional feature".
        raise ValueError("feature leak outside risk/calibration whitelist")
    return feats


def _label(entry: Dict[str, Any]) -> Optional[float]:
    """1.0 = win, 0.0 = loss, None = outcome not resolved / not scoreable."""
    outcome = entry.get("outcome")
    if isinstance(outcome, dict):
        if "trade_won" not in outcome:
            return None
        return 1.0 if outcome.get("trade_won") else 0.0
    if isinstance(outcome, str):
        s = outcome.strip().lower()
        if s in ("win", "loss"):
            return 1.0 if s == "win" else 0.0
    return None


@dataclass
class OnlineLogisticRegression:
    """Hand-rolled single-pass online logistic regression (SGD)."""

    lr: float = 0.05
    l2: float = 1e-4
    weights: Dict[str, float] = field(default_factory=dict)
    bias: float = 0.0
    n_seen: int = 0

    def predict_proba_one(self, x: Dict[str, float]) -> float:
        z = self.bias + sum(self.weights.get(k, 0.0) * v for k, v in x.items())
        # Numerically stable sigmoid.
        if z >= 0:
            return 1.0 / (1.0 + math.exp(-z))
        ez = math.exp(z)
        return ez / (1.0 + ez)

    def learn_one(self, x: Dict[str, float], y: float) -> None:
        p = self.predict_proba_one(x)
        err = p - y
        for k, v in x.items():
            g = err * v + self.l2 * self.weights.get(k, 0.0)
            self.weights[k] = self.weights.get(k, 0.0) - self.lr * g
        self.bias -= self.lr * err
        self.n_seen += 1

    def to_state(self) -> Dict[str, Any]:
        return {
            "lr": self.lr,
            "l2": self.l2,
            "weights": dict(self.weights),
            "bias": self.bias,
            "n_seen": self.n_seen,
        }

    @classmethod
    def from_state(cls, state: Dict[str, Any]) -> "OnlineLogisticRegression":
        obj = cls(lr=state.get("lr", 0.05), l2=state.get("l2", 1e-4))
        obj.weights = dict(state.get("weights", {}))
        obj.bias = float(state.get("bias", 0.0))
        obj.n_seen = int(state.get("n_seen", 0))
        return obj


def brier_score(
    model: OnlineLogisticRegression, samples: List[Tuple[Dict[str, float], float]]
) -> Optional[float]:
    """Mean squared error between predicted win-probability and realized
    outcome on a held-out slice. Lower is better-calibrated; this is the
    walk-forward promotion-gate metric."""
    if not samples:
        return None
    errs = [(model.predict_proba_one(x) - y) ** 2 for x, y in samples]
    return sum(errs) / len(errs)


@dataclass
class RiskCalibrationLearner:
    """Predicts realized win-probability from risk/execution features only,
    to recalibrate confidence and derive a risk-decreasing-only size
    multiplier."""

    model: OnlineLogisticRegression = field(default_factory=OnlineLogisticRegression)

    def fit_incremental(self, entries: List[Dict[str, Any]]) -> int:
        n = 0
        for e in entries:
            y = _label(e)
            if y is None:
                continue
            x = build_features(e)
            self.model.learn_one(x, y)
            n += 1
        return n

    def size_multiplier(self, entry_features: Dict[str, float]) -> float:
        """Risk-decreasing-only multiplier in [0.5, 1.0]. Can only ask for
        LESS risk than the caller's existing sizing when calibration says
        raw confidence overstates realized win probability — never more."""
        p = self.model.predict_proba_one(entry_features)
        raw_conf = entry_features.get("confidence", 0.5)
        if raw_conf <= 0:
            return 1.0
        ratio = p / raw_conf
        if not math.isfinite(ratio):
            # Corrupted/NaN model state must de-risk to the floor, not rely
            # on min()/max() argument-order luck (NaN comparisons are always
            # False, so a differently-ordered clamp could silently return 1.0).
            return 0.5
        return max(0.5, min(1.0, ratio))

    def to_state(self) -> Dict[str, Any]:
        return {"model": self.model.to_state()}

    @classmethod
    def from_state(cls, state: Optional[Dict[str, Any]]) -> "RiskCalibrationLearner":
        if not state:
            return cls()
        return cls(model=OnlineLogisticRegression.from_state(state.get("model", {})))

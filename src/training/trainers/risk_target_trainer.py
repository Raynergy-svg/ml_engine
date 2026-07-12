"""RiskTargetTrainer — forward volatility (regression) + forward
drawdown-state (classification), the risk-target ML redirect.

New, decoupled model family (2026-07-08). Does NOT subclass or import
`LightGBMRiskTrainer`/`LightGBMMomentumTrainer` (`lightgbm_trainers.py`) —
those classes are wired into the LIVE FX direction ensemble's hot path
(`src/core/modular_inference.py`); this trainer must never share an import
edge with that path (see `tests/test_risk_target_no_hotpath_coupling.py`).
Uses the same `_create_lgbm_regressor`/`_create_lgbm_classifier` factory
helpers from `trainers/utils.py` (generic, not FX-specific) and the same
atomic-write discipline (`atomic_pickle_dump`/`atomic_json_dump`).

Outputs a RISK ESTIMATE (predicted forward volatility, predicted
stressed-state probability) for future sizing/gating consumption — never a
directional trade signal. `predict()` returns plain floats/probabilities,
no buy/sell/hold semantics anywhere in this module.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from src.training.trainers.base import BaseTrainer
from src.training.trainers.config import TrainerConfig
from src.training.trainers.utils import (
    atomic_json_dump,
    atomic_pickle_dump,
    _create_lgbm_classifier,
    _create_lgbm_regressor,
)
from src.training.risk_target_features import RISK_TARGET_FEATURE_PIPELINE_VERSION

logger = logging.getLogger(__name__)

_EPS = 1e-12
# Floor applied to the annualized-vol TARGET before taking log() for the
# regression head (never to the model's output/prediction). An annualized
# realized vol below this on 20 forward FX daily bars is not physically
# plausible (would require ~zero price movement for a month) — the floor
# only guards against a pathological log(0) on a synthetic/degenerate
# input, it does not change any real training row's value.
_MIN_REALISTIC_ANNUALIZED_VOL = 1e-4
# Safety cap on the exponentiated prediction (500% annualized vol) — guards
# against a pathological extrapolation blowing up downstream QLIKE/pinball
# computation; real FX daily vol never approaches this.
_MAX_PLAUSIBLE_ANNUALIZED_VOL = 5.0


def qlike_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """QLIKE variance-forecast loss: mean(a/p - log(a/p) - 1), lower=better.

    Standard vol-forecast evaluation loss (asymmetrically penalizes
    under-prediction of variance more than over-prediction). Operates on
    variance (squared vol); callers passing volatility (not variance) get
    an equivalent monotone-consistent loss since both numerator forms use
    the same ratio.
    """
    a = np.clip(np.asarray(y_true, dtype=np.float64), _EPS, None)
    p = np.clip(np.asarray(y_pred, dtype=np.float64), _EPS, None)
    ratio = a / p
    return float(np.mean(ratio - np.log(ratio) - 1.0))


def pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, quantile: float = 0.5) -> float:
    """Pinball (quantile) loss at the given quantile, lower=better."""
    a = np.asarray(y_true, dtype=np.float64)
    p = np.asarray(y_pred, dtype=np.float64)
    diff = a - p
    return float(np.mean(np.maximum(quantile * diff, (quantile - 1.0) * diff)))


def brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Brier score (mean squared error of a probability forecast), lower=better."""
    y = np.asarray(y_true, dtype=np.float64)
    p = np.asarray(y_prob, dtype=np.float64)
    return float(np.mean((p - y) ** 2))


# Same threshold/formula as the drawdown-state learnability bar
# pre-registered in docs/prereg-risk-target-vol-drawdown-2026-07-08.md:
# "OOS AUC >= 0.55 AND OOS Brier beats the base-rate baseline" (baseline
# Brier = p(1-p) at the observed base rate). IMPORTANT: applied here to
# whatever validation split the caller passes to `train()` — this is a
# per-retrain PROXY check using the same formula, not a re-derivation of
# the pre-reg doc's one-time frozen verdict (which was computed on the
# untouched 2024-01-01->2026-06-11 OOS test slice by
# scripts/experiment_risk_target_vol_drawdown.py, a different split). Both
# currently agree the head is not learnable, but they are not the same
# measurement and could diverge on a future retrain. This is an ABSOLUTE
# bar (independent of any incumbent) — the relative-regression gate in
# cli/risk_target_training.py alone cannot catch a head that never clears
# this bar, since a candidate that merely doesn't regress vs. a similarly-
# bad incumbent (or vs. no incumbent at all, on first deploy) would still
# pass a regression-only
# check. See tests/test_risk_target_training_gate.py.
DRAWDOWN_LEARNABLE_MIN_AUC = 0.55


def brier_baseline(y_true: np.ndarray) -> float:
    """Base-rate Brier score p(1-p) — the naive "always predict the base
    rate" baseline the drawdown head's pre-registered bar must beat.

    On a degenerate one-class `y_true` this returns exactly 0.0 (p=0 or
    p=1), which a strict `<` comparison can never beat — but that case is
    moot in practice: `_safe_auc` already returns 0.5 (< the 0.55 bar) on a
    one-class holdout, so `drawdown_learnable` is refused by the AUC guard
    before this baseline's edge value is ever load-bearing.
    """
    y = np.asarray(y_true, dtype=np.float64)
    p = float(np.mean(y)) if len(y) else 0.0
    return p * (1.0 - p)


def _safe_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """AUC-ROC without a hard sklearn dependency at import time; returns
    0.5 (uninformative) if the holdout has only one class present."""
    y = np.asarray(y_true, dtype=np.int64)
    if len(np.unique(y)) < 2:
        return 0.5
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, y_prob))


class RiskTargetTrainer(BaseTrainer):
    """Dual-head risk-target trainer: forward-vol regressor + forward
    drawdown-state classifier. Two independent LightGBM models (not a
    joint architecture) — each target is evaluated and gated on its own
    honest OOS metric.
    """

    def __init__(self, config: Optional[TrainerConfig] = None):
        super().__init__(config)
        self.vol_model = None
        self.drawdown_model = None
        self.feature_names: Optional[List[str]] = None
        self.categorical_features: Optional[List[str]] = None

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        x_val: np.ndarray,
        y_val: np.ndarray,
        feature_names: Optional[List[str]] = None,
        instrument: str = "",
        fold_id: Optional[int] = None,
        **kwargs: Any,
    ) -> Dict[str, float]:
        """Train both heads.

        `y_train`/`y_val` are the forward-vol REGRESSION target (per the
        `BaseTrainer` single-target signature). The drawdown-state
        CLASSIFICATION target is passed via kwargs (`y_train_drawdown`,
        `y_val_drawdown`) — both required.
        """
        y_train_dd = kwargs.get("y_train_drawdown")
        y_val_dd = kwargs.get("y_val_drawdown")
        if y_train_dd is None or y_val_dd is None:
            raise ValueError(
                "RiskTargetTrainer.train requires y_train_drawdown/y_val_drawdown kwargs"
            )
        categorical_features = kwargs.get("categorical_features")

        self.feature_names = list(feature_names) if feature_names else None
        self.categorical_features = list(categorical_features) if categorical_features else []

        n_estimators = int(kwargs.get("n_estimators", 500))
        max_depth = int(kwargs.get("max_depth", 6))
        learning_rate = float(kwargs.get("learning_rate", 0.03))
        subsample = float(kwargs.get("subsample", 0.8))
        colsample_bytree = float(kwargs.get("colsample_bytree", 0.8))
        early_stopping_rounds = int(kwargs.get("early_stopping_rounds", 50))

        import lightgbm as lgb

        # --- Vol regression head — fit in LOG-space ---
        # Realized volatility is strictly positive and (empirically, and by
        # the log-normal working assumption standard in the vol-forecasting
        # literature — Andersen & Bollerslev 1998) closer to log-normally
        # than normally distributed. An unconstrained regression on the raw
        # level can and does predict non-positive values on low-vol rows
        # (verified empirically on this exact panel: ~7.6% of OOS
        # predictions were <=0 on a raw-level fit), which makes QLIKE
        # (which divides by the prediction) explode. Fitting on log(vol)
        # and exponentiating the prediction back enforces positivity by
        # construction and is the standard practice for this reason, not a
        # post-hoc metric patch — see
        # docs/prereg-risk-target-vol-drawdown-2026-07-08.md for the
        # disclosed correction note.
        y_train_log = np.log(np.clip(y_train, _MIN_REALISTIC_ANNUALIZED_VOL, None))
        y_val_log = np.log(np.clip(y_val, _MIN_REALISTIC_ANNUALIZED_VOL, None))

        self.vol_model = _create_lgbm_regressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            objective="regression",
        )
        self.vol_model.fit(
            X_train, y_train_log,
            eval_set=[(x_val, y_val_log)],
            callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False)],
        )
        vol_pred_val = self._predict_vol(x_val)

        # --- Drawdown-state classification head ---
        self.drawdown_model = _create_lgbm_classifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
        )
        self.drawdown_model.fit(
            X_train, y_train_dd,
            eval_set=[(x_val, y_val_dd)],
            callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False)],
        )
        dd_prob_val = self.drawdown_model.predict_proba(x_val)[:, 1]

        val_auc_drawdown = _safe_auc(y_val_dd, dd_prob_val)
        val_brier_drawdown = brier_score(y_val_dd, dd_prob_val)
        val_brier_drawdown_baseline = brier_baseline(y_val_dd)
        drawdown_learnable = bool(
            val_auc_drawdown >= DRAWDOWN_LEARNABLE_MIN_AUC
            and val_brier_drawdown < val_brier_drawdown_baseline
        )
        self.metrics = {
            "val_qlike": qlike_loss(y_val, vol_pred_val),
            "val_pinball_median": pinball_loss(y_val, vol_pred_val, quantile=0.5),
            "val_r2": float(1.0 - np.sum((y_val - vol_pred_val) ** 2) / max(np.sum((y_val - np.mean(y_val)) ** 2), _EPS)),
            "val_mae": float(np.mean(np.abs(y_val - vol_pred_val))),
            "val_auc_drawdown": val_auc_drawdown,
            "val_brier_drawdown": val_brier_drawdown,
            "val_brier_drawdown_baseline": val_brier_drawdown_baseline,
            "drawdown_learnable": drawdown_learnable,
            "n_train": int(len(X_train)),
            "n_val": int(len(x_val)),
            "instrument": instrument,
            "fold_id": fold_id,
        }
        self.is_trained = True
        return self.metrics

    def _predict_vol(self, X: np.ndarray) -> np.ndarray:
        """Exponentiate the log-space regressor's output back to
        annualized-vol units, with a plausibility cap (see module-level
        `_MAX_PLAUSIBLE_ANNUALIZED_VOL` docstring note)."""
        log_pred = self.vol_model.predict(X)
        return np.clip(np.exp(log_pred), _MIN_REALISTIC_ANNUALIZED_VOL, _MAX_PLAUSIBLE_ANNUALIZED_VOL)

    def predict(self, X: np.ndarray) -> Dict[str, Any]:
        if not self.is_trained or self.vol_model is None or self.drawdown_model is None:
            raise RuntimeError("RiskTargetTrainer.predict called before train()/load()")
        vol_pred = self._predict_vol(X)
        dd_prob = self.drawdown_model.predict_proba(X)[:, 1]
        return {
            "predicted_forward_volatility": vol_pred,
            "predicted_drawdown_stressed_prob": dd_prob,
        }

    def save(self, path: str) -> None:
        if not self.is_trained:
            raise RuntimeError("RiskTargetTrainer.save called before train()")
        base = Path(path)
        atomic_pickle_dump(
            {"vol_model": self.vol_model, "drawdown_model": self.drawdown_model},
            base.with_suffix(".pkl"),
        )
        meta = {
            "feature_names": self.feature_names,
            "categorical_features": self.categorical_features,
            "risk_target_feature_pipeline_version": RISK_TARGET_FEATURE_PIPELINE_VERSION,
            "scaler": "not_applicable_tree_based_model",
            "metrics": self.metrics,
        }
        atomic_json_dump(meta, base.with_suffix(".meta.json"), indent=2, sort_keys=True, default=str)

    def load(self, path: str) -> None:
        # Pickle is used here only to deserialize our OWN LightGBM model
        # artifacts, written exclusively by this module's save() through the
        # gated retrain path (cli/risk_target_training.py) — never loading
        # third-party or user-supplied data. Same convention as every other
        # trainer in src/training/trainers/*.py (LightGBM/sklearn objects
        # have no JSON-native serialization).
        import json
        import pickle

        base = Path(path)
        with open(base.with_suffix(".pkl"), "rb") as f:
            payload = pickle.load(f)
        self.vol_model = payload["vol_model"]
        self.drawdown_model = payload["drawdown_model"]
        with open(base.with_suffix(".meta.json")) as f:
            meta = json.load(f)
        self.feature_names = meta.get("feature_names")
        self.categorical_features = meta.get("categorical_features")
        self.metrics = meta.get("metrics", {})
        self.is_trained = True


__all__ = [
    "RiskTargetTrainer",
    "qlike_loss",
    "pinball_loss",
    "brier_score",
]

"""Read-only consumption interface for the risk-target model artifact.

This is the documented seam through which sizing/gating code MAY, in a
future operator-approved change, consume the risk-target model's outputs —
analogous to `src/hedge/portfolio_exposure.py`'s read-only report pattern.
It is NOT wired into any live path today (no `ScannerConfig` field, no
caller in `src/scanner/`, no execution/gates import — enforced by
`tests/test_risk_target_no_hotpath_coupling.py`). Wiring it into live
sizing is an explicit ESCALATION-class decision, not something this module
does implicitly.

Contract:
    - OUTPUT IS A RISK ESTIMATE ONLY. `predicted_forward_volatility` (an
      annualized vol forecast) and `predicted_drawdown_stressed_prob` (a
      probability the next-20-day max drawdown lands in the stressed top
      quartile). Neither is, or may ever be reinterpreted as, a
      directional trade signal — there is no direction/side/entry output.
    - Fail-closed: missing artifact, missing features, or a feature-
      pipeline version mismatch returns `{"available": False, "reason":
      ...}` — never a fabricated estimate.
    - The drawdown head's probability is currently NOT calibrated (its
      Brier score failed the pre-registered bar — see docs/prereg-risk-
      target-vol-drawdown-2026-07-08.md §6). `drawdown_head_trustworthy`
      is False in the output until a recalibrated candidate clears the
      bar; honest consumers should use the volatility output only.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from src.training.risk_target_features import (
    FEATURE_COLUMNS,
    RISK_TARGET_FEATURE_PIPELINE_VERSION,
    compute_risk_target_features,
)

logger = logging.getLogger(__name__)

DEFAULT_MODEL_PATH = Path("trained_data/risk_targets/models/risk_target_model")

# Honest-consumption flag: the drawdown head's OOS Brier failed the
# pre-registered bar on 2026-07-08 (AUC 0.625 passed, Brier 0.232 vs
# baseline 0.143 failed). Flip only when a retrained candidate clears BOTH
# bar conditions through the gated path (cli/risk_target_training.py).
DRAWDOWN_HEAD_TRUSTWORTHY = False


def predict_risk_state(
    ohlcv_df: pd.DataFrame,
    pair: str,
    *,
    model_path: Path = DEFAULT_MODEL_PATH,
) -> Dict[str, Any]:
    """Compute the current risk estimate for one instrument, read-only.

    Args:
        ohlcv_df: daily OHLCV history for the instrument (columns: date,
            open, high, low, close, volume; ascending; needs >= 61 rows
            for the 60-bar trailing warm-up).
        pair: instrument name (must be a pair the model was trained on —
            it is a categorical feature).
        model_path: artifact base path (no suffix).

    Returns:
        On success: {"available": True, "pair": ...,
            "predicted_forward_volatility_annualized": float,
            "predicted_drawdown_stressed_prob": float,
            "drawdown_head_trustworthy": bool,
            "asof_date": str, "model_meta": {...}}
        On any failure: {"available": False, "reason": str} — fail-closed,
        never a fabricated number.
    """
    meta_path = Path(model_path).with_suffix(".meta.json")
    if not meta_path.exists() or not Path(model_path).with_suffix(".pkl").exists():
        return {"available": False, "reason": f"no_artifact_at:{model_path}"}

    try:
        with open(meta_path) as f:
            meta = json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        return {"available": False, "reason": f"unreadable_meta:{exc}"}

    artifact_version = meta.get("risk_target_feature_pipeline_version")
    if artifact_version != RISK_TARGET_FEATURE_PIPELINE_VERSION:
        # Hard refuse on version mismatch — mirrors gates.py's fail-closed
        # contract for the direction pipeline (improvement.md).
        return {
            "available": False,
            "reason": (
                f"feature_pipeline_version_mismatch:artifact={artifact_version}"
                f"!=runtime={RISK_TARGET_FEATURE_PIPELINE_VERSION}"
            ),
        }

    try:
        feat = compute_risk_target_features(ohlcv_df)
    except ValueError as exc:
        return {"available": False, "reason": f"feature_computation_failed:{exc}"}

    last = feat.iloc[[-1]].copy()
    if last[FEATURE_COLUMNS].isna().any(axis=1).iloc[0]:
        return {"available": False, "reason": "insufficient_trailing_history_for_features"}

    last["pair"] = pd.Categorical([pair])
    last["day_of_week"] = last["day_of_week"].astype("category")

    from src.training.trainers.risk_target_trainer import RiskTargetTrainer
    trainer = RiskTargetTrainer()
    try:
        trainer.load(str(model_path))
    except Exception as exc:  # noqa: BLE001 — any load failure is a valid fail-closed outcome
        logger.warning("predict_risk_state: artifact load failed: %s", exc)
        return {"available": False, "reason": f"artifact_load_failed:{exc}"}

    expected = trainer.feature_names or []
    X = last[expected] if all(c in last.columns for c in expected) else None
    if X is None:
        return {"available": False, "reason": "feature_columns_missing_vs_artifact_contract"}

    pred = trainer.predict(X)
    return {
        "available": True,
        "pair": pair,
        "predicted_forward_volatility_annualized": float(pred["predicted_forward_volatility"][0]),
        "predicted_drawdown_stressed_prob": float(pred["predicted_drawdown_stressed_prob"][0]),
        "drawdown_head_trustworthy": DRAWDOWN_HEAD_TRUSTWORTHY,
        "asof_date": str(last["date"].iloc[0]),
        "model_meta": {
            "risk_target_feature_pipeline_version": artifact_version,
            "oos_metrics": meta.get("metrics", {}),
        },
    }


__all__ = ["predict_risk_state", "DEFAULT_MODEL_PATH", "DRAWDOWN_HEAD_TRUSTWORTHY"]

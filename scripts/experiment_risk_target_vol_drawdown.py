#!/usr/bin/env python3
"""Frozen risk-target experiment: forward volatility (regression) + forward
drawdown-state (classification).

Pre-registered BEFORE this script was run:
`docs/prereg-risk-target-vol-drawdown-2026-07-08.md`. One construction, one
feature set, one hyperparameter config — no sweep, no post-hoc bar changes.

Offline/research only: reads the cached daily FX panel at
`market_data/factor/*_D.csv` — no OANDA API call, no broker/execution
import, no halt/state touch.

Usage:
    python scripts/experiment_risk_target_vol_drawdown.py
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.training.labels import (  # noqa: E402
    compute_forward_realized_volatility,
    compute_forward_drawdown_state_labels,
    DRAWDOWN_NAN_SENTINEL,
)
from src.training.risk_target_features import (  # noqa: E402
    compute_risk_target_features,
    FEATURE_COLUMNS,
    RISK_TARGET_FEATURE_PIPELINE_VERSION,
)
from src.training.risk_target_splits import (  # noqa: E402
    compute_global_date_buckets,
    rows_in_bucket,
)
from src.training.trainers.risk_target_trainer import (  # noqa: E402
    RiskTargetTrainer,
    qlike_loss,
    pinball_loss,
    brier_score,
    _safe_auc,
)
from src.training.trainers.config import TrainerConfig  # noqa: E402
from src.training.mlflow_mirror import mirror_training_session  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

FACTOR_DIR = Path("market_data/factor")
PAIRS = [
    "AUD_JPY", "AUD_NZD", "AUD_USD", "CAD_JPY", "CHF_JPY", "EUR_AUD",
    "EUR_CAD", "EUR_CHF", "EUR_GBP", "EUR_JPY", "EUR_USD", "GBP_AUD",
    "GBP_JPY", "GBP_USD", "NZD_JPY", "NZD_USD", "USD_CAD", "USD_CHF", "USD_JPY",
]
HORIZON_BARS = 20
OOS_START = "2024-01-01"
VAL_FRAC = 0.2
STRESSED_QUANTILE = 0.75
ANNUALIZATION_FACTOR = float(np.sqrt(252.0))
MIN_WARMUP_BARS = 60  # longest rolling window in compute_risk_target_features

RESULTS_DIR = Path("trained_data/backtests")
MODEL_DIR = Path("trained_data/risk_targets/models")


def load_pair_ohlcv(pair: str) -> pd.DataFrame:
    path = FACTOR_DIR / f"{pair}_D.csv"
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"], utc=True)
    return df.sort_values("date").reset_index(drop=True)


def build_pair_dataset(pair: str, train_dates, val_dates, test_dates):
    """Return per-pair (X_df, y_vol, y_dd, split_name) rows, dropping
    warm-up and forward-window-tail rows. `split_name` is one of
    train/val/test/embargo per row (embargo rows are excluded downstream)."""
    raw = load_pair_ohlcv(pair)
    feat = compute_risk_target_features(raw)

    vol_values, vol_meta = compute_forward_realized_volatility(
        feat, HORIZON_BARS, annualization_factor=ANNUALIZATION_FACTOR,
    )

    train_idx = rows_in_bucket(feat["date"], train_dates)
    val_idx = rows_in_bucket(feat["date"], val_dates)
    test_idx = rows_in_bucket(feat["date"], test_dates)

    dd_labels, dd_meta = compute_forward_drawdown_state_labels(
        feat, HORIZON_BARS, train_idx=train_idx,
        stressed_quantile=STRESSED_QUANTILE, val_idx=val_idx, test_idx=test_idx,
    )

    features_ok = feat[FEATURE_COLUMNS].notna().all(axis=1).values
    vol_ok = np.isfinite(vol_values)
    dd_ok = dd_labels != DRAWDOWN_NAN_SENTINEL
    valid = features_ok & vol_ok & dd_ok

    split = pd.Series("embargo", index=feat.index, dtype=object)
    split.iloc[train_idx] = "train"
    split.iloc[val_idx] = "val"
    split.iloc[test_idx] = "test"

    feat = feat.loc[valid].copy()
    feat["pair"] = pair
    feat["split"] = split.loc[valid].values
    y_vol = vol_values[valid]
    y_dd = dd_labels[valid]

    return feat, y_vol, y_dd, {"vol_meta": vol_meta, "dd_meta": dd_meta}


def naive_persistence_baseline(df: pd.DataFrame) -> np.ndarray:
    """Naive vol-forecast baseline: predict tomorrow's forward-vol as
    today's trailing realized_vol_20 (persistence-in-vol)."""
    return df["realized_vol_20"].values


def base_rate_baseline(y_train_dd: np.ndarray) -> float:
    return float(np.mean(y_train_dd))


def main() -> dict:
    logger.info("Loading %d-pair daily FX panel and computing global date buckets...", len(PAIRS))
    all_dates = pd.concat([load_pair_ohlcv(p)["date"] for p in PAIRS], ignore_index=True)
    train_dates, val_dates, test_dates = compute_global_date_buckets(
        all_dates, oos_start=OOS_START, val_frac=VAL_FRAC, embargo_bars=HORIZON_BARS,
    )
    logger.info(
        "Date buckets: train=%d val=%d test=%d unique dates",
        len(train_dates), len(val_dates), len(test_dates),
    )

    per_pair_frames = []
    per_pair_y_vol = []
    per_pair_y_dd = []
    label_meta_by_pair = {}
    for pair in PAIRS:
        feat, y_vol, y_dd, meta = build_pair_dataset(pair, train_dates, val_dates, test_dates)
        per_pair_frames.append(feat)
        per_pair_y_vol.append(y_vol)
        per_pair_y_dd.append(y_dd)
        label_meta_by_pair[pair] = meta

    pooled = pd.concat(per_pair_frames, ignore_index=True)
    y_vol_all = np.concatenate(per_pair_y_vol)
    y_dd_all = np.concatenate(per_pair_y_dd)

    pooled["pair"] = pooled["pair"].astype("category")
    pooled["day_of_week"] = pooled["day_of_week"].astype("category")

    model_feature_cols = list(FEATURE_COLUMNS) + ["pair"]
    X_all = pooled[model_feature_cols]

    train_mask = (pooled["split"] == "train").values
    val_mask = (pooled["split"] == "val").values
    test_mask = (pooled["split"] == "test").values

    logger.info(
        "Pooled rows: train=%d val=%d test=%d (of %d total, embargo-dropped=%d)",
        train_mask.sum(), val_mask.sum(), test_mask.sum(), len(pooled),
        len(pooled) - train_mask.sum() - val_mask.sum() - test_mask.sum(),
    )

    trainer = RiskTargetTrainer(TrainerConfig())
    val_metrics = trainer.train(
        X_all[train_mask], y_vol_all[train_mask],
        X_all[val_mask], y_vol_all[val_mask],
        feature_names=model_feature_cols,
        instrument=f"pooled_{len(PAIRS)}_fx_pairs",
        y_train_drawdown=y_dd_all[train_mask],
        y_val_drawdown=y_dd_all[val_mask],
    )
    logger.info("Validation metrics: %s", val_metrics)

    # --- OOS evaluation (read ONCE, per pre-registration) ---
    X_test = X_all[test_mask]
    y_vol_test = y_vol_all[test_mask]
    y_dd_test = y_dd_all[test_mask]
    test_pred = trainer.predict(X_test)
    vol_pred_test = test_pred["predicted_forward_volatility"]
    dd_prob_test = test_pred["predicted_drawdown_stressed_prob"]

    model_qlike = qlike_loss(y_vol_test, vol_pred_test)
    model_pinball = pinball_loss(y_vol_test, vol_pred_test, quantile=0.5)
    model_r2 = float(1.0 - np.sum((y_vol_test - vol_pred_test) ** 2) / max(np.sum((y_vol_test - np.mean(y_vol_test)) ** 2), 1e-12))
    model_mae = float(np.mean(np.abs(y_vol_test - vol_pred_test)))

    naive_pred_test = naive_persistence_baseline(pooled.loc[test_mask])
    naive_qlike = qlike_loss(y_vol_test, naive_pred_test)
    naive_pinball = pinball_loss(y_vol_test, naive_pred_test, quantile=0.5)
    naive_r2 = float(1.0 - np.sum((y_vol_test - naive_pred_test) ** 2) / max(np.sum((y_vol_test - np.mean(y_vol_test)) ** 2), 1e-12))

    # --- Per-pair OOS breakdown (QA remediation 2026-07-08: pooled R² on a
    # cross-pair panel is structurally inflated by cross-pair scale
    # separation; the honest read requires within-pair numbers too) ---
    per_pair_oos: Dict[str, Dict[str, float]] = {}
    test_pairs = pooled.loc[test_mask, "pair"].values
    for p in PAIRS:
        pm = test_pairs == p
        if pm.sum() < 30:
            continue
        yv, pv, nv = y_vol_test[pm], vol_pred_test[pm], naive_pred_test[pm]
        denom = max(float(np.sum((yv - np.mean(yv)) ** 2)), 1e-12)
        per_pair_oos[p] = {
            "n": int(pm.sum()),
            "model_qlike": qlike_loss(yv, pv),
            "naive_qlike": qlike_loss(yv, nv),
            "model_r2": float(1.0 - np.sum((yv - pv) ** 2) / denom),
            "naive_r2": float(1.0 - np.sum((yv - nv) ** 2) / denom),
        }
    n_pairs_model_beats_naive_qlike = sum(
        1 for v in per_pair_oos.values() if v["model_qlike"] < v["naive_qlike"]
    )
    n_pairs_model_r2_positive = sum(1 for v in per_pair_oos.values() if v["model_r2"] > 0)

    model_auc = _safe_auc(y_dd_test, dd_prob_test)
    model_brier = brier_score(y_dd_test, dd_prob_test)
    train_base_rate = base_rate_baseline(y_dd_all[train_mask])
    baseline_prob_test = np.full_like(dd_prob_test, train_base_rate, dtype=np.float64)
    baseline_brier = brier_score(y_dd_test, baseline_prob_test)
    baseline_auc = 0.5

    vol_learnable = bool(model_qlike < naive_qlike and model_r2 > 0.0)
    dd_learnable = bool(model_auc >= 0.55 and model_brier < baseline_brier)

    results = {
        "run_at_utc": datetime.now(timezone.utc).isoformat(),
        "pre_registration": "docs/prereg-risk-target-vol-drawdown-2026-07-08.md",
        "universe": PAIRS,
        "horizon_bars": HORIZON_BARS,
        "oos_start": OOS_START,
        "risk_target_feature_pipeline_version": RISK_TARGET_FEATURE_PIPELINE_VERSION,
        "n_train": int(train_mask.sum()),
        "n_val": int(val_mask.sum()),
        "n_test": int(test_mask.sum()),
        "validation_metrics": val_metrics,
        "target_forward_volatility": {
            "metric": "QLIKE (lower=better) + pinball@median + R2 + MAE",
            "oos_model_qlike": model_qlike,
            "oos_naive_persistence_qlike": naive_qlike,
            "oos_model_pinball_median": model_pinball,
            "oos_naive_persistence_pinball_median": naive_pinball,
            "oos_model_r2": model_r2,
            "oos_naive_persistence_r2": naive_r2,
            "oos_model_mae": model_mae,
            "learnable_bar": "model_qlike < naive_qlike AND model_r2 > 0",
            "learnable": vol_learnable,
            "per_pair_oos": per_pair_oos,
            "n_pairs_model_beats_naive_qlike": n_pairs_model_beats_naive_qlike,
            "n_pairs_model_r2_positive": n_pairs_model_r2_positive,
            "n_pairs_evaluated": len(per_pair_oos),
        },
        "target_drawdown_state": {
            "metric": "AUC-ROC + Brier (lower=better)",
            "oos_model_auc": model_auc,
            "oos_baseline_auc": baseline_auc,
            "oos_model_brier": model_brier,
            "oos_baseline_brier": baseline_brier,
            "train_stressed_base_rate": train_base_rate,
            "learnable_bar": "model_auc >= 0.55 AND model_brier < baseline_brier",
            "learnable": dd_learnable,
        },
        "label_meta_by_pair": {p: m for p, m in label_meta_by_pair.items()},
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "risk_target_vol_drawdown_result.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info("Wrote results to %s", out_path)

    # Fail-open MLflow mirror (never raises; see src/training/mlflow_mirror.py docstring).
    mirror_training_session(
        {
            "pair": "pooled_18_fx_pairs",
            "timestamp": "risk_target_vol_drawdown_experiment",
            "granularity": "D",
            "warm_start": False,
            "duration_seconds": 0.0,
            "models_trained": ["risk_target_vol_regression", "risk_target_drawdown_classification"],
            "hyperparams": {"horizon_bars": HORIZON_BARS, "oos_start": OOS_START},
            "metrics": {
                "oos_vol_qlike": model_qlike,
                "oos_vol_r2": model_r2,
                "oos_drawdown_auc": model_auc,
                "oos_drawdown_brier": model_brier,
            },
        },
        experiment_name="risk_target_vol_drawdown",
    )

    if vol_learnable or dd_learnable:
        # Security-review finding (2026-07-08): never save() directly over a
        # possible incumbent — route through the gated retrain path so a
        # re-run cannot silently overwrite a better existing artifact.
        from cli.risk_target_training import train_risk_targets
        gate_result = train_risk_targets(
            X_all[train_mask], y_vol_all[train_mask], y_dd_all[train_mask],
            X_all[val_mask], y_vol_all[val_mask], y_dd_all[val_mask],
            feature_names=model_feature_cols,
            model_dir=MODEL_DIR,
        )
        results["artifact_gate"] = {
            "verdict": gate_result["verdict"],
            "written": gate_result["written"],
            "vol_gate": gate_result["vol_gate"],
            "drawdown_gate": gate_result["drawdown_gate"],
        }
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(
            "Artifact gate: %s (written=%s)", gate_result["verdict"], gate_result["written"],
        )
    else:
        logger.info("Neither target cleared its pre-registered bar — no artifact written.")

    return results


if __name__ == "__main__":
    result = main()
    print(json.dumps(
        {
            "vol_learnable": result["target_forward_volatility"]["learnable"],
            "dd_learnable": result["target_drawdown_state"]["learnable"],
            "oos_vol_qlike_model_vs_naive": (
                result["target_forward_volatility"]["oos_model_qlike"],
                result["target_forward_volatility"]["oos_naive_persistence_qlike"],
            ),
            "oos_dd_auc": result["target_drawdown_state"]["oos_model_auc"],
        },
        indent=2,
    ))

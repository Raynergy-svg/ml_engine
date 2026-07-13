"""Authority-free risk-target evaluation.

This module trains and evaluates the dual risk-target heads on a set of
already-verified dataset partitions and returns deterministic per-head
results. It imports only the (offline, hot-path-decoupled) trainer, the
feature/label/split helpers, and the evidence contracts — never the broker
path, the trade execution/gate path, or the state engine. It is handed its
data as bytes and has no repository-root or control-file parameter, so it can
neither read ``.claude/state.json`` nor reach a broker credential.

Determinism is deliberate: with fixed seeds, single-threaded LightGBM, and
``deterministic=True``, two runs on identical partitions produce identical
metrics — the property the local metric-replay verification depends on.
"""

from __future__ import annotations

import io
import pickle
from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd

from src.evidence.contracts import GateResult, GateStatus
from src.training.labels import (
    DRAWDOWN_NAN_SENTINEL,
    compute_forward_drawdown_state_labels,
    compute_forward_realized_volatility,
)
from src.training.risk_target_features import (
    FEATURE_COLUMNS,
    compute_risk_target_features,
)
from src.training.risk_target_splits import compute_global_date_buckets, rows_in_bucket
from src.training.trainers.config import TrainerConfig
from src.training.trainers.risk_target_trainer import (
    DRAWDOWN_LEARNABLE_MIN_AUC,
    RiskTargetTrainer,
    brier_score,
    pinball_loss,
    qlike_loss,
    _safe_auc,
)

from .models import (
    DRAWDOWN_HEAD_ID,
    VOLATILITY_HEAD_ID,
    HeadResult,
)

ANNUALIZATION_FACTOR = float(np.sqrt(252.0))


@dataclass(frozen=True)
class EvaluationParams:
    """Frozen evaluation configuration (mirrors the pre-registered run)."""

    horizon_bars: int = 20
    oos_start: str = "2024-01-01"
    val_frac: float = 0.2
    stressed_quantile: float = 0.75
    n_estimators: int = 300
    max_depth: int = 6
    learning_rate: float = 0.03
    early_stopping_rounds: int = 50
    seed: int = 20260708
    # Absolute reproduction tolerance the local importer replays against. With
    # deterministic single-thread LightGBM the delta is ~0; the tolerance is a
    # small guard against cross-run float noise, declared per-metric in the
    # EvaluationReport so verification is against the producer's stated bound.
    #
    # KNOWN LIMITATION (device-dependent determinism): LightGBM reproduces
    # bit-for-bit only for an identical device+params combination. The factory
    # helpers pick device/max_bin by platform/GPU availability, so a genuine
    # remote-GPU / local-CPU split could make an *honest* replay diverge beyond
    # this tolerance and REJECT. That is the safe direction (fail-closed —
    # never a false accept), but a real heterogeneous-hardware fabric would need
    # a device-pinned worker image or a looser, explicitly-declared tolerance.
    # Same-machine runs (today's only mode) reproduce within 1e-9. (F5.)
    replay_tolerance: float = 1e-9


@dataclass(frozen=True)
class _Frame:
    X: pd.DataFrame
    y_vol: np.ndarray
    y_dd: np.ndarray
    train_mask: np.ndarray
    val_mask: np.ndarray
    test_mask: np.ndarray
    naive_vol: np.ndarray
    feature_names: list[str]


def _read_partition(instrument: str, data: bytes) -> pd.DataFrame:
    df = pd.read_csv(io.BytesIO(data))
    required = {"date", "open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"partition {instrument!r} missing columns {sorted(missing)}")
    df["date"] = pd.to_datetime(df["date"], utc=True)
    return df.sort_values("date").reset_index(drop=True)


def assemble_frame(partitions: Mapping[str, bytes], params: EvaluationParams) -> _Frame:
    """Pool declared partitions into one leakage-safe train/val/test frame.

    Rolling features never cross a pair boundary; the split is on the shared
    calendar-date axis with an embargo gap, exactly as the pre-registered
    experiment (`scripts/experiment_risk_target_vol_drawdown.py`).
    """
    if not partitions:
        raise ValueError("no dataset partitions supplied to evaluator")

    raw_by_pair = {inst: _read_partition(inst, data) for inst, data in partitions.items()}
    all_dates = pd.concat([df["date"] for df in raw_by_pair.values()], ignore_index=True)
    train_dates, val_dates, test_dates = compute_global_date_buckets(
        all_dates, oos_start=params.oos_start, val_frac=params.val_frac,
        embargo_bars=params.horizon_bars,
    )

    frames: list[pd.DataFrame] = []
    y_vol_parts: list[np.ndarray] = []
    y_dd_parts: list[np.ndarray] = []
    for pair in sorted(raw_by_pair):
        feat = compute_risk_target_features(raw_by_pair[pair])
        vol_values, _ = compute_forward_realized_volatility(
            feat, params.horizon_bars, annualization_factor=ANNUALIZATION_FACTOR,
        )
        train_idx = rows_in_bucket(feat["date"], train_dates)
        val_idx = rows_in_bucket(feat["date"], val_dates)
        test_idx = rows_in_bucket(feat["date"], test_dates)
        dd_labels, _ = compute_forward_drawdown_state_labels(
            feat, params.horizon_bars, train_idx=train_idx,
            stressed_quantile=params.stressed_quantile, val_idx=val_idx, test_idx=test_idx,
        )
        features_ok = feat[FEATURE_COLUMNS].notna().all(axis=1).values
        vol_ok = np.isfinite(vol_values)
        dd_ok = dd_labels != DRAWDOWN_NAN_SENTINEL
        valid = features_ok & vol_ok & dd_ok

        split = pd.Series("embargo", index=feat.index, dtype=object)
        split.iloc[train_idx] = "train"
        split.iloc[val_idx] = "val"
        split.iloc[test_idx] = "test"

        kept = feat.loc[valid].copy()
        kept["pair"] = pair
        kept["split"] = split.loc[valid].values
        frames.append(kept)
        y_vol_parts.append(vol_values[valid])
        y_dd_parts.append(dd_labels[valid])

    pooled = pd.concat(frames, ignore_index=True)
    y_vol = np.concatenate(y_vol_parts)
    y_dd = np.concatenate(y_dd_parts).astype(np.int64)
    pooled["pair"] = pooled["pair"].astype("category")
    pooled["day_of_week"] = pooled["day_of_week"].astype("category")

    feature_names = list(FEATURE_COLUMNS) + ["pair"]
    return _Frame(
        X=pooled[feature_names],
        y_vol=y_vol,
        y_dd=y_dd,
        train_mask=(pooled["split"] == "train").values,
        val_mask=(pooled["split"] == "val").values,
        test_mask=(pooled["split"] == "test").values,
        naive_vol=pooled["realized_vol_20"].values,
        feature_names=feature_names,
    )


def _gate(gate_id: str, ok: bool, observed, threshold, reason: str) -> GateResult:
    return GateResult(
        gate_id=gate_id,
        status=GateStatus.PASS if ok else GateStatus.FAIL,
        observed=observed,
        threshold=threshold,
        reason=reason,
    )


def evaluate_frame(frame: _Frame, params: EvaluationParams) -> tuple[HeadResult, ...]:
    """Train both heads and return their independent, gated results.

    The OOS test bucket is the temporal holdout; the val bucket is used only
    for early stopping. Each head is scored and gated on its own honest OOS
    metric — the vol head can pass while the drawdown head fails, and neither
    verdict touches the other.
    """
    if not (frame.train_mask.any() and frame.val_mask.any() and frame.test_mask.any()):
        raise ValueError("evaluation requires non-empty train, val and test buckets")

    trainer = RiskTargetTrainer(TrainerConfig())
    trainer.train(
        frame.X[frame.train_mask], frame.y_vol[frame.train_mask],
        frame.X[frame.val_mask], frame.y_vol[frame.val_mask],
        feature_names=frame.feature_names,
        instrument="pooled_fx_pairs",
        y_train_drawdown=frame.y_dd[frame.train_mask],
        y_val_drawdown=frame.y_dd[frame.val_mask],
        n_estimators=params.n_estimators,
        max_depth=params.max_depth,
        learning_rate=params.learning_rate,
        early_stopping_rounds=params.early_stopping_rounds,
        # Determinism controls forwarded to LightGBM so replay reproduces.
        n_jobs=1,
        deterministic=True,
        force_row_wise=True,
        random_state=params.seed,
    )

    X_test = frame.X[frame.test_mask]
    y_vol_test = frame.y_vol[frame.test_mask]
    y_dd_test = frame.y_dd[frame.test_mask]
    preds = trainer.predict(X_test)
    vol_pred = preds["predicted_forward_volatility"]
    dd_prob = preds["predicted_drawdown_stressed_prob"]

    # --- Volatility head ---
    naive_pred = frame.naive_vol[frame.test_mask]
    model_qlike = qlike_loss(y_vol_test, vol_pred)
    naive_qlike = qlike_loss(y_vol_test, naive_pred)
    denom = max(float(np.sum((y_vol_test - np.mean(y_vol_test)) ** 2)), 1e-12)
    model_r2 = float(1.0 - np.sum((y_vol_test - vol_pred) ** 2) / denom)
    vol_metrics = {
        "oos_qlike": model_qlike,
        "oos_naive_qlike": naive_qlike,
        "oos_pinball_median": pinball_loss(y_vol_test, vol_pred, quantile=0.5),
        "oos_r2": model_r2,
        "oos_mae": float(np.mean(np.abs(y_vol_test - vol_pred))),
    }
    vol_gates = (
        _gate("oos_qlike_beats_naive", model_qlike < naive_qlike, model_qlike, naive_qlike,
              "OOS QLIKE must beat naive persistence"),
        _gate("oos_r2_positive", model_r2 > 0.0, model_r2, 0.0, "OOS R2 must be positive"),
    )
    vol_passed = all(g.status == GateStatus.PASS for g in vol_gates)

    # --- Drawdown-state head ---
    model_auc = _safe_auc(y_dd_test, dd_prob)
    model_brier = brier_score(y_dd_test, dd_prob)
    base_rate = float(np.mean(frame.y_dd[frame.train_mask]))
    baseline_brier = brier_score(y_dd_test, np.full_like(dd_prob, base_rate))
    dd_metrics = {
        "oos_auc": model_auc,
        "oos_brier": model_brier,
        "oos_baseline_brier": baseline_brier,
        "train_base_rate": base_rate,
    }
    dd_gates = (
        _gate("oos_auc_ge_bar", model_auc >= DRAWDOWN_LEARNABLE_MIN_AUC, model_auc,
              DRAWDOWN_LEARNABLE_MIN_AUC, "OOS AUC must clear the learnability bar"),
        _gate("oos_brier_beats_baseline", model_brier < baseline_brier, model_brier,
              baseline_brier, "OOS Brier must beat the base-rate baseline"),
    )
    dd_passed = all(g.status == GateStatus.PASS for g in dd_gates)

    # NOTE: model_bytes are pickled LightGBM objects. They are content-addressed
    # (sha256) and wrapped in a signed EvidencePackage, and this slice NEVER
    # unpickles them (verification re-derives metrics by retraining from CSV).
    # Any future consumer that loads these bytes MUST unpickle only AFTER
    # verifying the package signature + artifact digest against a trusted
    # producer key — pickle.loads executes arbitrary code. (Security review A.)
    n_test = int(frame.test_mask.sum())
    tol = {k: params.replay_tolerance for k in vol_metrics}
    dd_tol = {k: params.replay_tolerance for k in dd_metrics}
    holdout = f"OOS calendar bucket (dates >= {params.oos_start}), {params.horizon_bars}-bar embargo"

    vol_head = HeadResult(
        head_id=VOLATILITY_HEAD_ID,
        lane_id="risk_target_vol",
        target="forward_realized_volatility",
        metrics=vol_metrics,
        metric_tolerances=tol,
        gates=vol_gates,
        passed=vol_passed,
        model_bytes=pickle.dumps(trainer.vol_model, protocol=pickle.DEFAULT_PROTOCOL),
        media_type="application/octet-stream",
        trial_count=1,
        effective_sample_size=float(n_test),
        temporal_holdout=holdout,
        purge_observations=params.horizon_bars,
        embargo_observations=params.horizon_bars,
        incumbent_comparison={"has_incumbent": False},
    )
    dd_head = HeadResult(
        head_id=DRAWDOWN_HEAD_ID,
        lane_id="risk_target_drawdown",
        target="forward_drawdown_state",
        metrics=dd_metrics,
        metric_tolerances=dd_tol,
        gates=dd_gates,
        passed=dd_passed,
        model_bytes=pickle.dumps(trainer.drawdown_model, protocol=pickle.DEFAULT_PROTOCOL),
        media_type="application/octet-stream",
        trial_count=1,
        effective_sample_size=float(n_test),
        temporal_holdout=holdout,
        purge_observations=params.horizon_bars,
        embargo_observations=params.horizon_bars,
        incumbent_comparison={"has_incumbent": False},
    )
    return (vol_head, dd_head)


def evaluate_partitions(
    partitions: Mapping[str, bytes],
    params: EvaluationParams | None = None,
) -> tuple[HeadResult, ...]:
    """Assemble the declared partitions and evaluate both heads (deterministic)."""
    params = params or EvaluationParams()
    return evaluate_frame(assemble_frame(partitions, params), params)


__all__ = [
    "EvaluationParams",
    "assemble_frame",
    "evaluate_frame",
    "evaluate_partitions",
    "ANNUALIZATION_FACTOR",
]

"""
Phase 2.B inference contract tests (2026-05-08).

Tests gates.py inference-side fixes from
docs/superpowers/plans/2026-05-08-pipeline-reconciliation-phase1-audit.md:

1. `_build_transformer_inference_matrix` builds X in the exact order of
   feature_names (training-time ordering).
2. Regime one-hots are computed from saved quantiles applied to saved
   regime_atr_col — matching what `load_direction_data` would have produced
   at training time.
3. Missing required features cause refusal (returns None), not silent
   zero-fill.

End-to-end load+predict integration is verified at Phase 4 holdout (a real
trained artifact loaded by gates.py); these tests cover the deterministic
column-assembly logic without needing TF or a trained keras file.

No mocks — real DataFrames, real GateEvaluator instance with attrs populated
directly per the project's `.claude/rules/improvement.md` no-mock rule.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

from pathlib import Path

from src.scanner.gates import GateEvaluator


def _make_evaluator_with_contract(
    feature_names: list,
    regime_quantiles: dict,
    regime_atr_col: str = "atr_pct_20",
    pipeline_version: str = "2026-05-08-v1",
):
    """Construct a GateEvaluator and populate its inference contract attrs directly.

    Uses the real GateEvaluator class (no mocks); just bypasses model loading
    by setting attrs after __init__. This isolates the contract-application
    logic from disk-loading concerns covered separately.
    """
    ev = GateEvaluator(model_dir=Path("trained_data/models"), use_joint_only=True)
    ev._transformer = object()  # truthy non-None so evaluate path doesn't early-return
    ev._transformer_feature_names = feature_names
    ev._transformer_regime_quantiles = regime_quantiles
    ev._transformer_regime_atr_col = regime_atr_col
    ev._transformer_pipeline_version = pipeline_version
    return ev


def _norm_df(n_rows: int = 80) -> pd.DataFrame:
    """Synthetic normalized-feature DataFrame with all the columns we exercise."""
    rng = np.random.default_rng(3)
    return pd.DataFrame({
        "returns_5":   rng.normal(0.0, 1.0, n_rows),
        "atr_pct_20":  np.linspace(0.0001, 0.005, n_rows),  # spans regime quantiles
        "volume_zscore": rng.normal(0.0, 1.0, n_rows),
        "hour_sin":    np.sin(2 * np.pi * np.arange(n_rows) / 24),
        "hour_cos":    np.cos(2 * np.pi * np.arange(n_rows) / 24),
    })


def test_build_inference_matrix_respects_feature_names_order():
    """X must have columns in the exact order of feature_names, regardless of DF column order."""
    feature_names = ["volume_zscore", "returns_5", "hour_sin", "hour_cos"]
    ev = _make_evaluator_with_contract(feature_names, regime_quantiles=None)

    df = _norm_df(n_rows=80)
    X = ev._build_transformer_inference_matrix(df, feature_names, seq_len=60)

    assert X is not None
    assert X.shape == (60, 4)
    # Column 0 must be volume_zscore values (last 60 rows), not returns_5
    np.testing.assert_array_equal(X[:, 0], df["volume_zscore"].values[-60:].astype(np.float32))
    np.testing.assert_array_equal(X[:, 1], df["returns_5"].values[-60:].astype(np.float32))
    np.testing.assert_array_equal(X[:, 2], df["hour_sin"].values[-60:].astype(np.float32))
    np.testing.assert_array_equal(X[:, 3], df["hour_cos"].values[-60:].astype(np.float32))


def test_build_inference_matrix_computes_regime_one_hots_from_saved_quantiles():
    """regime_low/normal/high/extreme columns must reflect the quantile classification of atr_pct_20."""
    feature_names = [
        "returns_5", "atr_pct_20",
        "regime_low", "regime_normal", "regime_high", "regime_extreme",
    ]
    # Quantiles chosen such that atr=0.001 → LOW, 0.002 → NORMAL, 0.003 → HIGH, 0.004 → EXTREME
    regime_quantiles = {"q25": 0.0015, "q50": 0.0025, "q75": 0.0035}
    ev = _make_evaluator_with_contract(feature_names, regime_quantiles)

    # Build a DataFrame with deterministic atr values bracketing the quantiles
    n = 80
    atr = np.linspace(0.001, 0.004, n)  # spans all 4 regimes
    df = pd.DataFrame({
        "returns_5": np.zeros(n),
        "atr_pct_20": atr,
    })

    X = ev._build_transformer_inference_matrix(df, feature_names, seq_len=60)

    assert X is not None
    assert X.shape == (60, 6)

    # Check that for the last 60 rows, exactly one regime column is 1.0 per row
    regime_cols = X[:, 2:6]  # last 4 cols are regime one-hots
    one_hot_sums = regime_cols.sum(axis=1)
    np.testing.assert_array_equal(one_hot_sums, np.ones(60, dtype=np.float32))

    # Reconstruct expected classification for the last 60 atr values
    last_atr = atr[-60:]
    q25, q50, q75 = regime_quantiles["q25"], regime_quantiles["q50"], regime_quantiles["q75"]
    expected_cls = np.where(
        last_atr <= q25, 0,
        np.where(last_atr <= q50, 1,
                 np.where(last_atr <= q75, 2, 3)),
    )
    actual_cls = np.argmax(regime_cols, axis=1)
    np.testing.assert_array_equal(actual_cls, expected_cls)


def test_build_inference_matrix_refuses_missing_required_feature(caplog):
    """Missing required feature → returns None and logs once. NO silent zero-fill."""
    feature_names = ["returns_5", "this_does_not_exist"]
    ev = _make_evaluator_with_contract(feature_names, regime_quantiles=None)

    df = _norm_df(n_rows=80)

    with caplog.at_level(logging.WARNING, logger="src.scanner.gates"):
        X = ev._build_transformer_inference_matrix(df, feature_names, seq_len=60)

    assert X is None, "Missing required feature must return None, not zero-fill"
    assert any("contract gap" in r.message.lower() for r in caplog.records), (
        "Contract gap must be logged"
    )


def test_build_inference_matrix_refuses_regime_without_quantiles(caplog):
    """If regime_* requested but regime_quantiles absent, refuse — don't fabricate."""
    feature_names = ["returns_5", "regime_normal"]
    ev = _make_evaluator_with_contract(feature_names, regime_quantiles=None)

    df = _norm_df(n_rows=80)
    with caplog.at_level(logging.WARNING, logger="src.scanner.gates"):
        X = ev._build_transformer_inference_matrix(df, feature_names, seq_len=60)

    assert X is None
    assert any(
        "regime" in r.message.lower() and "missing" in r.message.lower()
        for r in caplog.records
    ), "Must explicitly log the regime-quantiles gap"


def test_build_inference_matrix_refuses_regime_without_atr_col(caplog):
    """If regime requested + quantiles available but atr_pct column missing from DF, refuse."""
    feature_names = ["regime_high"]
    quantiles = {"q25": 0.001, "q50": 0.002, "q75": 0.003}
    ev = _make_evaluator_with_contract(
        feature_names, regime_quantiles=quantiles, regime_atr_col="atr_pct_20"
    )

    # DF is missing atr_pct_20 entirely
    df = pd.DataFrame({"returns_5": np.zeros(80)})

    with caplog.at_level(logging.WARNING, logger="src.scanner.gates"):
        X = ev._build_transformer_inference_matrix(df, feature_names, seq_len=60)

    assert X is None
    assert any(
        "atr_pct_20" in r.message and "missing" in r.message.lower()
        for r in caplog.records
    ), "Must explicitly log which atr column is missing"


def test_build_inference_matrix_returns_none_on_insufficient_rows():
    """If DF has fewer rows than seq_len, return None (don't pad silently)."""
    feature_names = ["returns_5"]
    ev = _make_evaluator_with_contract(feature_names, regime_quantiles=None)

    df = _norm_df(n_rows=30)  # below seq_len=60
    X = ev._build_transformer_inference_matrix(df, feature_names, seq_len=60)
    assert X is None


def test_build_inference_matrix_handles_non_finite_atr_in_regime_classification():
    """Non-finite atr values fall through to NORMAL (loader's behavior at line 1934)."""
    feature_names = ["regime_normal"]
    quantiles = {"q25": 0.001, "q50": 0.002, "q75": 0.003}
    ev = _make_evaluator_with_contract(feature_names, regime_quantiles=quantiles)

    n = 80
    atr = np.full(n, np.nan)  # all non-finite
    df = pd.DataFrame({"atr_pct_20": atr})

    X = ev._build_transformer_inference_matrix(df, feature_names, seq_len=60)
    assert X is not None
    # All rows should be regime_normal=1 since NaN atr falls through to NORMAL
    np.testing.assert_array_equal(X[:, 0], np.ones(60, dtype=np.float32))

"""
Phase 2.A pipeline reconciliation tests (2026-05-08).

Tests the trainer-side fixes from
docs/superpowers/plans/2026-05-08-pipeline-reconciliation-phase1-audit.md:

1. `_subset_scaler` correctly narrows a fitted StandardScaler to selected indices.
2. `_assert_scaler_not_identity` flags the var_=1.0 identity-fingerprint pattern.
3. `load_direction_data` returns `regime_quantiles`, `regime_atr_col`, and
   `feature_pipeline_version` when ATR data is present.

End-to-end save→reload integration is verified at Phase 3 retrain (a real
training run with disk inspection); these unit tests cover the deterministic
logic that doesn't require a TF session.

No mocks — uses real sklearn StandardScaler and real DataFrames per the
project's `.claude/rules/improvement.md` no-mock rule.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

from src.core.modular_data_loaders import (
    FEATURE_PIPELINE_VERSION,
    compute_normalized_features,
    load_direction_data,
)


# ---------------------------------------------------------------------------
# 1. _subset_scaler: scaler subset preserves per-column standardization stats
# ---------------------------------------------------------------------------

def _make_real_scaler(n_rows: int = 5000, n_cols: int = 12) -> StandardScaler:
    """Return a StandardScaler fit on raw data with mixed per-column scales."""
    rng = np.random.default_rng(42)
    cols = [
        rng.normal(0.001, 0.0005, n_rows),    # tiny returns
        rng.normal(0.005, 0.002, n_rows),     # ATR pct
        rng.normal(0.0, 1.5, n_rows),         # already-ish standardized
        rng.uniform(0.0, 1.0, n_rows),        # 0-1 range
    ]
    while len(cols) < n_cols:
        cols.append(rng.normal(rng.uniform(-3, 3), rng.uniform(0.1, 8), n_rows))
    raw = np.column_stack(cols[:n_cols])
    sc = StandardScaler()
    sc.fit(raw)
    return sc


def test_subset_scaler_preserves_per_column_stats():
    """Subsetted scaler.mean_/scale_/var_ must match the source scaler at the picked indices."""
    from src.training.trainers.transformer_trainer import TransformerDirectionTrainer

    src = _make_real_scaler(n_rows=5000, n_cols=12)
    indices = [0, 3, 5, 7]

    sub = TransformerDirectionTrainer._subset_scaler(src, indices)

    np.testing.assert_array_equal(sub.mean_, src.mean_[indices])
    np.testing.assert_array_equal(sub.scale_, src.scale_[indices])
    np.testing.assert_array_equal(sub.var_, src.var_[indices])
    assert sub.n_features_in_ == len(indices)
    assert sub.n_samples_seen_ == src.n_samples_seen_


def test_subset_scaler_transform_matches_source_on_selected_columns():
    """transform() with subsetted scaler must match source scaler's transform on those cols."""
    from src.training.trainers.transformer_trainer import TransformerDirectionTrainer

    src = _make_real_scaler(n_rows=5000, n_cols=12)
    indices = [1, 4, 9]

    sub = TransformerDirectionTrainer._subset_scaler(src, indices)

    rng = np.random.default_rng(0)
    raw_row = rng.normal(0.0, 2.0, 12).reshape(1, -1)
    expected = src.transform(raw_row)[:, indices]
    actual = sub.transform(raw_row[:, indices])

    np.testing.assert_allclose(expected, actual, rtol=1e-12, atol=0.0)


def test_subset_scaler_handles_none_input():
    """Passing None returns None — no crash, no fabricated scaler."""
    from src.training.trainers.transformer_trainer import TransformerDirectionTrainer

    assert TransformerDirectionTrainer._subset_scaler(None, [0, 1]) is None


# ---------------------------------------------------------------------------
# 2. _assert_scaler_not_identity: identity-fingerprint warning
# ---------------------------------------------------------------------------

def test_assert_scaler_not_identity_warns_on_double_fit_signature(caplog):
    """A scaler fit on already-standardized data has var_=1.0 and should trigger warning."""
    from src.training.trainers.transformer_trainer import TransformerDirectionTrainer

    rng = np.random.default_rng(1)
    raw = rng.normal(0.0, 1.0, (3000, 8))
    sc1 = StandardScaler().fit(raw)
    pre_scaled = sc1.transform(raw)
    sc2 = StandardScaler().fit(pre_scaled)  # the bug pattern: var_≈1.0 across all cols

    with caplog.at_level(logging.ERROR, logger="src.training.trainers.transformer_trainer"):
        TransformerDirectionTrainer._assert_scaler_not_identity(sc2)

    fingerprint_logged = any(
        "SCALER IDENTITY-FINGERPRINT DETECTED" in record.message
        for record in caplog.records
    )
    assert fingerprint_logged, "Identity-fingerprint warning must fire on double-fit pattern"


def test_assert_scaler_not_identity_quiet_on_real_fit(caplog):
    """A scaler fit on raw mixed-scale data has real var_; no warning expected."""
    from src.training.trainers.transformer_trainer import TransformerDirectionTrainer

    real = _make_real_scaler(n_rows=3000, n_cols=10)
    with caplog.at_level(logging.ERROR, logger="src.training.trainers.transformer_trainer"):
        TransformerDirectionTrainer._assert_scaler_not_identity(real)

    fingerprint_logged = any(
        "SCALER IDENTITY-FINGERPRINT DETECTED" in record.message
        for record in caplog.records
    )
    assert not fingerprint_logged, (
        "Real per-column stats should NOT trigger identity warning"
    )


def test_assert_scaler_not_identity_handles_unfit_or_none(caplog):
    """Unfit scaler / None must not crash and must not emit the warning."""
    from src.training.trainers.transformer_trainer import TransformerDirectionTrainer

    with caplog.at_level(logging.ERROR, logger="src.training.trainers.transformer_trainer"):
        TransformerDirectionTrainer._assert_scaler_not_identity(None)
        TransformerDirectionTrainer._assert_scaler_not_identity(StandardScaler())  # unfit

    assert not any(
        "SCALER IDENTITY-FINGERPRINT DETECTED" in record.message
        for record in caplog.records
    )


# ---------------------------------------------------------------------------
# 3. load_direction_data persists regime_quantiles + version into result dict
# ---------------------------------------------------------------------------

def _make_synthetic_ohlcv_df(n: int = 4000, freq: str = "15min") -> pd.DataFrame:
    """Synthetic FX-like OHLCV DataFrame with realistic ATR variation."""
    rng = np.random.default_rng(7)
    times = pd.date_range("2024-01-01", periods=n, freq=freq)
    close = 1.10 + np.cumsum(rng.normal(0.0, 0.0001, n))
    high = close + np.abs(rng.normal(0.0, 0.0002, n))
    low = close - np.abs(rng.normal(0.0, 0.0002, n))
    open_ = close + rng.normal(0.0, 0.0001, n)
    volume = rng.integers(1000, 5000, n).astype(float)
    return pd.DataFrame({
        "time": times,
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    })


def test_load_direction_data_returns_regime_quantiles_when_atr_available():
    """When compute_normalized_features produces atr_pct_*, regime_quantiles must populate."""
    df = _make_synthetic_ohlcv_df(n=4000)
    df = compute_normalized_features(df)
    assert "atr_pct_20" in df.columns or "atr_pct_14" in df.columns, (
        "Synthetic data should produce ATR features; check test fixture if this fails"
    )

    result = load_direction_data(df, apply_scaler=False, lookahead=24, threshold=0.0001)

    assert result is not None, "load_direction_data must succeed on valid synthetic data"
    assert "regime_quantiles" in result, "regime_quantiles key must be present in result"
    assert "regime_atr_col" in result, "regime_atr_col key must be present in result"
    assert "feature_pipeline_version" in result, (
        "feature_pipeline_version key must be present in result"
    )

    rq = result["regime_quantiles"]
    assert rq is not None, "regime_quantiles must populate when ATR data exists"
    assert set(rq.keys()) == {"q25", "q50", "q75"}, f"unexpected quantile keys: {rq.keys()}"
    assert rq["q25"] < rq["q50"] < rq["q75"], (
        f"quantiles must be ordered: q25={rq['q25']}, q50={rq['q50']}, q75={rq['q75']}"
    )

    assert result["regime_atr_col"] in ("atr_pct_20", "atr_pct_14")
    assert result["feature_pipeline_version"] == FEATURE_PIPELINE_VERSION


def test_load_direction_data_includes_regime_features_in_feature_names():
    """The 4 regime one-hots must appear in the returned feature_names list."""
    df = _make_synthetic_ohlcv_df(n=4000)
    df = compute_normalized_features(df)
    result = load_direction_data(df, apply_scaler=False, lookahead=24, threshold=0.0001)

    assert result is not None
    fn = result["feature_names"]
    for regime_name in ("regime_low", "regime_normal", "regime_high", "regime_extreme"):
        assert regime_name in fn, (
            f"{regime_name} must be in feature_names; got first 10: {fn[:10]}"
        )


def test_feature_pipeline_version_constant_format():
    """Version string follows YYYY-MM-DD-vN format so version bumps are unambiguous."""
    import re
    assert re.match(r"^\d{4}-\d{2}-\d{2}-v\d+$", FEATURE_PIPELINE_VERSION), (
        f"FEATURE_PIPELINE_VERSION must follow YYYY-MM-DD-vN format; got {FEATURE_PIPELINE_VERSION}"
    )

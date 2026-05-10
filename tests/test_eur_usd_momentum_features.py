"""Test EUR_USD lgbm_momentum runtime feature alignment.

Bug 2026-05-09: EUR_USD's lgbm_momentum was suspected of producing
near-constant outputs because its saved feature_names contains
``instrument_EUR_USD`` (per-pair training-time constant=1) while live
inference dataframes do not have that column. ``align_features_to_model``
historically defaulted that slot to ``fill_value=0.0`` (RF_EXCLUDED_FEATURES).
After the saved StandardScaler ran (mean=1.0, scale=1.0), the feature
became -1.0 at inference vs 0.0 at training — feeding tree-based models an
out-of-distribution value.

Fix: ``align_features_to_model(..., instrument='EUR_USD')`` now restores
the training-time one-hot (1.0) so the saved scaler reproduces the
training-time scaled value (0.0).

Tests use REAL classes against REAL disk per CLAUDE.md No-Mock Rule.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent
EUR_USD_MODEL = REPO_ROOT / "trained_data/models/EUR_USD/lgbm_momentum.pkl"


def _load_eur_usd_lgbm_momentum() -> dict:
    if not EUR_USD_MODEL.exists():
        pytest.skip(f"EUR_USD lgbm_momentum.pkl missing at {EUR_USD_MODEL}")
    with open(EUR_USD_MODEL, "rb") as f:
        return pickle.load(f)


def _build_synthetic_feature_frame(seed: int, n_bars: int = 60) -> pd.DataFrame:
    """Build a feature frame matching the live `compute_normalized_features`
    output. We deliberately omit the `instrument_*` one-hot (live dataframe
    does not produce it) — the alignment system should add it.
    """
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "returns_1": rng.normal(0, 0.001, n_bars),
            "returns_2": rng.normal(0, 0.001, n_bars),
            "returns_3": rng.normal(0, 0.001, n_bars),
            "returns_5": rng.normal(0, 0.002, n_bars),
            "returns_10": rng.normal(0, 0.003, n_bars),
            "returns_20": rng.normal(0, 0.004, n_bars),
            "log_returns_1": rng.normal(0, 0.001, n_bars),
            "atr_pct_5": np.abs(rng.normal(0, 0.001, n_bars)),
            "atr_pct_10": np.abs(rng.normal(0, 0.001, n_bars)),
            "atr_pct_14": np.abs(rng.normal(0, 0.001, n_bars)),
            "volatility_5": np.abs(rng.normal(0, 0.0005, n_bars)),
            "volatility_10": np.abs(rng.normal(0, 0.0005, n_bars)),
            "rsi_norm": rng.uniform(0, 1, n_bars),
            "stoch_k_norm": rng.uniform(0, 1, n_bars),
            "macd_norm": rng.normal(0, 0.001, n_bars),
            "macd_hist_norm": rng.normal(0, 0.0003, n_bars),
            "volume_ratio_5": np.abs(rng.normal(1, 1, n_bars)),
            "volume_ratio_10": np.abs(rng.normal(1, 1, n_bars)),
            "volume_zscore": rng.normal(0, 1, n_bars),
        }
    )


def test_align_produces_correct_shape_and_instrument_one_hot():
    """Verify align_features_to_model returns the saved feature count and
    that ``instrument_EUR_USD`` is set to 1.0 (training value) at inference.
    """
    from src.core.modular_data_loaders import align_features_to_model

    model = _load_eur_usd_lgbm_momentum()
    feature_names = model["feature_names"]
    assert "instrument_EUR_USD" in feature_names

    df = _build_synthetic_feature_frame(seed=0)
    X = align_features_to_model(
        df, feature_names, fill_value=0.0, instrument="EUR_USD"
    )

    assert X.shape == (len(df), len(feature_names)), (
        f"shape {X.shape} != ({len(df)}, {len(feature_names)})"
    )
    instrument_idx = feature_names.index("instrument_EUR_USD")
    assert np.allclose(X[:, instrument_idx], 1.0), (
        f"instrument_EUR_USD column should be 1.0, got {X[:, instrument_idx]}"
    )


def test_align_without_instrument_keeps_old_behavior():
    """Backward-compat: with instrument=None the bug-state behavior persists
    (instrument one-hot defaults to fill_value=0). Confirms the fix is
    opt-in and we have not changed the default for callers that didn't
    update.
    """
    from src.core.modular_data_loaders import align_features_to_model

    model = _load_eur_usd_lgbm_momentum()
    feature_names = model["feature_names"]
    df = _build_synthetic_feature_frame(seed=0)

    X = align_features_to_model(df, feature_names, fill_value=0.0)
    instrument_idx = feature_names.index("instrument_EUR_USD")
    assert np.allclose(X[:, instrument_idx], 0.0)


def test_eur_usd_momentum_varies_across_distinct_inputs():
    """Drive the production momentum-scoring code path with 5 distinct
    synthetic feature snapshots and assert the model produces non-trivial
    variance (>0.005). Uses real ``LightGBMMomentumTrainer`` against the
    live pickle, exercising the same scaler/model objects production loads.
    """
    from src.training.modular_trainers import LightGBMMomentumTrainer
    from src.core.modular_data_loaders import align_features_to_model

    if not EUR_USD_MODEL.exists():
        pytest.skip(f"EUR_USD lgbm_momentum.pkl missing at {EUR_USD_MODEL}")

    trainer = LightGBMMomentumTrainer()
    trainer.load(str(EUR_USD_MODEL))

    feature_names = trainer.feature_names
    assert feature_names is not None
    assert len(feature_names) == 20

    predictions = []
    for seed in range(5):
        df = _build_synthetic_feature_frame(seed=seed)
        X = align_features_to_model(
            df, feature_names, fill_value=0.0, instrument="EUR_USD"
        )
        # LightGBMMomentumTrainer.predict applies its own scaler internally.
        result = trainer.predict(X)
        predictions.append(float(result["momentum"]))

    variance = float(np.var(predictions, ddof=0))
    std = float(np.std(predictions, ddof=0))
    assert std > 0.005, (
        f"EUR_USD momentum std {std:.6f} < 0.005 — "
        f"model may be receiving near-identical features. "
        f"Predictions: {predictions}"
    )
    # At least 3 distinct values (sanity)
    assert len({round(p, 6) for p in predictions}) >= 3, (
        f"EUR_USD predictions collapsed to <3 unique values: {predictions}"
    )
    # Sanity: variance is positive
    assert variance > 0.0


def test_eur_usd_inference_through_modular_ensemble():
    """End-to-end: drive ``ModularEnsembleInference._extract_xgb_features``
    on a live-shape DataFrame and confirm it returns the correct shape
    (20 features) with the instrument one-hot set.

    This pins the call-site change in ``modular_inference.py`` so future
    refactors can't silently drop the ``instrument=self.instrument`` kwarg
    that closes the OOD-feature gap.
    """
    from src.core.modular_inference import (
        InferenceConfig,
        ModularEnsembleInference,
    )

    if not EUR_USD_MODEL.exists():
        pytest.skip(f"EUR_USD lgbm_momentum.pkl missing at {EUR_USD_MODEL}")

    cfg = InferenceConfig()
    cfg.enforce_model_contracts = False
    cfg.require_direction_model = False
    cfg.require_volatility_model = False
    cfg.require_xgb_model = False
    cfg.require_rf_model = False

    ensemble = ModularEnsembleInference(
        model_dir=str(REPO_ROOT / "trained_data/models"),
        config=cfg,
        instrument="EUR_USD",
        use_rl_sizer=False,
        use_rl_gates=False,
        use_rl_exits=False,
        enable_market_intelligence=False,
        enable_drift_detection=False,
    )
    ensemble.load_models(instrument="EUR_USD")

    if ensemble.xgb is None:
        pytest.skip("EUR_USD momentum model failed to load via ensemble")

    feature_names = getattr(ensemble.xgb, "feature_names", None)
    assert feature_names is not None
    assert "instrument_EUR_USD" in feature_names

    df = _build_synthetic_feature_frame(seed=42)
    X = ensemble._extract_xgb_features(df)

    assert X.shape == (len(df), len(feature_names)), (
        f"_extract_xgb_features returned shape {X.shape}, "
        f"expected ({len(df)}, {len(feature_names)})"
    )
    instrument_idx = feature_names.index("instrument_EUR_USD")
    assert np.allclose(X[:, instrument_idx], 1.0), (
        f"instrument one-hot at index {instrument_idx} should be 1.0 — "
        f"the bug-trace fix in _extract_xgb_features must pass "
        f"instrument=self.instrument"
    )

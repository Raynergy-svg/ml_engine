"""
Test the Phase 2.A inference-contract round-trip in TransformerDirectionTrainer.

Sibling bug to c0530b5 (Bug A's MC-dropout scaler+regime fix): the trainer
SAVES `regime_quantiles`, `regime_atr_col`, `feature_pipeline_version` into
the meta sidecar (transformer_trainer.py:3211-3213) but `load()` historically
NEVER read them back. As a result:

  modular_inference._apply_scaler_and_regime
    -> getattr(self.tcn, "_regime_quantiles", None)  # always None
    -> regime one-hot rebuild is skipped
    -> helper degrades to the legacy unscaled path on every contract-complete
       artifact, silently neutralising Bug A's just-landed fix.

The fix (transformer_trainer.py:3517-3527) wires three `meta.get(...)` reads
into the matching instance attributes. These tests are the round-trip witness.

NO-MOCK RULE: per `.claude/rules/improvement.md` "No-Mock Rule" (promoted
2026-05-01). Tests build a real tiny keras model, save it via the real
trainer.save() path, load it via trainer.load(), and assert on real disk
state + real instance attributes. No `unittest.mock`, no `MagicMock`, no
patching. The pickle reads/writes here mirror the production save/load
contract in src/training/trainers/transformer_trainer.py (lines 3223-3225,
3502-3503) — pickle is intrinsic to the code path under test.
"""

from __future__ import annotations

import pickle  # noqa: S403 - mirrors production trainer.save/load
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.training.trainers.callbacks import TrainingLineage
from src.training.trainers.transformer_trainer import TransformerDirectionTrainer
from src.training.trainers.utils import META_PKL_SUFFIX


# --------------------------- helpers (real, no mocks) ---------------------------


def _build_tiny_keras_model(seq_len: int = 8, n_features: int = 4):
    """A real keras model small enough to save/load quickly."""
    from tensorflow import keras

    inp = keras.Input(shape=(seq_len, n_features), name="features")
    x = keras.layers.GlobalAveragePooling1D()(inp)
    out = keras.layers.Dense(1, activation="sigmoid", name="direction")(x)
    m = keras.Model(inp, out, name="tiny_transformer_for_contract_tests")
    return m


def _write_real_artifact(
    model_path: Path,
    *,
    seq_len: int = 8,
    n_features: int = 4,
    extra_meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Persist a real .keras model AND a real .meta.pkl beside it.

    Returns the meta dict actually written, so tests can assert round-trip
    equality.
    """
    from sklearn.preprocessing import StandardScaler

    model = _build_tiny_keras_model(seq_len=seq_len, n_features=n_features)
    model.save(str(model_path))

    # A real fitted scaler with NON-IDENTITY stats (so any assert_scaler_not_identity
    # tripwire would NOT fire on this artifact).
    rng = np.random.default_rng(0)
    raw = rng.normal(loc=0.5, scale=2.0, size=(64, n_features))
    scaler = StandardScaler().fit(raw)

    feature_names = [f"f{i}" for i in range(n_features)]

    meta: Dict[str, Any] = {
        "scaler": scaler,
        "feature_names": feature_names,
        "n_features": n_features,
        "seq_len": seq_len,
        "selected_indices": None,
        "metrics": {},
        "model_type": "transformer",
        "lineage": None,
        "is_warm_start": False,
        "ema_enabled": False,
        "ewc_enabled": False,
        "replay_enabled": False,
        "output_calibration": None,
        "architecture": {
            "input_shape": (seq_len, n_features),
            "transformer_d_model": 16,
            "transformer_num_heads": 2,
            "transformer_num_layers": 1,
            "transformer_dff": 32,
            "transformer_dropout": 0.2,
        },
        # Phase 2.A contract fields default to None unless the caller overrides
        "regime_quantiles": None,
        "regime_atr_col": None,
        "feature_pipeline_version": None,
        # Stage 4-A news fields — None on a contract-only test artifact
        "news_pca": None,
        "news_event_class_count_columns": [],
        "news_lookback_window_hours": None,
        "news_pca_n_components": None,
    }
    if extra_meta:
        meta.update(extra_meta)

    meta_path = model_path.with_suffix(META_PKL_SUFFIX)
    with open(meta_path, "wb") as f:
        pickle.dump(meta, f)  # noqa: S301 - mirrors production trainer.save

    return meta


# --------------------------------- the 6 tests ----------------------------------


def test_load_reads_regime_quantiles_from_meta(tmp_path: Path) -> None:
    """meta.regime_quantiles is wired to self._regime_quantiles on load()."""
    model_path = tmp_path / "transformer_direction.keras"
    quantiles = {"q25": 0.0125, "q50": 0.0250, "q75": 0.0500}
    _write_real_artifact(model_path, extra_meta={"regime_quantiles": quantiles})

    trainer = TransformerDirectionTrainer()
    trainer.load(str(model_path))

    assert trainer._regime_quantiles == quantiles


def test_load_reads_regime_atr_col_from_meta(tmp_path: Path) -> None:
    """meta.regime_atr_col is wired to self._regime_atr_col on load()."""
    model_path = tmp_path / "transformer_direction.keras"
    _write_real_artifact(model_path, extra_meta={"regime_atr_col": "atr_pct_20"})

    trainer = TransformerDirectionTrainer()
    trainer.load(str(model_path))

    assert trainer._regime_atr_col == "atr_pct_20"


def test_load_reads_feature_pipeline_version_from_meta(tmp_path: Path) -> None:
    """meta.feature_pipeline_version is wired on load()."""
    model_path = tmp_path / "transformer_direction.keras"
    _write_real_artifact(
        model_path, extra_meta={"feature_pipeline_version": "2026-05-08-v1"}
    )

    trainer = TransformerDirectionTrainer()
    trainer.load(str(model_path))

    assert trainer._feature_pipeline_version == "2026-05-08-v1"


def test_load_falls_back_to_none_when_keys_missing(tmp_path: Path) -> None:
    """Older artifacts (pre-Phase-2.A) lack the contract keys: load must not
    crash, attributes must default to None so the helper can degrade gracefully
    to the legacy path."""
    model_path = tmp_path / "transformer_direction.keras"

    # Build the artifact, then REMOVE the three contract keys to simulate an
    # older meta sidecar.
    meta = _write_real_artifact(model_path)
    meta_path = model_path.with_suffix(META_PKL_SUFFIX)
    for k in ("regime_quantiles", "regime_atr_col", "feature_pipeline_version"):
        meta.pop(k, None)
    with open(meta_path, "wb") as f:
        pickle.dump(meta, f)  # noqa: S301 - mirrors production trainer.save

    trainer = TransformerDirectionTrainer()
    trainer.load(str(model_path))  # must NOT raise

    assert trainer._regime_quantiles is None
    assert trainer._regime_atr_col is None
    assert trainer._feature_pipeline_version is None


def test_round_trip_save_then_load_preserves_all_contract_keys(
    tmp_path: Path,
) -> None:
    """Set attributes on trainer A, save, load into trainer B, assert all three
    round-trip identically through the real disk."""
    from sklearn.preprocessing import StandardScaler

    model_path = tmp_path / "transformer_direction.keras"

    # --- Build trainer A with a real keras model + real scaler + real attrs ---
    trainer_a = TransformerDirectionTrainer()
    trainer_a.model = _build_tiny_keras_model(seq_len=8, n_features=4)
    trainer_a.seq_len = 8
    trainer_a.n_features = 4
    trainer_a.feature_names = ["f0", "f1", "f2", "f3"]
    trainer_a.selected_indices = None
    trainer_a.metrics = {"holdout_accuracy": 0.71}
    rng = np.random.default_rng(1)
    raw = rng.normal(loc=0.5, scale=2.0, size=(64, 4))
    trainer_a.scaler = StandardScaler().fit(raw)
    trainer_a.lineage = TrainingLineage()
    trainer_a.output_calibration = None
    # Disable continual-learning save side-effects to keep the artifact minimal.
    trainer_a._use_ema = False
    trainer_a._use_ewc = False
    trainer_a._use_replay = False

    # The three contract fields under test
    expected_quantiles = {"q25": 0.011, "q50": 0.022, "q75": 0.044}
    trainer_a._regime_quantiles = expected_quantiles
    trainer_a._regime_atr_col = "atr_pct_14"
    trainer_a._feature_pipeline_version = "2026-05-11-test"

    trainer_a.save(str(model_path), instrument="TEST_PAIR")

    # --- Fresh trainer B loads from disk ---
    trainer_b = TransformerDirectionTrainer()
    trainer_b.load(str(model_path))

    assert trainer_b._regime_quantiles == expected_quantiles
    assert trainer_b._regime_atr_col == "atr_pct_14"
    assert trainer_b._feature_pipeline_version == "2026-05-11-test"


def test_real_eur_usd_artifact_load_populates_contract_fields_consistently() -> (
    None
):
    """Sanity check against the real production artifact on disk.

    The May 11 EUR_USD artifact predates Phase 2.A's save-side wiring so all
    three keys are absent from its meta sidecar. Verify that load() handles
    the older shape and that the three attributes are EITHER all populated
    (contract-complete future artifact) OR all None (current degraded artifact)
    — but NOT a partial state, which would indicate inconsistent loading.
    """
    artifact = (
        REPO_ROOT
        / "trained_data"
        / "models"
        / "EUR_USD"
        / "transformer_direction.keras"
    )
    if not artifact.exists():
        pytest.skip(
            f"Real EUR_USD artifact not present at {artifact}; nothing to verify"
        )

    trainer = TransformerDirectionTrainer()
    trainer.load(str(artifact))

    rq = trainer._regime_quantiles
    rc = trainer._regime_atr_col
    rv = trainer._feature_pipeline_version

    # The contract fields are either all-populated (post Phase 2.A retrain)
    # or all-None (pre Phase 2.A artifact). NEVER partial.
    populated = [x is not None for x in (rq, rc, rv)]
    assert populated == [True, True, True] or populated == [False, False, False], (
        f"Partial contract state on real EUR_USD artifact: "
        f"regime_quantiles={rq!r}, regime_atr_col={rc!r}, "
        f"feature_pipeline_version={rv!r}"
    )


# ------------------------- integration sanity check (Step 5) -------------------


def test_integration_helper_sees_loaded_contract_attrs(tmp_path: Path) -> None:
    """End-to-end: after the load-side fix, modular_inference's helper sees
    real (non-None) regime_quantiles + regime_atr_col on a contract-complete
    trainer instance. Without the fix this would be None and the helper would
    degrade to the legacy unscaled path.

    We DO NOT invoke the full predict path (it needs a fitted TCN, news
    pipeline, etc.). We assert the contract-attr handoff that the helper
    relies on via getattr() is non-None — the precondition for the helper
    NOT degrading to the unscaled legacy path (modular_inference.py:2787-2790).
    """
    from sklearn.preprocessing import StandardScaler

    model_path = tmp_path / "transformer_direction.keras"

    src_trainer = TransformerDirectionTrainer()
    src_trainer.model = _build_tiny_keras_model(seq_len=8, n_features=4)
    src_trainer.seq_len = 8
    src_trainer.n_features = 4
    src_trainer.feature_names = ["atr_pct_20", "f1", "f2", "f3"]
    src_trainer.selected_indices = None
    src_trainer.metrics = {}
    rng = np.random.default_rng(2)
    src_trainer.scaler = StandardScaler().fit(rng.normal(size=(64, 4)))
    src_trainer.lineage = TrainingLineage()
    src_trainer.output_calibration = None
    src_trainer._use_ema = False
    src_trainer._use_ewc = False
    src_trainer._use_replay = False
    src_trainer._regime_quantiles = {"q25": 0.01, "q50": 0.02, "q75": 0.04}
    src_trainer._regime_atr_col = "atr_pct_20"
    src_trainer._feature_pipeline_version = "2026-05-11-test"

    src_trainer.save(str(model_path), instrument="TEST_PAIR")

    dst_trainer = TransformerDirectionTrainer()
    dst_trainer.load(str(model_path))

    # The exact getattr() calls modular_inference._predict_with_uncertainty
    # performs at lines 2787-2788. Must be non-None for the helper NOT to
    # fall back to the legacy unscaled path.
    rq_seen_by_helper = getattr(dst_trainer, "_regime_quantiles", None)
    rc_seen_by_helper = getattr(dst_trainer, "_regime_atr_col", None)
    scaler_seen = getattr(dst_trainer, "scaler", None)
    fnames_seen = getattr(dst_trainer, "feature_names", None)

    assert rq_seen_by_helper == {"q25": 0.01, "q50": 0.02, "q75": 0.04}
    assert rc_seen_by_helper == "atr_pct_20"
    assert scaler_seen is not None
    assert fnames_seen == ["atr_pct_20", "f1", "f2", "f3"]

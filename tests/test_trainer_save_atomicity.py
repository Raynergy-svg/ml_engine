"""Real-disk tests for the 2026-06-10 training-infra audit fixes.

Covers three trainer bugs:
1. Best-epoch metric honesty — ``train_accuracy_at_best_epoch`` helper used by
   tcn_trainer / tcn_volatility_trainer (transformer already had the fix).
2. Atomic saves — every trainer save path goes through tmp+fsync+os.replace
   helpers in ``src/training/trainers/utils.py``.
3. ``feature_pipeline_version`` inference-contract key saved by every
   non-transformer head.

NO MOCKS (per .claude/rules/improvement.md "No-Mock Rule"): real trainer
classes, real fitted sklearn/LightGBM estimators, real disk via tmp_path.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pytest

from src.core.modular_data_loaders import FEATURE_PIPELINE_VERSION
from src.training.trainers.utils import (
    _atomic_tmp_path,
    atomic_json_dump,
    atomic_pickle_dump,
    atomic_text_write,
    train_accuracy_at_best_epoch,
)


def _assert_no_tmp_residue(directory: Path) -> None:
    leftovers = [p.name for p in directory.rglob("*.tmp*")]
    assert leftovers == [], f"temp-file residue after atomic save: {leftovers}"


def _tiny_xy(n: int = 40, d: int = 4, seed: int = 7):
    rng = np.random.default_rng(seed)
    return rng.random((n, d)), rng.integers(0, 2, n)


# =============================================================================
# Atomic helper primitives
# =============================================================================


class TestAtomicHelpers:
    def test_pickle_dump_round_trip_no_residue(self, tmp_path: Path) -> None:
        path = tmp_path / "obj.pkl"
        payload = {"a": 1, "arr": [1.0, 2.0]}
        atomic_pickle_dump(payload, path)

        assert path.exists()
        _assert_no_tmp_residue(tmp_path)
        with open(path, "rb") as f:
            assert pickle.load(f) == payload

    def test_pickle_dump_failure_preserves_existing_artifact(
        self, tmp_path: Path
    ) -> None:
        """ENOSPC/kill mid-save must never corrupt the live artifact. We
        simulate a mid-write failure with an unpicklable object — the write
        dies inside pickle.dump, exactly like a torn write would."""
        path = tmp_path / "model.pkl"
        good = {"model": "GOOD"}
        atomic_pickle_dump(good, path)
        original_bytes = path.read_bytes()

        unpicklable = {"model": lambda: None}  # lambdas cannot be pickled
        with pytest.raises((pickle.PicklingError, AttributeError, TypeError)):
            atomic_pickle_dump(unpicklable, path)

        assert path.read_bytes() == original_bytes, "live artifact was corrupted"
        _assert_no_tmp_residue(tmp_path)

    def test_text_and_json_round_trip(self, tmp_path: Path) -> None:
        txt = tmp_path / "arch.json"
        atomic_text_write('{"layers": []}', txt)
        assert txt.read_text() == '{"layers": []}'

        js = tmp_path / "meta.json"
        atomic_json_dump({"k": 1}, js, indent=2)
        assert '"k": 1' in js.read_text()
        _assert_no_tmp_residue(tmp_path)

    def test_tmp_path_preserves_keras_suffixes(self, tmp_path: Path) -> None:
        """Keras refuses non-.keras / non-.weights.h5 paths, so the temp file
        must keep the trailing extension."""
        keras_tmp = _atomic_tmp_path(
            tmp_path / "transformer_direction.keras", ".keras"
        )
        assert keras_tmp.name == "transformer_direction.tmp.keras"

        weights_tmp = _atomic_tmp_path(
            tmp_path / "transformer_direction.weights.h5", ".weights.h5"
        )
        assert weights_tmp.name == "transformer_direction.tmp.weights.h5"

        plain_tmp = _atomic_tmp_path(tmp_path / "model.pkl")
        assert plain_tmp.name == "model.pkl.tmp"


# =============================================================================
# Metric honesty — train_accuracy at the best-val epoch
# =============================================================================


class TestTrainAccuracyAtBestEpoch:
    def test_uses_best_val_epoch_not_last(self) -> None:
        # Val peaks at epoch 1; train keeps climbing (overfit) afterwards.
        train = [0.60, 0.70, 0.85, 0.95]
        val = [0.50, 0.58, 0.52, 0.49]
        assert train_accuracy_at_best_epoch(train, val) == pytest.approx(0.70)

    def test_explicit_best_epoch_idx_takes_precedence(self) -> None:
        # tcn_volatility path: saved epoch keyed on best val_loss, not val_acc.
        train = [0.60, 0.70, 0.85, 0.95]
        val = [0.50, 0.58, 0.52, 0.49]
        assert train_accuracy_at_best_epoch(
            train, val, best_epoch_idx=2
        ) == pytest.approx(0.85)

    def test_sentinel_negative_idx_falls_back_to_last(self) -> None:
        # tcn_volatility initializes best_epoch_idx=-1; if it never fires we
        # must not index from the end by accident.
        train = [0.60, 0.70]
        assert train_accuracy_at_best_epoch(
            train, best_epoch_idx=-1
        ) == pytest.approx(0.70)

    def test_missing_val_history_falls_back_to_last(self) -> None:
        assert train_accuracy_at_best_epoch([0.6, 0.9]) == pytest.approx(0.9)
        assert train_accuracy_at_best_epoch([0.6, 0.9], []) == pytest.approx(0.9)

    def test_empty_history_returns_zero(self) -> None:
        assert train_accuracy_at_best_epoch([]) == 0.0
        assert train_accuracy_at_best_epoch([], [0.5]) == 0.0

    def test_val_history_longer_than_train_is_clamped(self) -> None:
        # Defensive: argmax(val) beyond train range must not IndexError.
        train = [0.6]
        val = [0.5, 0.9, 0.7]
        assert train_accuracy_at_best_epoch(train, val) == pytest.approx(0.6)


# =============================================================================
# Trainer save round-trips (real classes, real disk)
# =============================================================================


class TestHistGBSaveRoundTrip:
    def test_save_load_atomic_and_versioned(self, tmp_path: Path) -> None:
        from sklearn.ensemble import HistGradientBoostingClassifier
        from sklearn.preprocessing import StandardScaler

        from src.training.trainers.histgb_trainer import (
            HistGradientBoostingDirectionTrainer,
        )

        X, y = _tiny_xy()
        trainer = HistGradientBoostingDirectionTrainer()
        trainer.model = HistGradientBoostingClassifier(max_iter=5).fit(X, y)
        trainer.scaler = StandardScaler().fit(X)
        trainer.metrics = {"val_accuracy": 0.5}
        trainer.feature_names = ["f0", "f1", "f2", "f3"]
        trainer.n_features = 4
        trainer.is_trained = True

        path = tmp_path / "histgb_direction.pkl"
        trainer.save(str(path))

        assert path.exists()
        _assert_no_tmp_residue(tmp_path)
        with open(path, "rb") as f:
            payload = pickle.load(f)
        assert payload["feature_pipeline_version"] == FEATURE_PIPELINE_VERSION

        fresh = HistGradientBoostingDirectionTrainer()
        fresh.load(str(path))
        assert fresh.is_trained
        assert fresh.feature_names == trainer.feature_names
        assert fresh.n_features == 4
        # The restored model actually predicts (not a truncated artifact)
        assert fresh.model.predict(X[:3]).shape == (3,)


class TestRidgeConfidenceSaveRoundTrip:
    def test_save_load_atomic_and_versioned(self, tmp_path: Path) -> None:
        from sklearn.linear_model import Ridge
        from sklearn.preprocessing import StandardScaler

        from src.training.trainers.ridge_trainer import RidgeTrainer

        X, _ = _tiny_xy()
        y_conf = np.linspace(0, 100, len(X))
        trainer = RidgeTrainer()
        trainer.model = Ridge().fit(X, y_conf)
        trainer.scaler = StandardScaler().fit(X)
        trainer.metrics = {"val_mae": 5.0}
        trainer.feature_names = ["f0", "f1", "f2", "f3"]
        trainer.n_features = 4
        trainer.is_trained = True

        path = tmp_path / "ridge_confidence.pkl"
        trainer.save(str(path))

        assert path.exists()
        _assert_no_tmp_residue(tmp_path)
        with open(path, "rb") as f:
            payload = pickle.load(f)
        assert payload["feature_pipeline_version"] == FEATURE_PIPELINE_VERSION

        fresh = RidgeTrainer()
        fresh.load(str(path))
        assert fresh.is_trained
        assert fresh.model.predict(X[:2]).shape == (2,)


class TestLightGBMMomentumSaveRoundTrip:
    def test_save_load_atomic_and_versioned(self, tmp_path: Path) -> None:
        lightgbm = pytest.importorskip("lightgbm")
        from sklearn.preprocessing import StandardScaler

        from src.training.trainers.lightgbm_trainers import LightGBMMomentumTrainer

        X, _ = _tiny_xy()
        y_reg = np.linspace(0.0, 1.0, len(X))
        trainer = LightGBMMomentumTrainer()
        trainer.momentum_model = lightgbm.LGBMRegressor(
            n_estimators=5, verbose=-1, min_child_samples=2
        ).fit(X, y_reg)
        trainer.accel_model = lightgbm.LGBMRegressor(
            n_estimators=5, verbose=-1, min_child_samples=2
        ).fit(X, y_reg)
        trainer.scaler = StandardScaler().fit(X)
        trainer.metrics = {"val_mae": 0.1}
        trainer.feature_names = ["f0", "f1", "f2", "f3"]
        trainer.n_features = 4
        trainer.momentum_norm_factor = 1.0
        trainer.is_trained = True

        path = tmp_path / "lgbm_momentum.pkl"
        trainer.save(str(path))

        assert path.exists()
        _assert_no_tmp_residue(tmp_path)
        with open(path, "rb") as f:
            payload = pickle.load(f)
        assert payload["feature_pipeline_version"] == FEATURE_PIPELINE_VERSION
        assert payload["model_type"] == "lightgbm_momentum"

        fresh = LightGBMMomentumTrainer()
        fresh.load(str(path))
        assert fresh.is_trained
        assert fresh.momentum_norm_factor == 1.0
        assert fresh.momentum_model.predict(X[:2]).shape == (2,)


class TestLightGBMRiskSaveRoundTrip:
    def test_save_load_atomic_and_versioned(self, tmp_path: Path) -> None:
        lightgbm = pytest.importorskip("lightgbm")
        from sklearn.preprocessing import StandardScaler

        from src.training.trainers.lightgbm_trainers import LightGBMRiskTrainer

        X, y = _tiny_xy()
        y_reg = np.linspace(0.0, 0.05, len(X))
        trainer = LightGBMRiskTrainer()
        trainer.drawdown_model = lightgbm.LGBMRegressor(
            n_estimators=5, verbose=-1, min_child_samples=2
        ).fit(X, y_reg)
        trainer.streak_model = lightgbm.LGBMClassifier(
            n_estimators=5, verbose=-1, min_child_samples=2
        ).fit(X, y)
        trainer.scaler = StandardScaler().fit(X)
        trainer.metrics = {"val_mae": 0.01}
        trainer.feature_names = ["f0", "f1", "f2", "f3"]
        trainer.n_features = 4
        trainer.is_trained = True

        path = tmp_path / "lgbm_risk.pkl"
        trainer.save(str(path))

        assert path.exists()
        _assert_no_tmp_residue(tmp_path)
        with open(path, "rb") as f:
            payload = pickle.load(f)
        assert payload["feature_pipeline_version"] == FEATURE_PIPELINE_VERSION
        assert payload["model_type"] == "lightgbm_risk"

        fresh = LightGBMRiskTrainer()
        fresh.load(str(path))
        assert fresh.is_trained
        assert fresh.drawdown_model.predict(X[:2]).shape == (2,)


# =============================================================================
# Keras atomic save (real tiny model — validates the .tmp.keras suffix path)
# =============================================================================


class TestAtomicKerasSave:
    def test_keras_model_and_weights_atomic_save(self, tmp_path: Path) -> None:
        keras = pytest.importorskip("keras")
        from src.training.trainers.utils import atomic_keras_save

        model = keras.Sequential(
            [keras.layers.Input(shape=(3,)), keras.layers.Dense(2)]
        )

        model_path = tmp_path / "tiny_model.keras"
        atomic_keras_save(model, model_path)
        assert model_path.exists()
        _assert_no_tmp_residue(tmp_path)

        reloaded = keras.models.load_model(str(model_path))
        out = reloaded.predict(np.zeros((1, 3)), verbose=0)
        assert out.shape == (1, 2)

        weights_path = tmp_path / "tiny_model.weights.h5"
        atomic_keras_save(model, weights_path, weights_only=True)
        assert weights_path.exists()
        _assert_no_tmp_residue(tmp_path)

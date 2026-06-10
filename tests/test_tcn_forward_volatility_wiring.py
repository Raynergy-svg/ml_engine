from __future__ import annotations

from pathlib import Path

import numpy as np


def test_train_tcn_vol_regime_uses_forward_sequence_loader(monkeypatch, tmp_path):
    import scripts.train_single_model_m1 as m1
    import src.core.modular_data_loaders as loaders
    import src.training.trainers.tcn_volatility_trainer as trainer_mod

    seq_len = 12
    vol_data = {
        "X_train": np.zeros((8, seq_len, 5), dtype=np.float32),
        "y_train": np.array([0, 1, 2, 3, 0, 1, 2, 3], dtype=np.int32),
        "X_val": np.zeros((4, seq_len, 5), dtype=np.float32),
        "y_val": np.array([0, 1, 2, 3], dtype=np.int32),
        "y_train_reg": np.zeros(8, dtype=np.float32),
        "y_val_reg": np.zeros(4, dtype=np.float32),
        "w_train": np.ones(8, dtype=np.float32),
        "w_val": np.ones(4, dtype=np.float32),
        "class_weights": {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0},
        "feature_names": [f"f{i}" for i in range(5)],
        "seq_len": seq_len,
        "lookahead": 24,
        "reg_thresholds": {"quiet": -0.2, "stable_high": 0.1, "active_high": 0.3},
        "scaler": object(),
    }
    calls = {}

    def fake_loader(df, *, seq_len):
        calls["loader_seq_len"] = seq_len
        return vol_data

    class FakeTrainer:
        def __init__(self, config):
            self.config = config
            calls["config"] = config
            self.metrics = {"train_accuracy": 0.55, "val_accuracy": 0.52}
            self.reg_thresholds = {}
            self.lookahead = 48
            self.scaler = None

        def train(self, X_train, y_train, X_val, y_val, **kwargs):
            calls["x_train_shape"] = X_train.shape
            calls["x_val_shape"] = X_val.shape
            calls["kwargs"] = kwargs

        def save(self, path):
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            Path(path).write_bytes(b"fake")

    monkeypatch.setattr(m1, "MODELS_DIR", tmp_path)
    monkeypatch.setattr(loaders, "load_forward_volatility_data", fake_loader)
    monkeypatch.setattr(trainer_mod, "TCNVolatilityRegimeTrainer", FakeTrainer)

    params = {
        "seq_len": seq_len,
        "batch_size": 4,
        "lr": 0.001,
        "patience": 2,
        "tcn_kernel_size": 7,
        "tcn_nb_filters": 128,
        "tcn_dilations": [1, 2, 4, 8, 16],
    }
    result = m1.train_tcn_vol_regime("EUR_USD", object(), params)

    assert calls["loader_seq_len"] == seq_len
    assert calls["config"].tcn_kernel_size == 7
    assert getattr(calls["config"], "tcn_num_filters", 64) == 64
    assert calls["config"].tcn_num_residual_blocks == 5
    assert calls["x_train_shape"] == (8, seq_len, 5)
    assert calls["x_val_shape"] == (4, seq_len, 5)
    assert calls["kwargs"]["y_train_reg"] is vol_data["y_train_reg"]
    assert calls["kwargs"]["w_train"] is vol_data["w_train"]
    assert calls["kwargs"]["class_weights"] == vol_data["class_weights"]
    assert result["train_acc"] == 0.55
    assert result["val_acc"] == 0.52
    assert result["quarantined"] is False


def test_model_all_skips_tcn_vol_regime_alias():
    import scripts.train_single_model_m1 as m1

    models = m1._model_names_for_request("all")

    assert "tcn" in models
    assert "tcn_vol_regime" not in models
    assert m1._model_names_for_request("tcn_vol_regime") == ["tcn_vol_regime"]


def test_hard_quarantine_restores_previous_live_artifacts(monkeypatch, tmp_path):
    import scripts.train_single_model_m1 as m1

    pair_dir = tmp_path / "EUR_USD"
    pair_dir.mkdir()
    model_path = pair_dir / "transformer_direction.keras"
    meta_path = pair_dir / "transformer_direction.meta.pkl"
    model_path.write_bytes(b"old-live-model")
    meta_path.write_bytes(b"old-live-meta")

    monkeypatch.setattr(m1, "MODELS_DIR", tmp_path)
    rollback_dir = m1._snapshot_existing_artifacts(model_path)

    model_path.write_bytes(b"bad-new-model")
    meta_path.write_bytes(b"bad-new-meta")

    result = m1._quarantine_if_overshipped(
        [model_path],
        train_acc=0.80,
        val_acc=0.50,
        instrument="EUR_USD",
        model_name="transformer_direction",
        restore_dir=rollback_dir,
    )

    assert result["quarantined"] is True
    assert model_path.read_bytes() == b"old-live-model"
    assert meta_path.read_bytes() == b"old-live-meta"

    qdir = Path(result["quarantine_dir"])
    assert (qdir / "transformer_direction.keras").read_bytes() == b"bad-new-model"
    assert (qdir / "transformer_direction.meta.pkl").read_bytes() == b"bad-new-meta"


def test_tcn_forward_predict_scales_raw_2d_features(monkeypatch):
    from src.training.trainers.tcn_volatility_trainer import TCNVolatilityRegimeTrainer

    captured = {}

    class FakeScaler:
        def transform(self, X):
            captured["scaled_shape"] = X.shape
            return X + 1.0

    class FakeModel:
        def predict(self, X, verbose=0):
            captured["model_input"] = X
            return {
                "classification": np.array([[0.05, 0.15, 0.70, 0.10]]),
                "regression": np.array([[0.25]]),
            }

    trainer = TCNVolatilityRegimeTrainer()
    trainer.is_trained = True
    trainer.seq_len = 4
    trainer.scaler = FakeScaler()
    trainer.model = FakeModel()

    result = trainer.predict(np.zeros((6, 3), dtype=np.float32))

    assert captured["scaled_shape"] == (6, 3)
    assert captured["model_input"].shape == (1, 4, 3)
    assert np.all(captured["model_input"] == 1.0)
    assert result["regime"] == 2
    assert result["confidence"] == 0.70


def _synthetic_ohlcv(n, seed=0):
    """Random walk with a slowly oscillating volatility regime so the forward
    labeler sees all four classes. Real numpy/pandas, no mocks."""
    import pandas as pd

    rng = np.random.RandomState(seed)
    vol = 0.0004 + 0.0025 * (np.sin(np.linspace(0, 16 * np.pi, n)) ** 2)
    close = 1.10 * np.exp(np.cumsum(rng.randn(n) * vol))
    return pd.DataFrame({
        "time": pd.date_range("2024-01-01", periods=n, freq="15min"),
        "open": np.concatenate([[close[0]], close[:-1]]),
        "high": close * (1 + np.abs(rng.randn(n)) * vol),
        "low": close * (1 - np.abs(rng.randn(n)) * vol),
        "close": close,
        "volume": rng.randint(100, 1000, n),
    })


def test_load_forward_volatility_data_real_contract():
    """The real loader (no fakes) must satisfy the contract train_tcn_vol_regime
    and the gate depend on: 3D X, 0-3 labels, an aligned regression target, a
    fitted scaler whose transform works on raw 2D, and increasing reg_thresholds.
    """
    from src.core.modular_data_loaders import load_forward_volatility_data

    seq_len = 12
    out = load_forward_volatility_data(_synthetic_ohlcv(2000), seq_len=seq_len, lookahead=24)

    # 3D sequence shape, feature-count consistency
    assert out["X_train"].ndim == 3
    assert out["X_train"].shape[1] == seq_len
    nf = out["X_train"].shape[2]
    assert nf == len(out["feature_names"])
    assert out["X_val"].shape[1:] == (seq_len, nf)

    # 4-class labels, regression target aligned to the classification batch
    assert set(np.unique(out["y_train"]).tolist()).issubset({0, 1, 2, 3})
    assert out["y_train_reg"].shape[0] == out["X_train"].shape[0]
    assert out["y_val_reg"].shape[0] == out["X_val"].shape[0]
    assert out["w_train"].shape[0] == out["X_train"].shape[0]

    # Fitted scaler with a working transform on raw 2D (what trainer.predict feeds)
    scaler = out["scaler"]
    assert hasattr(scaler, "scale_")
    assert scaler.transform(np.zeros((5, nf), dtype=np.float32)).shape == (5, nf)

    # Inference-contract thresholds the gate reads back
    rt = out["reg_thresholds"]
    assert rt["quiet"] < rt["stable_high"] < rt["active_high"]
    assert out["seq_len"] == seq_len
    assert out["lookahead"] == 24
    assert isinstance(out["class_weights"], dict)

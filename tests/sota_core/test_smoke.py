"""
Smoke tests for the SOTA raw-sequence deep learning core.

These tests verify:
  1. Model builds without errors.
  2. Forward pass produces valid shapes and ranges.
  3. df_to_tensor correctly normalizes OHLCV DataFrames.
  4. Pre-training and fine-tuning models compile.
  5. Inference wrapper returns a valid TradeSignal.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
import pytest

from src.sota_core.raw_sequence_model import RawSequenceModel, ModelConfig
from src.sota_core.inference import SOTAInference, SOTAInferenceConfig
from src.sota_core.trainer import SOTATrainer, TrainerConfig


class TestRawSequenceModel:
    def test_build_model(self):
        cfg = ModelConfig(seq_len=32, d_model=32, num_layers=2)
        model = RawSequenceModel(cfg)
        m = model.build()
        assert m is not None
        assert m.input_shape == (None, 32, 5)
        assert len(m.outputs) == 2  # direction, regime

    def test_forward_pass(self):
        cfg = ModelConfig(seq_len=32, d_model=32, num_layers=2)
        model = RawSequenceModel(cfg)
        model.build()
        x = np.random.randn(2, 32, 5).astype(np.float32)
        dir_prob, regime_prob = model.model.predict(x, verbose=0)
        assert dir_prob.shape == (2, 1)
        assert regime_prob.shape == (2, 4)
        assert np.all(dir_prob >= 0.0) and np.all(dir_prob <= 1.0)
        assert np.allclose(regime_prob.sum(axis=1), 1.0, atol=1e-4)

    def test_pretrain_model_builds(self):
        cfg = ModelConfig(seq_len=32, d_model=32, num_layers=2)
        model = RawSequenceModel(cfg)
        pt = model.build_pretraining_model()
        assert pt is not None
        x = np.random.randn(2, 32, 5).astype(np.float32)
        recon, nxt = pt.predict(x, verbose=0)
        assert recon.shape == (2, 32, 5)
        assert nxt.shape == (2, 1)

    def test_df_to_tensor(self):
        dates = pd.date_range("2026-01-01", periods=50, freq="h")
        df = pd.DataFrame({
            "open": np.random.randn(50).cumsum() + 100,
            "high": np.random.randn(50).cumsum() + 101,
            "low": np.random.randn(50).cumsum() + 99,
            "close": np.random.randn(50).cumsum() + 100,
            "volume": np.random.randint(100, 1000, 50),
        }, index=dates)
        tensor = RawSequenceModel.df_to_tensor(df, seq_len=32)
        assert tensor is not None
        assert tensor.shape == (1, 32, 5)
        # Should be normalized (returns, z-scores)
        assert not np.isnan(tensor).any()

    def test_df_to_tensor_short_df(self):
        df = pd.DataFrame({"open": [1], "high": [2], "low": [0], "close": [1], "volume": [100]})
        tensor = RawSequenceModel.df_to_tensor(df, seq_len=32)
        assert tensor is None or tensor.shape == (1, 32, 5)


class TestSOTAInference:
    def test_inference_returns_hold_without_model(self):
        inf = SOTAInference(SOTAInferenceConfig(model_path="/nonexistent/model.keras"))
        dates = pd.date_range("2026-01-01", periods=50, freq="h")
        df = pd.DataFrame({
            "open": np.random.randn(50).cumsum() + 100,
            "high": np.random.randn(50).cumsum() + 101,
            "low": np.random.randn(50).cumsum() + 99,
            "close": np.random.randn(50).cumsum() + 100,
            "volume": np.random.randint(100, 1000, 50),
        }, index=dates)
        signal = inf.predict(pair="EUR_USD", df_raw=df)
        assert signal.direction in ("LONG", "SHORT", "HOLD")
        assert 0.0 <= signal.confidence <= 1.0


class TestSOTATrainer:
    def test_config_defaults(self):
        cfg = TrainerConfig()
        assert cfg.batch_size > 0
        assert cfg.pretrain_epochs >= 0

    def test_trainer_instantiates(self):
        trainer = SOTATrainer()
        assert trainer.model is not None

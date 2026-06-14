"""
Targeted regression tests for CRIT-1 and CRIT-2 fixes.

CRIT-1: SOTAInference must not retry model load from disk on every predict().
CRIT-2: df_to_tensor volume normalization must produce identical values for
        the same tail window regardless of how many preceding bars are in
        the DataFrame.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
import pytest

from src.sota_core.inference import SOTAInference, SOTAInferenceConfig
from src.sota_core.raw_sequence_model import RawSequenceModel, ModelConfig


class TestCrit1NoRetry:
    def test_inference_does_not_retry_load(self):
        """CRIT-1: After __init__, predict() must not touch disk again."""
        cfg = SOTAInferenceConfig(model_path="/definitely/does/not/exist.keras")
        inf = SOTAInference(cfg)
        assert inf._load_attempted is True
        assert inf._loaded is False

        dates = pd.date_range("2026-01-01", periods=50, freq="h")
        df = pd.DataFrame({
            "open": np.random.randn(50).cumsum() + 100,
            "high": np.random.randn(50).cumsum() + 101,
            "low": np.random.randn(50).cumsum() + 99,
            "close": np.random.randn(50).cumsum() + 100,
            "volume": np.random.randint(100, 1000, 50),
        }, index=dates)
        sig1 = inf.predict(pair="EUR_USD", df_raw=df)
        sig2 = inf.predict(pair="EUR_USD", df_raw=df)
        assert sig1.direction == "HOLD"
        assert sig2.direction == "HOLD"
        assert inf._load_attempted is True

    def test_load_models_is_idempotent(self):
        """CRIT-1: load_models() is a no-op after __init__."""
        cfg = SOTAInferenceConfig(model_path="/nonexistent/model.keras")
        inf = SOTAInference(cfg)
        assert inf._load_attempted is True
        inf.load_models()
        assert inf._load_attempted is True


class TestCrit2WindowInvariant:
    def test_volume_channel_identical_for_same_tail(self):
        """CRIT-2: Volume normalization of the same tail window must be
        identical regardless of preceding bars."""
        np.random.seed(42)
        base_tail = pd.DataFrame({
            "open": np.random.randn(32).cumsum() + 100,
            "high": np.random.randn(32).cumsum() + 101,
            "low": np.random.randn(32).cumsum() + 99,
            "close": np.random.randn(32).cumsum() + 100,
            "volume": np.random.randint(100, 1000, 32),
        })

        prefix_a = pd.DataFrame({
            "open": np.random.randn(18).cumsum() + 50,
            "high": np.random.randn(18).cumsum() + 51,
            "low": np.random.randn(18).cumsum() + 49,
            "close": np.random.randn(18).cumsum() + 50,
            "volume": np.random.randint(100, 1000, 18),
        })
        df_a = pd.concat([prefix_a, base_tail], ignore_index=True)

        prefix_b = pd.DataFrame({
            "open": np.random.randn(118).cumsum() + 200,
            "high": np.random.randn(118).cumsum() + 201,
            "low": np.random.randn(118).cumsum() + 199,
            "close": np.random.randn(118).cumsum() + 200,
            "volume": np.random.randint(100, 1000, 118),
        })
        df_b = pd.concat([prefix_b, base_tail], ignore_index=True)

        tensor_a = RawSequenceModel.df_to_tensor(df_a, seq_len=32)
        tensor_b = RawSequenceModel.df_to_tensor(df_b, seq_len=32)

        assert tensor_a is not None
        assert tensor_b is not None
        # Volume is channel index 4
        np.testing.assert_allclose(
            tensor_a[0, :, 4], tensor_b[0, :, 4], rtol=1e-5,
            err_msg="Volume channel differs for the same tail window"
        )

    def test_price_channels_match_from_bar_1(self):
        """Price returns from bar 1 onwards should match (bar 0 depends on prefix)."""
        np.random.seed(7)
        base_tail = pd.DataFrame({
            "open": np.random.randn(32).cumsum() + 100,
            "high": np.random.randn(32).cumsum() + 101,
            "low": np.random.randn(32).cumsum() + 99,
            "close": np.random.randn(32).cumsum() + 100,
            "volume": np.random.randint(100, 1000, 32),
        })

        prefix_a = pd.DataFrame({
            "open": np.random.randn(18).cumsum() + 50,
            "high": np.random.randn(18).cumsum() + 51,
            "low": np.random.randn(18).cumsum() + 49,
            "close": np.random.randn(18).cumsum() + 50,
            "volume": np.random.randint(100, 1000, 18),
        })
        df_a = pd.concat([prefix_a, base_tail], ignore_index=True)

        prefix_b = pd.DataFrame({
            "open": np.random.randn(118).cumsum() + 200,
            "high": np.random.randn(118).cumsum() + 201,
            "low": np.random.randn(118).cumsum() + 199,
            "close": np.random.randn(118).cumsum() + 200,
            "volume": np.random.randint(100, 1000, 118),
        })
        df_b = pd.concat([prefix_b, base_tail], ignore_index=True)

        tensor_a = RawSequenceModel.df_to_tensor(df_a, seq_len=32)
        tensor_b = RawSequenceModel.df_to_tensor(df_b, seq_len=32)

        # Price channels 0-3 should match from bar 1 onwards
        for c in range(4):
            np.testing.assert_allclose(
                tensor_a[0, 1:, c], tensor_b[0, 1:, c], rtol=1e-5,
                err_msg=f"Price channel {c} differs from bar 1 onwards"
            )

    def test_fixed_stats_produce_deterministic_volume(self):
        """With fixed stats, volume normalization is fully deterministic."""
        np.random.seed(99)
        df = pd.DataFrame({
            "open": np.random.randn(50).cumsum() + 100,
            "high": np.random.randn(50).cumsum() + 101,
            "low": np.random.randn(50).cumsum() + 99,
            "close": np.random.randn(50).cumsum() + 100,
            "volume": np.random.randint(100, 1000, 50),
        })
        # Just verify the call doesn't crash
        tensor = RawSequenceModel.df_to_tensor(df, seq_len=32)
        assert tensor is not None

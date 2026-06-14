"""Tests for hybrid inference with OOD fallback safety (US-008)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
import pytest

from src.sota_core.hybrid_inference import (
    HybridInference,
    HybridInferenceConfig,
)


class TestHybridInferenceInit:
    def test_default_config(self):
        hi = HybridInference()
        assert hi.cfg.seq_len == 128
        assert hi.cfg.confidence_threshold == 0.55
        assert hi.cfg.ood_threshold_sigma == 2.0
        assert hi.cfg.enable_ood_fallback is True

    def test_custom_config(self):
        cfg = HybridInferenceConfig(seq_len=64, confidence_threshold=0.6)
        hi = HybridInference(cfg)
        assert hi.cfg.seq_len == 64
        assert hi.cfg.confidence_threshold == 0.6


class TestOODDetection:
    def test_no_stats_returns_not_ood(self):
        hi = HybridInference()
        tensor = np.random.randn(1, 128, 5).astype(np.float32)
        assert hi._is_ood(tensor) is False
        assert hi._compute_ood_score(tensor) == 0.0

    def test_in_distribution(self):
        hi = HybridInference()
        hi.set_normalization_stats(
            {"mean": np.zeros(5, dtype=np.float32), "std": np.ones(5, dtype=np.float32)}
        )
        tensor = np.random.randn(1, 128, 5).astype(np.float32) * 0.5
        assert hi._is_ood(tensor) is False

    def test_out_of_distribution(self):
        hi = HybridInference()
        hi.set_normalization_stats(
            {"mean": np.zeros(5, dtype=np.float32), "std": np.ones(5, dtype=np.float32)}
        )
        tensor = np.random.randn(1, 128, 5).astype(np.float32) * 10.0
        assert hi._is_ood(tensor) is True

    def test_ood_disabled(self):
        cfg = HybridInferenceConfig(enable_ood_fallback=False)
        hi = HybridInference(cfg)
        hi.set_normalization_stats(
            {"mean": np.zeros(5, dtype=np.float32), "std": np.ones(5, dtype=np.float32)}
        )
        tensor = np.random.randn(1, 128, 5).astype(np.float32) * 10.0
        assert hi._is_ood(tensor) is False


class TestPredict:
    def _make_df(self, n: int = 50) -> pd.DataFrame:
        dates = pd.date_range("2026-01-01", periods=n, freq="h")
        return pd.DataFrame(
            {
                "open": np.random.randn(n).cumsum() + 100,
                "high": np.random.randn(n).cumsum() + 101,
                "low": np.random.randn(n).cumsum() + 99,
                "close": np.random.randn(n).cumsum() + 100,
                "volume": np.random.randint(100, 1000, n),
            },
            index=dates,
        )

    def test_no_data_returns_hold(self):
        hi = HybridInference()
        result = hi.predict("EUR_USD", None)
        assert result["direction"] == "HOLD"
        assert result["source"] == "no_data"

    def test_empty_df_returns_hold(self):
        hi = HybridInference()
        result = hi.predict("EUR_USD", pd.DataFrame())
        assert result["direction"] == "HOLD"
        assert result["source"] == "no_data"

    def test_insufficient_data_returns_hold(self):
        hi = HybridInference()
        df = self._make_df(n=5)
        result = hi.predict("EUR_USD", df)
        assert result["direction"] == "HOLD"
        assert result["source"] == "insufficient_data"

    def test_predict_structure(self):
        hi = HybridInference()
        df = self._make_df(n=200)
        result = hi.predict("EUR_USD", df)
        assert "direction" in result
        assert "confidence" in result
        assert "regime" in result
        assert "ood_detected" in result
        assert "weights_used" in result
        assert "source" in result
        assert result["direction"] in ("LONG", "SHORT", "HOLD")
        assert 0.0 <= result["confidence"] <= 1.0

    def test_ood_fallback_changes_weights(self):
        hi = HybridInference()
        # Monkeypatch _is_ood to always return True
        original_is_ood = hi._is_ood
        hi._is_ood = lambda tensor: True
        df = self._make_df(n=200)
        result = hi.predict("EUR_USD", df)
        assert result["ood_detected"] is True
        # When OOD, custom model weight should be boosted to 0.6
        if "custom" in result["weights_used"]:
            assert result["weights_used"]["custom"] == pytest.approx(0.6)
        hi._is_ood = original_is_ood


    def test_all_models_failed_fallback(self):
        hi = HybridInference()
        df = self._make_df(n=200)
        # Force OOD + no models registered = all models effectively fail
        hi.set_normalization_stats(
            {"mean": np.zeros(5, dtype=np.float32), "std": np.ones(5, dtype=np.float32)}
        )
        # Make it OOD so foundation is skipped, custom may fail on predict
        df["close"] = df["close"] * 1000.0
        result = hi.predict("EUR_USD", df)
        assert result["direction"] == "HOLD"
        assert result["source"] in ("all_models_failed", "zero_weight")

    def test_drop_in_replacement_signature(self):
        """Verify predict() accepts pair, df_raw, **kwargs."""
        hi = HybridInference()
        df = self._make_df(n=200)
        result = hi.predict("EUR_USD", df_raw=df, extra_kwarg=True)
        assert "direction" in result


class TestModelRegistration:
    def test_register_foundation(self):
        hi = HybridInference()
        class FakeModel:
            def predict(self, pair, df_raw, **kwargs):
                return {"confidence": 0.8}

        hi.register_foundation(FakeModel())
        assert hi._foundation is not None

    def test_register_xgboost(self):
        hi = HybridInference()
        hi.register_xgboost("fake_xgb")
        assert hi._xgb == "fake_xgb"

    def test_foundation_predict_integration(self):
        hi = HybridInference()
        df = pd.DataFrame(
            {
                "open": [100.0] * 200,
                "high": [101.0] * 200,
                "low": [99.0] * 200,
                "close": [100.0] * 200,
                "volume": [500] * 200,
            },
            index=pd.date_range("2026-01-01", periods=200, freq="h"),
        )

        class FakeFoundation:
            def predict(self, pair, df_raw, **kwargs):
                return {"confidence": 0.8}

        hi.register_foundation(FakeFoundation())
        result = hi.predict("EUR_USD", df)
        assert result["source"] == "hybrid_ensemble"
        assert "foundation" in result["weights_used"]
        assert result["weights_used"]["foundation"] == pytest.approx(0.5)


class TestConfigFlag:
    def test_use_hybrid_inference_flag_exists(self):
        """Config flag use_hybrid_inference is referenced in scanner config."""
        from src.scanner.config import ScannerConfig

        cfg = ScannerConfig()
        assert hasattr(cfg, "use_hybrid_inference")
        assert isinstance(cfg.use_hybrid_inference, bool)

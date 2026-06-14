"""Tests for TimesFM Adapter (US-007)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.sota_core.timesfm_adapter import TimesFMAdapter, TimesFMAdapterConfig


class TestTimesFMAdapter:

    def test_init_without_timesfm(self):
        """Should instantiate even when TimesFM is not installed."""
        adapter = TimesFMAdapter(TimesFMAdapterConfig(fallback_on_error=True))
        assert adapter._loaded is False
        assert adapter._fallback is not None

    def test_predict_fallback(self):
        """Predict should return a valid dict when using fallback."""
        adapter = TimesFMAdapter(TimesFMAdapterConfig(fallback_on_error=True))
        df = pd.DataFrame({
            "open": [1.0] * 130,
            "high": [1.01] * 130,
            "low": [0.99] * 130,
            "close": [1.0] * 130,
            "volume": [1000] * 130,
        })
        result = adapter.predict("EUR_USD", df)
        assert "direction" in result
        assert "confidence" in result
        assert "regime" in result
        assert result["direction"] in ("LONG", "SHORT", "HOLD")

    def test_predict_no_data(self):
        adapter = TimesFMAdapter(TimesFMAdapterConfig(fallback_on_error=True))
        result = adapter.predict("EUR_USD", None)
        assert result["direction"] == "HOLD"

    def test_config_defaults(self):
        cfg = TimesFMAdapterConfig()
        assert cfg.seq_len == 128
        assert cfg.horizon == 5
        assert cfg.use_lora is True

    def test_finetune_lora_not_loaded(self):
        adapter = TimesFMAdapter(TimesFMAdapterConfig(fallback_on_error=True))
        ok = adapter.finetune_lora([], [])
        assert ok is False

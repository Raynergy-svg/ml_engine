"""Smoke tests for soak_runner.py."""
import pytest
import tempfile
import os

from src.evaluation.soak_runner import (
    SoakResult,
    save_soak_report,
    load_soak_report,
)


class TestSoakRunner:
    def test_soak_result_defaults(self):
        sr = SoakResult(
            pair="EUR_USD",
            start_time="2026-01-01T00:00:00Z",
            end_time="2026-01-02T00:00:00Z",
            n_bars=96,
        )
        assert sr.recommendation == "UNDECIDED"
        assert sr.confidence == 0.0
        assert sr.MIN_BARS_FOR_DECISION == 500

    def test_roundtrip_save_load(self):
        sr = SoakResult(
            pair="EUR_USD",
            start_time="2026-01-01T00:00:00Z",
            end_time="2026-01-02T00:00:00Z",
            n_bars=96,
        )
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            path = f.name
        try:
            save_soak_report(sr, path)
            loaded = load_soak_report(path)
            assert loaded.pair == "EUR_USD"
            assert loaded.n_bars == 96
            assert loaded.recommendation == "UNDECIDED"
        finally:
            os.unlink(path)

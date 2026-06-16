"""Test HarvestScheduler Dukascopy source integration."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.data.harvest import HarvestConfig, HarvestScheduler


def _make_ticks(n: int = 96) -> pd.DataFrame:
    idx = pd.date_range("2023-01-01 00:00", periods=n, freq="1min", tz="UTC")
    return pd.DataFrame({
        "ask": [1.0700] * n,
        "bid": [1.0699] * n,
        "mid": [1.06995] * n,
        "ask_volume": [1.0] * n,
        "bid_volume": [1.0] * n,
        "spread": [0.0001] * n,
    }, index=idx)


class TestDukascopySource:
    @patch("src.data.harvest.DukascopyHarvester")
    def test_run_dukascopy_creates_parquet(self, MockHarvester: MagicMock, tmp_path: Path):
        mock_h = MagicMock()
        mock_h.download_range.return_value = _make_ticks()
        MockHarvester.return_value = mock_h

        cfg = HarvestConfig(
            pairs=["EUR_USD"],
            source="dukascopy",
            start_date="2023-01-01",
            end_date="2023-01-01",
            harvest_root=tmp_path,
        )
        scheduler = HarvestScheduler(cfg)
        results = scheduler.run(dry_run=False)

        assert results["EUR_USD"]["status"] == "ok"
        out = tmp_path / "EUR_USD_M15_dukascopy.parquet"
        assert out.exists()
        df = pd.read_parquet(out)
        assert {"open", "high", "low", "close", "volume"}.issubset(set(df.columns))

    @patch("src.data.harvest.DukascopyHarvester")
    def test_run_dukascopy_dry_run(self, MockHarvester: MagicMock):
        mock_h = MagicMock()
        mock_h.download_range.return_value = _make_ticks()
        MockHarvester.return_value = mock_h

        cfg = HarvestConfig(
            pairs=["EUR_USD"],
            source="dukascopy",
            start_date="2023-01-01",
            end_date="2023-01-01",
        )
        scheduler = HarvestScheduler(cfg)
        results = scheduler.run(dry_run=True)
        assert results["EUR_USD"]["status"] == "dry_run"

    def test_run_oanda_unchanged(self, tmp_path: Path):
        cfg = HarvestConfig(
            pairs=["EUR_USD"],
            source="oanda",
            harvest_root=tmp_path,
        )
        scheduler = HarvestScheduler(cfg)
        assert scheduler.cfg.source == "oanda"

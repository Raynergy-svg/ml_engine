"""Tests for Dukascopy tick data harvester."""

from __future__ import annotations

import struct
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import lz4.frame
import numpy as np
import pandas as pd
import pytest

from src.data.sources.dukascopy_harvester import (
    DukascopyHarvester,
    _parse_bi5,
    convert_dukascopy_to_harvest_format,
)


def make_bi5_payload(n_ticks: int = 3) -> bytes:
    """Create a valid bi5 payload: lz4-compressed 20-byte tick records."""
    records = []
    for i in range(n_ticks):
        time_delta = i * 1000          # ms
        ask = 107000 + i               # 1.07000 + i/100000
        bid = 106990 + i
        ask_vol = 1_000_000
        bid_vol = 1_000_000
        records.append(struct.pack(">IIIII", time_delta, ask, bid, ask_vol, bid_vol))
    raw = b"".join(records)
    return lz4.frame.compress(raw)


class TestParseBi5:
    def test_empty(self):
        empty = lz4.frame.compress(b"")
        arr = _parse_bi5(empty)
        assert len(arr) == 0

    def test_three_ticks(self):
        payload = make_bi5_payload(3)
        arr = _parse_bi5(payload)
        assert arr.shape == (3, 5)
        assert arr[0, 0] == 0      # time_delta
        assert arr[1, 0] == 1000
        assert arr[0, 1] == 107000 # ask
        assert arr[0, 2] == 106990 # bid


class TestDukascopyHarvester:
    def test_build_url(self):
        h = DukascopyHarvester("EURUSD")
        url = h._build_url(2023, 1, 15, 14)
        assert "EURUSD/2023/01/15/14h_ticks.bi5" in url

    @patch("src.data.sources.dukascopy_harvester.requests.Session.get")
    def test_download_hour_success(self, mock_get: MagicMock):
        payload = make_bi5_payload(5)
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.content = payload
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        h = DukascopyHarvester("EURUSD")
        df = h.download_hour(2023, 1, 15, 14)
        assert df is not None
        assert len(df) == 5
        assert "ask" in df.columns
        assert "bid" in df.columns
        assert "mid" in df.columns
        assert "spread" in df.columns
        # Price scaling sanity
        assert 1.0 < df["ask"].iloc[0] < 2.0

    @patch("src.data.sources.dukascopy_harvester.requests.Session.get")
    def test_download_hour_404(self, mock_get: MagicMock):
        mock_resp = MagicMock()
        mock_resp.status_code = 404
        mock_get.return_value = mock_resp

        h = DukascopyHarvester("EURUSD")
        df = h.download_hour(2023, 1, 15, 14)
        assert df is None

    def test_download_day_concatenates(self):
        h = DukascopyHarvester("EURUSD")
        with patch.object(h, "download_hour", side_effect=[
            pd.DataFrame({"ask": [1.07], "bid": [1.069], "mid": [1.0695], "ask_volume": [1.0], "bid_volume": [1.0], "spread": [0.001]}, index=pd.DatetimeIndex([pd.Timestamp("2023-01-15 00:00:00+00")])),
            None,
            pd.DataFrame({"ask": [1.08], "bid": [1.079], "mid": [1.0795], "ask_volume": [1.0], "bid_volume": [1.0], "spread": [0.001]}, index=pd.DatetimeIndex([pd.Timestamp("2023-01-15 02:00:00+00")])),
        ] + [None] * 21):
            df = h.download_day(2023, 1, 15)
            assert df is not None
            assert len(df) == 2

    def test_download_range_empty_returns_empty_df(self):
        h = DukascopyHarvester("EURUSD")
        with patch.object(h, "download_day", return_value=None):
            df = h.download_range(
                datetime(2023, 1, 1, tzinfo=timezone.utc),
                datetime(2023, 1, 2, tzinfo=timezone.utc),
            )
            assert df.empty

    def test_to_parquet(self, tmp_path: Path):
        h = DukascopyHarvester("EURUSD", save_dir=tmp_path)
        df = pd.DataFrame({
            "ask": [1.07],
            "bid": [1.069],
            "mid": [1.0695],
            "ask_volume": [1.0],
            "bid_volume": [1.0],
            "spread": [0.001],
        }, index=pd.DatetimeIndex([pd.Timestamp("2023-01-15 00:00:00+00")]))
        out = tmp_path / "test.parquet"
        h.to_parquet(df, out)
        assert out.exists()
        read_back = pd.read_parquet(out)
        assert len(read_back) == 1


class TestConvertHarvestFormat:
    def test_basic(self, tmp_path: Path):
        idx = pd.date_range("2023-01-01 00:00", periods=100, freq="1min", tz="UTC")
        df = pd.DataFrame({
            "ask": np.linspace(1.07, 1.08, 100),
            "bid": np.linspace(1.069, 1.079, 100),
            "mid": np.linspace(1.0695, 1.0795, 100),
            "ask_volume": np.ones(100),
            "bid_volume": np.ones(100),
            "spread": np.ones(100) * 0.001,
        }, index=idx)
        out = tmp_path / "harvest.parquet"
        convert_dukascopy_to_harvest_format("EUR_USD", df, out)
        assert out.exists()
        ohlcv = pd.read_parquet(out)
        assert {"open", "high", "low", "close", "volume"}.issubset(set(ohlcv.columns))


class TestSmoke:
    """Real network smoke test — disabled by default to avoid flakiness in CI."""
    @pytest.mark.skip(reason="Live network call; run manually with pytest -k smoke -v")
    def test_live_single_hour(self):
        h = DukascopyHarvester("EURUSD")
        # Pick a weekday a few weeks ago to maximise chance of data existing
        df = h.download_hour(2024, 5, 15, 10)
        if df is None:
            pytest.skip("Dukascopy returned no data for chosen hour (weekend/holiday)")
        assert len(df) > 0
        assert {"ask", "bid", "mid"}.issubset(set(df.columns))


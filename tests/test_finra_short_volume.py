"""No-mock tests for the FINRA short-volume loader (bronze-layer PIT parser).

Everything here uses REAL tiny synthetic FINRA-format files written to
``tmp_path`` — no ``unittest.mock``, no network (per ``.claude/rules/
improvement.md`` No-Mock Rule). Mirrors the price-shuffle PIT pattern in
``tests/test_research_harness.py::test_causality_positions_depend_only_on_past``:
shuffling/deleting FUTURE short-volume files must not change PAST z-scores or
panel rows.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.equity.research.finra_short_volume_loader import (
    FinraShortVolumeContractError,
    load_short_volume_ratio_panel,
)

TICKERS = ["AAA", "BBB", "CCC"]


def _write_raw_file(
    cache_dir: Path,
    ymd: str,
    rows: list[tuple[str, float, float, float, str]],
    *,
    trailer: bool = True,
) -> Path:
    """Write one real pipe-delimited CNMS-format file (no mocks)."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    lines = ["Date|Symbol|ShortVolume|ShortExemptVolume|TotalVolume|Market"]
    for symbol, short_vol, short_exempt, total_vol, market in rows:
        lines.append(
            f"{ymd}|{symbol}|{short_vol}|{short_exempt}|{total_vol}|{market}"
        )
    if trailer:
        lines.append(str(len(rows)))  # FINRA's real bare row-count trailer
    path = cache_dir / f"CNMSshvol{ymd}.txt"
    path.write_text("\n".join(lines) + "\n")
    return path


def _build_world(cache_dir: Path, days: list[str]) -> None:
    """Deterministic synthetic short-volume history for 3 tickers over N days."""
    for i, ymd in enumerate(days):
        rows = [
            ("AAA", 100.0 + i, 0.0, 500.0, "B,Q,N"),
            ("BBB", 200.0 + 2 * i, 5.0, 500.0, "B,Q,N"),
            ("CCC", 50.0 + i, 0.0, 500.0, "B,Q,N"),
        ]
        _write_raw_file(cache_dir, ymd, rows)


DAYS = [
    "20190102", "20190103", "20190104", "20190107", "20190108",
    "20190109", "20190110",
]


def test_panel_basic_shape_and_values(tmp_path: Path):
    cache_dir = tmp_path / "raw"
    _build_world(cache_dir, DAYS)

    panel = load_short_volume_ratio_panel(TICKERS, cache_dir=cache_dir)

    assert list(panel.columns) == sorted(TICKERS)
    assert len(panel) == len(DAYS)
    assert isinstance(panel.index, pd.DatetimeIndex)
    assert panel.index.tz is not None

    # Day 0 (2019-01-02): AAA short_volume=100, total=500 -> ratio 0.2
    first_day = panel.index[0]
    assert panel.loc[first_day, "AAA"] == pytest.approx(100.0 / 500.0)
    assert panel.loc[first_day, "BBB"] == pytest.approx(200.0 / 500.0)


def test_trailer_line_does_not_corrupt_or_leak_as_pit_violation(tmp_path: Path):
    """The bare row-count trailer FINRA appends must be stripped, not raise."""
    cache_dir = tmp_path / "raw"
    _build_world(cache_dir, DAYS[:1])
    # Should not raise FinraShortVolumeContractError on the trailer line.
    panel = load_short_volume_ratio_panel(TICKERS, cache_dir=cache_dir)
    assert len(panel) == 1


def test_missing_ticker_on_a_day_is_nan_not_imputed(tmp_path: Path):
    """A ticker absent from one day's file must be NaN, never forward-filled."""
    cache_dir = tmp_path / "raw"
    _write_raw_file(
        cache_dir, "20190102",
        [("AAA", 100.0, 0.0, 500.0, "B,Q,N")],
    )
    _write_raw_file(
        cache_dir, "20190103",
        [("AAA", 110.0, 0.0, 500.0, "B,Q,N"), ("BBB", 50.0, 0.0, 500.0, "B,Q,N")],
    )
    panel = load_short_volume_ratio_panel(TICKERS, cache_dir=cache_dir)
    day0, day1 = panel.index[0], panel.index[1]
    assert pd.isna(panel.loc[day0, "BBB"])
    assert panel.loc[day1, "BBB"] == pytest.approx(50.0 / 500.0)
    # No back-fill: day0's BBB must still be NaN even though day1 has data.
    assert pd.isna(panel.loc[day0, "BBB"])


def test_duplicate_date_symbol_rows_raise_not_silently_aggregated(tmp_path: Path):
    """Violating the one-row-per-(date,symbol) assumption must fail loud."""
    cache_dir = tmp_path / "raw"
    cache_dir.mkdir(parents=True)
    lines = [
        "Date|Symbol|ShortVolume|ShortExemptVolume|TotalVolume|Market",
        "20190102|AAA|100|0|500|B",
        "20190102|AAA|50|0|200|Q",  # duplicate row for the SAME (date, symbol)
    ]
    (cache_dir / "CNMSshvol20190102.txt").write_text("\n".join(lines) + "\n")
    with pytest.raises(FinraShortVolumeContractError, match="duplicate"):
        load_short_volume_ratio_panel(TICKERS, cache_dir=cache_dir)


def test_embedded_date_mismatch_vs_filename_raises(tmp_path: Path):
    """A file whose embedded Date disagrees with its own filename must refuse."""
    cache_dir = tmp_path / "raw"
    cache_dir.mkdir(parents=True)
    lines = [
        "Date|Symbol|ShortVolume|ShortExemptVolume|TotalVolume|Market",
        "20190103|AAA|100|0|500|B,Q,N",  # wrong date embedded vs filename below
    ]
    (cache_dir / "CNMSshvol20190102.txt").write_text("\n".join(lines) + "\n")
    with pytest.raises(FinraShortVolumeContractError, match="PIT contract violation"):
        load_short_volume_ratio_panel(TICKERS, cache_dir=cache_dir)


def test_no_cached_files_raises(tmp_path: Path):
    cache_dir = tmp_path / "raw"
    cache_dir.mkdir(parents=True)
    with pytest.raises(FinraShortVolumeContractError, match="no cached FINRA files"):
        load_short_volume_ratio_panel(TICKERS, cache_dir=cache_dir)


def test_future_files_deleted_or_shuffled_does_not_change_past_panel(tmp_path: Path):
    """PIT regression: deleting/shuffling FUTURE files must not change PAST rows.

    Mirrors tests/test_research_harness.py::test_causality_positions_depend_only_on_past
    — build the full panel, then rebuild with every day strictly after a cut
    date deleted (simulating "future data never arrived") and separately with
    future days' VALUES shuffled/scrambled. Past rows must be byte-identical
    in both cases, since each date's row is parsed only from that date's own
    file with no cross-date fill.
    """
    cache_dir = tmp_path / "raw"
    _build_world(cache_dir, DAYS)

    panel_full = load_short_volume_ratio_panel(TICKERS, cache_dir=cache_dir)

    cut_idx = 3  # "20190107" — split point
    cut_ymd = DAYS[cut_idx]
    cut_ts = pd.Timestamp(cut_ymd, tz="UTC")

    # Case 1: delete every file strictly after the cut date.
    deleted_dir = tmp_path / "raw_deleted"
    _build_world(deleted_dir, DAYS[: cut_idx + 1])
    panel_deleted = load_short_volume_ratio_panel(TICKERS, cache_dir=deleted_dir)

    past_full = panel_full.loc[panel_full.index <= cut_ts]
    past_deleted = panel_deleted.loc[panel_deleted.index <= cut_ts]
    pd.testing.assert_frame_equal(past_full, past_deleted)

    # Case 2: scramble (multiply + reverse) every day strictly AFTER the cut,
    # rewriting those raw files with different values, and confirm past rows
    # in the freshly-reloaded panel are unchanged.
    scrambled_dir = tmp_path / "raw_scrambled"
    _build_world(scrambled_dir, DAYS)
    future_days = DAYS[cut_idx + 1:]
    for j, ymd in enumerate(reversed(future_days)):
        rows = [
            ("AAA", 999.0 + j * 7, 0.0, 500.0, "B,Q,N"),
            ("BBB", 888.0 + j * 11, 0.0, 500.0, "B,Q,N"),
            ("CCC", 777.0 + j * 13, 0.0, 500.0, "B,Q,N"),
        ]
        _write_raw_file(scrambled_dir, ymd, rows)
    panel_scrambled = load_short_volume_ratio_panel(TICKERS, cache_dir=scrambled_dir)

    past_scrambled = panel_scrambled.loc[panel_scrambled.index <= cut_ts]
    pd.testing.assert_frame_equal(past_full, past_scrambled)
    # Sanity: the future actually did change (scrambling wasn't a no-op).
    future_full = panel_full.loc[panel_full.index > cut_ts]
    future_scrambled = panel_scrambled.loc[panel_scrambled.index > cut_ts]
    assert not future_full.equals(future_scrambled)

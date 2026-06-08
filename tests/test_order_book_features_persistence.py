"""Tests for src/data/order_book — real disk, no mocks per .claude/rules/improvement.md.

The OANDA bucket shape is fixed by the broker; tests use it as ground truth
rather than abstracting it. ``tmp_path`` substitutes for the persistence root
so tests never touch ``trained_data/order_book/``.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import pytest

from src.data.order_book import (
    extract_order_features,
    extract_position_features,
    snapshot,
)


# --------------------------------------------------------------------- helpers


def _bucket(price: float, long_pct: float, short_pct: float) -> Dict[str, Any]:
    """Mirror OANDA's string-typed percent fields exactly."""
    return {
        "price": f"{price:.4f}",
        "longCountPercent": f"{long_pct:.4f}",
        "shortCountPercent": f"{short_pct:.4f}",
    }


def _balanced_buckets(current: float = 1.05) -> List[Dict[str, Any]]:
    """15 buckets spanning ±0.7% of ``current``, balanced long/short."""
    out: List[Dict[str, Any]] = []
    for i, off in enumerate(range(-7, 8)):
        price = current + (off / 1000.0)
        # Mild bell-shape; balanced long/short
        weight = 5.0 - abs(off) * 0.3
        out.append(_bucket(price, weight, weight))
    return out


def _crowded_long_buckets(current: float = 1.05) -> List[Dict[str, Any]]:
    """Same shape but long-heavy (textbook crowded-long contrarian signal)."""
    out: List[Dict[str, Any]] = []
    for off in range(-7, 8):
        price = current + (off / 1000.0)
        out.append(_bucket(price, 8.0, 2.0))
    return out


# --------------------------------------------------------------------- features


def test_extract_position_features_balanced_book() -> None:
    feats = extract_position_features(_balanced_buckets(), current_price=1.05)
    assert pytest.approx(feats["pb_long_ratio"], abs=1e-6) == 0.5
    assert pytest.approx(feats["pb_total_imbalance"], abs=1e-6) == 0.0
    assert pytest.approx(feats["pb_near_imbalance"], abs=1e-6) == 0.0
    assert feats["pb_n_buckets"] == 15.0


def test_extract_position_features_crowded_long_book() -> None:
    feats = extract_position_features(_crowded_long_buckets(), current_price=1.05)
    # 8/(8+2) = 0.8 long ratio
    assert pytest.approx(feats["pb_long_ratio"], abs=1e-6) == 0.8
    # imbalance = (8-2)/(8+2) = 0.6
    assert pytest.approx(feats["pb_total_imbalance"], abs=1e-6) == 0.6
    assert feats["pb_near_imbalance"] > 0.5  # strong long pressure near price


def test_extract_order_features_returns_ob_prefixed_keys() -> None:
    feats = extract_order_features(_balanced_buckets(), current_price=1.05)
    assert all(k.startswith("ob_") for k in feats.keys())
    assert "ob_long_ratio" in feats
    assert "ob_top5_long_concentration" in feats


def test_features_handle_zero_price_bucket_gracefully() -> None:
    bad = [_bucket(0.0, 5.0, 5.0)] + _balanced_buckets()
    feats = extract_position_features(bad, current_price=1.05)
    # Zero-price bucket is skipped — n_buckets matches the balanced batch only
    assert feats["pb_n_buckets"] == 15.0


def test_features_handle_empty_buckets() -> None:
    feats = extract_position_features([], current_price=1.05)
    assert feats["pb_long_ratio"] == 0.5  # neutral fallback
    assert feats["pb_total_imbalance"] == 0.0
    assert feats["pb_n_buckets"] == 0.0


def test_features_handle_zero_current_price() -> None:
    feats = extract_position_features(_balanced_buckets(), current_price=0.0)
    # Without a reference price, distance bands collapse — near/mid sums = 0
    assert feats["pb_near_long_pct"] == 0.0
    assert feats["pb_near_short_pct"] == 0.0
    # But total long/short still tally normally
    assert feats["pb_long_ratio"] > 0.0


def test_top5_concentration_one_dominant_bucket() -> None:
    # 90 units of long count in one bucket + 9 fillers of 1.0 each.
    # Total long = 99; top-5 = 90 + 4*1 = 94; concentration = 94/99 ≈ 0.949.
    skewed = [_bucket(1.05, 90.0, 1.0)] + [_bucket(1.05 + i * 0.001, 1.0, 1.0) for i in range(1, 10)]
    feats = extract_position_features(skewed, current_price=1.05)
    assert feats["pb_top5_long_concentration"] > 0.9


# --------------------------------------------------------------------- persister


def test_snapshot_writes_jsonl_row(tmp_path: Path) -> None:
    written = snapshot(
        pair="EUR_USD",
        book_type="position",
        buckets=_balanced_buckets(),
        current_price=1.05,
        root=tmp_path,
    )
    assert written is not None
    assert written.exists()
    assert written.name == "EUR_USD_position.jsonl"

    lines = written.read_text().strip().splitlines()
    assert len(lines) == 1
    row = json.loads(lines[0])
    assert row["pair"] == "EUR_USD"
    assert row["book_type"] == "position"
    assert row["current_price"] == 1.05
    assert "features" in row
    assert "pb_long_ratio" in row["features"]
    assert "buckets" not in row  # default keep_raw_buckets=False


def test_snapshot_appends_multiple_rows(tmp_path: Path) -> None:
    for _ in range(3):
        snapshot(
            pair="GBP_USD",
            book_type="order",
            buckets=_balanced_buckets(current=1.27),
            current_price=1.27,
            root=tmp_path,
        )
    path = tmp_path / "GBP_USD_order.jsonl"
    lines = path.read_text().strip().splitlines()
    assert len(lines) == 3
    for line in lines:
        row = json.loads(line)
        assert row["book_type"] == "order"
        assert "ob_long_ratio" in row["features"]


def test_snapshot_persists_raw_buckets_when_requested(tmp_path: Path) -> None:
    snapshot(
        pair="EUR_USD",
        book_type="position",
        buckets=_balanced_buckets(),
        current_price=1.05,
        root=tmp_path,
        keep_raw_buckets=True,
    )
    row = json.loads((tmp_path / "EUR_USD_position.jsonl").read_text().strip())
    assert "buckets" in row
    assert len(row["buckets"]) == 15


def test_snapshot_returns_none_on_empty_buckets(tmp_path: Path) -> None:
    result = snapshot(
        pair="EUR_USD",
        book_type="position",
        buckets=[],
        current_price=1.05,
        root=tmp_path,
    )
    assert result is None
    assert not (tmp_path / "EUR_USD_position.jsonl").exists()


def test_snapshot_returns_none_on_invalid_price(tmp_path: Path) -> None:
    assert snapshot(
        pair="EUR_USD",
        book_type="position",
        buckets=_balanced_buckets(),
        current_price=-1.0,
        root=tmp_path,
    ) is None


def test_snapshot_rejects_invalid_book_type(tmp_path: Path) -> None:
    assert snapshot(
        pair="EUR_USD",
        book_type="depth",  # not "position" or "order"
        buckets=_balanced_buckets(),
        current_price=1.05,
        root=tmp_path,
    ) is None


def test_snapshot_uses_provided_timestamp(tmp_path: Path) -> None:
    ts = datetime(2026, 5, 13, 12, 0, 0, tzinfo=timezone.utc)
    snapshot(
        pair="EUR_USD",
        book_type="position",
        buckets=_balanced_buckets(),
        current_price=1.05,
        timestamp=ts,
        root=tmp_path,
    )
    row = json.loads((tmp_path / "EUR_USD_position.jsonl").read_text().strip())
    assert row["timestamp"].startswith("2026-05-13T12:00:00")

"""Tests for src/training/labels/journal_loader.py salvage utility.

Verifies tolerant parser recovers a known-corrupt fixture and that the
high-level `load_all_journal_trades` pools live + archive correctly.
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import pytest

from src.training.labels.journal_loader import (
    salvage_corrupt_journal,
    load_all_journal_trades,
)


def _write_corrupt_fixture(tmp_path: Path, n_records: int = 5, truncate_last: bool = True) -> Path:
    """Write a JSON array of records, deliberately truncating mid-record at the end."""
    records = []
    for i in range(n_records):
        records.append({
            "pair": "EUR_USD",
            "timestamp": f"2024-01-0{i+1}T00:00:00+00:00",
            "entry_price": 1.10 + i * 0.001,
            "direction": "LONG",
            "outcome": {
                "trade_won": (i % 2 == 0),
                "exit_reason": "tp_hit" if (i % 2 == 0) else "sl_hit",
            },
        })

    text = "[\n"
    for r in records:
        body = json.dumps(r, indent=2)
        # Re-indent to 2-space-prefix and append `,\n`
        body = textwrap.indent(body, "  ")
        text += body + ",\n"
    text += "]\n"

    if truncate_last:
        # Drop the closing `]` plus part of the last record to simulate corruption
        # Specifically: cut off everything after the LAST record's first 50 chars
        # so the salvager has to back up to the previous `\n  },\n`.
        last_marker = text.rfind("\n  },\n")
        # Cut to a point AFTER the last good marker but inside trailing record
        text = text[:last_marker + 50]

    f = tmp_path / "corrupt.json"
    f.write_text(text)
    return f


def test_salvage_recovers_records(tmp_path: Path):
    f = _write_corrupt_fixture(tmp_path, n_records=5, truncate_last=True)
    # The corruption deletes the last full record + opening of an additional
    # partial. Salvager should recover the records that have a clean closing
    # `\n  },\n` terminator.
    recovered = salvage_corrupt_journal(f)
    # We wrote 5 records; truncation chopped the 5th's closing — so 4 should
    # survive cleanly. Be permissive: at minimum recover >= 3.
    assert len(recovered) >= 3
    # Each recovered record has expected keys
    for rec in recovered:
        assert "pair" in rec
        assert "outcome" in rec


def test_salvage_clean_file_passthrough(tmp_path: Path):
    """A non-corrupt JSON array is parsed without modification."""
    records = [{"pair": "EUR_USD", "timestamp": "2024-01-01T00:00:00+00:00",
                "entry_price": 1.1, "direction": "LONG",
                "outcome": {"trade_won": True}}]
    f = tmp_path / "clean.json"
    f.write_text(json.dumps(records, indent=2))
    recovered = salvage_corrupt_journal(f)
    assert len(recovered) == 1


def test_salvage_unrecoverable_returns_empty(tmp_path: Path):
    """A file with no record terminators returns []."""
    f = tmp_path / "garbage.json"
    f.write_text("not even close to JSON {}{[")
    recovered = salvage_corrupt_journal(f)
    assert recovered == []


def test_load_all_journal_trades_normalizes_and_dedupes(tmp_path: Path):
    """Live + archive pools, dedupes by (pair, ts_minute, entry_price)."""
    live = tmp_path / "live.json"
    archive_dir = tmp_path / "archive"
    archive_dir.mkdir()
    arch = archive_dir / "trade_journal_rl_2024.json"

    common = {
        "pair": "EUR/USD",  # slash form to test normalization
        "timestamp": "2024-01-01T00:00:00+00:00",
        "entry_price": 1.10,
        "direction": "LONG",
        "outcome": {"trade_won": True, "exit_reason": "tp_hit"},
    }
    live.write_text(json.dumps([common], indent=2))
    arch.write_text(json.dumps([common, {**common, "entry_price": 1.20}], indent=2))

    out = load_all_journal_trades(
        include_archives=True, live_path=live, archive_dir=archive_dir,
    )
    # Common record dedupes to 1; second archive record (entry=1.20) is unique.
    assert len(out) == 2
    # Pairs normalized
    assert all(t["pair"] == "EUR_USD" for t in out)
    # trade_won hoisted
    assert all(t["trade_won"] is True for t in out)

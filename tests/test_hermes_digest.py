"""Hermes digest writer + rotation."""
from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from src.scanner.automation.hermes_digest import (
    DigestEntry,
    HermesDigest,
)


UTC = timezone.utc


def test_first_entry_creates_file_with_header(tmp_path: Path):
    digest_path = tmp_path / "hermes_watchdog.md"
    archive_dir = tmp_path / ".archive"
    d = HermesDigest(digest_path=digest_path, archive_dir=archive_dir, hard_cap=8000)
    now = datetime(2026, 5, 12, 7, 0, 0, tzinfo=UTC)
    d.append(DigestEntry(at=now, kind="brief", body=["halted: true · mode: live"]))
    text = digest_path.read_text()
    assert text.startswith("# Hermes Watchdog Digest")
    assert "## 2026-05-12" in text
    assert "### 07:00Z — brief" in text
    assert "halted: true · mode: live" in text


def test_same_day_second_entry_does_not_repeat_day_header(tmp_path: Path):
    digest_path = tmp_path / "hermes_watchdog.md"
    archive_dir = tmp_path / ".archive"
    d = HermesDigest(digest_path=digest_path, archive_dir=archive_dir, hard_cap=8000)
    d.append(DigestEntry(at=datetime(2026, 5, 12, 7, 0, tzinfo=UTC),
                         kind="brief", body=["a"]))
    d.append(DigestEntry(at=datetime(2026, 5, 12, 14, 30, tzinfo=UTC),
                         kind="watch", body=["b"]))
    text = digest_path.read_text()
    assert text.count("## 2026-05-12") == 1
    assert "### 07:00Z — brief" in text
    assert "### 14:30Z — watch" in text


def test_new_day_inserts_new_day_header(tmp_path: Path):
    digest_path = tmp_path / "hermes_watchdog.md"
    archive_dir = tmp_path / ".archive"
    d = HermesDigest(digest_path=digest_path, archive_dir=archive_dir, hard_cap=8000)
    d.append(DigestEntry(at=datetime(2026, 5, 12, 23, 30, tzinfo=UTC),
                         kind="watch", body=["a"]))
    d.append(DigestEntry(at=datetime(2026, 5, 13, 7, 0, tzinfo=UTC),
                         kind="brief", body=["b"]))
    text = digest_path.read_text()
    assert "## 2026-05-12" in text
    assert "## 2026-05-13" in text


def test_rotation_when_size_exceeds_1_5x_hard_cap(tmp_path: Path):
    digest_path = tmp_path / "hermes_watchdog.md"
    archive_dir = tmp_path / ".archive"
    # Pre-populate over 1.5 * hard_cap=8000 → > 12000 chars
    digest_path.write_text("# Hermes Watchdog Digest\n\n" + "x" * 13000)
    d = HermesDigest(digest_path=digest_path, archive_dir=archive_dir, hard_cap=8000)
    d.append(DigestEntry(at=datetime(2026, 5, 12, 7, 0, tzinfo=UTC),
                         kind="brief", body=["fresh"]))
    # Original should be rotated into the archive.
    archived = list(archive_dir.glob("hermes_watchdog_*.md"))
    assert len(archived) == 1
    # Fresh file should contain only the new entry + header.
    text = digest_path.read_text()
    assert len(text) < 1000
    assert "fresh" in text
    assert "## 2026-05-12" in text


def test_rotation_filename_uses_iso_year_week(tmp_path: Path):
    digest_path = tmp_path / "hermes_watchdog.md"
    archive_dir = tmp_path / ".archive"
    digest_path.write_text("# Hermes Watchdog Digest\n\n" + "x" * 13000)
    d = HermesDigest(digest_path=digest_path, archive_dir=archive_dir, hard_cap=8000)
    # 2026-05-12 is a Tuesday in ISO week 20 of 2026.
    d.append(DigestEntry(at=datetime(2026, 5, 12, 7, 0, tzinfo=UTC),
                         kind="brief", body=["fresh"]))
    archived = list(archive_dir.glob("hermes_watchdog_*.md"))
    assert len(archived) == 1
    assert re.match(r"hermes_watchdog_\d{4}_\d{2}\.md", archived[0].name)


def test_body_lines_render_as_bullets(tmp_path: Path):
    digest_path = tmp_path / "hermes_watchdog.md"
    archive_dir = tmp_path / ".archive"
    d = HermesDigest(digest_path=digest_path, archive_dir=archive_dir, hard_cap=8000)
    d.append(DigestEntry(at=datetime(2026, 5, 12, 14, 30, tzinfo=UTC),
                         kind="watch",
                         body=["trigger: consecutive_losses=3",
                               "recent: EUR_USD -$54"]))
    text = digest_path.read_text()
    assert "- trigger: consecutive_losses=3" in text
    assert "- recent: EUR_USD -$54" in text


def test_atomic_write_survives_many_appends(tmp_path: Path):
    """Verifies repeated appends produce a valid file readable by our own scanner."""
    digest_path = tmp_path / "hermes_watchdog.md"
    archive_dir = tmp_path / ".archive"
    d = HermesDigest(digest_path=digest_path, archive_dir=archive_dir, hard_cap=8000)
    for i in range(10):
        d.append(DigestEntry(at=datetime(2026, 5, 12, 7, i, tzinfo=UTC),
                             kind="watch", body=[f"event #{i}"]))
    text = digest_path.read_text(encoding="utf-8")
    assert "event #0" in text
    assert "event #9" in text
    # Day-header scanner survives the file:
    d2 = HermesDigest(digest_path=digest_path, archive_dir=archive_dir, hard_cap=8000)
    assert "2026-05-12" in d2._last_day_header_in_file()

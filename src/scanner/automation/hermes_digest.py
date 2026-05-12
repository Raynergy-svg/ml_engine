"""Hermes watchdog + brief digest writer.

Append-only markdown writer for .claude/brain/hermes_watchdog.md with
atomic temp+rename writes, day-header de-duplication, and weekly rotation
into .archive/ when the file exceeds hard_cap * 1.5.

Buddy policies honored:
- Atomic writes: temp file + os.replace (matches safe_json_write pattern).
- Specific exception types (OSError) — no bare except.
- Append-only — never edits past entries; rotation moves the whole file.
- Rotation filename uses ISO year-week (datetime.isocalendar()) so naming
  is stable across daylight savings transitions and operator timezones.
"""
from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List

logger = logging.getLogger(__name__)


_DAY_HEADER_RE = re.compile(r"^## (\d{4}-\d{2}-\d{2})\s*$", re.MULTILINE)


@dataclass(frozen=True)
class DigestEntry:
    """One entry in the watchdog digest."""

    at: datetime          # tz-aware UTC
    kind: str             # "brief" | "watch" | "silent"
    body: List[str]       # rendered as bullets


class HermesDigest:
    """Append-only writer + rotation for .claude/brain/hermes_watchdog.md.

    Args:
        digest_path: live digest file (created on first append)
        archive_dir: where rotations land
        hard_cap: from brain_caps[hermes_watchdog.md].hard_cap (8000 typ).
            Rotation triggers at 1.5x hard_cap.
    """

    _FILE_HEADER = (
        "# Hermes Watchdog Digest\n"
        "\n"
        "> Auto-written by `hermes_watchdog` (every 30 min) and "
        "`hermes_daily_brief` (daily 07:00 UTC).\n"
        "> Operator-readable. Rotated weekly into `.archive/` when over cap.\n"
    )

    def __init__(
        self,
        *,
        digest_path: Path,
        archive_dir: Path,
        hard_cap: int,
    ) -> None:
        self._digest_path = Path(digest_path)
        self._archive_dir = Path(archive_dir)
        self._hard_cap = int(hard_cap)
        self._rotation_threshold = int(hard_cap * 1.5)

    # ── public API ─────────────────────────────────────────────────────

    def append(self, entry: DigestEntry) -> None:
        """Atomic append. Rotates first if the existing file is over threshold."""
        self._maybe_rotate(entry.at)

        existing = self._read_existing()
        new_block = self._render_entry(entry, existing)
        out = existing + new_block

        self._atomic_write(out)

    # ── internals ──────────────────────────────────────────────────────

    def _read_existing(self) -> str:
        if not self._digest_path.exists():
            return self._FILE_HEADER
        try:
            return self._digest_path.read_text(encoding="utf-8")
        except OSError as e:
            logger.warning("HermesDigest: read failed %s: %s", self._digest_path, e)
            return self._FILE_HEADER

    def _maybe_rotate(self, now: datetime) -> None:
        try:
            size = self._digest_path.stat().st_size
        except FileNotFoundError:
            return
        except OSError as e:
            logger.warning("HermesDigest: stat failed: %s", e)
            return
        if size < self._rotation_threshold:
            return
        # Rotate.
        iso_year, iso_week, _ = now.isocalendar()
        archive_name = f"hermes_watchdog_{iso_year}_{iso_week:02d}.md"
        try:
            self._archive_dir.mkdir(parents=True, exist_ok=True)
            target = self._archive_dir / archive_name
            # If the target already exists (multiple rotations in one week),
            # append a numeric suffix.
            if target.exists():
                i = 2
                while True:
                    alt = self._archive_dir / f"hermes_watchdog_{iso_year}_{iso_week:02d}_{i}.md"
                    if not alt.exists():
                        target = alt
                        break
                    i += 1
            self._digest_path.replace(target)
            logger.info("HermesDigest: rotated %s -> %s", self._digest_path, target)
        except OSError as e:
            logger.warning("HermesDigest: rotation failed: %s", e)

    def _render_entry(self, entry: DigestEntry, existing: str) -> str:
        date_str = entry.at.strftime("%Y-%m-%d")
        time_str = entry.at.strftime("%H:%MZ")
        last_day = self._last_day_header(existing)
        parts: List[str] = []
        if last_day != date_str:
            parts.append(f"\n## {date_str}\n")
        parts.append(f"\n### {time_str} — {entry.kind}\n")
        for line in entry.body:
            parts.append(f"- {line}\n")
        return "".join(parts)

    def _atomic_write(self, content: str) -> None:
        self._digest_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._digest_path.with_suffix(self._digest_path.suffix + ".tmp")
        try:
            tmp.write_text(content, encoding="utf-8")
            os.replace(tmp, self._digest_path)
        except OSError as e:
            logger.error("HermesDigest: atomic write failed: %s", e)
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                pass
            raise

    def _last_day_header(self, text: str) -> str:
        matches = list(_DAY_HEADER_RE.finditer(text))
        return matches[-1].group(1) if matches else ""

    def _last_day_header_in_file(self) -> str:
        return self._last_day_header(self._read_existing())

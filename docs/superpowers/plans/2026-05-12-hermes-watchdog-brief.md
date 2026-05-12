# Hermes Watchdog + Daily Brief Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build two in-process scheduled jobs that write structured observations to `.claude/brain/hermes_watchdog.md` — a 30-minute watchdog that surfaces unacknowledged alerts + heartbeat staleness + scheduler failures, and a daily 07:00 UTC brief composing a fixed-schema 24h summary. Plus the supporting state-persistence helper, digest writer, brain_caps entry, and default_jobs additions.

**Architecture:** Five new modules under `src/scanner/automation/` (hermes_state, hermes_digest, hermes_watchdog, hermes_daily_brief — plus brain_caps + scheduled_jobs extensions). Both scripts are subprocess entry points fired by the T1 `ScheduledJobsRegistry`; Claude-free in the hot path. All file I/O uses the cloud-branch `safe_json_read` / `safe_json_write` atomic pattern. Markdown digest writes use temp+rename. Tests are real-class + real-disk via `tmp_path` per CLAUDE.md no-mocks rule.

**Tech Stack:** Python 3.11, dataclasses, pytest, `safe_json_read`/`safe_json_write` (from `src.scanner.automation.safe_json`), `brain_caps` (from `src.scanner.automation.brain_caps`), `ScheduledJobsRegistry` (from `src.scanner.automation.scheduled_jobs`).

**Buddy's policies honored** (per CLAUDE.md + `.claude/rules/improvement.md`):
- NO MOCK CODE — all tests use real classes against real disk via `tmp_path`.
- Atomic writes everywhere — `safe_json_write` for JSON, temp+rename for markdown.
- JSON reads wrapped in try/except with graceful fallback.
- Specific exception types — never bare `except:`.
- Config keys validated against `ScannerConfig` field names before use.
- Claude-free runtime — both scripts run as subprocesses, no LLM in the loop.
- Append-only writes to the digest; rotation never mutates past entries.
- Honesty / verification — every recon claim is grounded in actual file inspection.

**Prerequisite:** Tier 1 (T1–T6) must be live on main (DONE — commit `047f0e3`). Specifically T1's `ScheduledJobsRegistry` with `JobRuntimeState.last_status/last_error/state` fields, and the post-merge `safe_json_read/write` + `brain_caps` + `default_jobs()` surfaces.

---

## Task 1: Foundation — brain_caps entry + default_jobs additions

**Files:**
- Modify: `src/scanner/automation/brain_caps.py` (add one entry)
- Modify: `src/scanner/automation/scheduled_jobs.py` (extend `default_jobs()`)
- Create: `tests/test_hermes_jobs_default.py`

This task lands first because the watchdog and brief scripts (Tasks 4 + 5) need the brain_caps cap entry to validate their writes, and the `default_jobs()` change makes the two new jobs discoverable to fresh installs without operator intervention.

- [ ] **Step 1: Write failing test for default_jobs() including hermes entries**

Create `tests/test_hermes_jobs_default.py`:

```python
"""Hermes watchdog: default_jobs() includes the 30-min watchdog and daily brief."""
from __future__ import annotations

from src.scanner.automation.scheduled_jobs import default_jobs
from src.scanner.automation.brain_caps import caps


def test_default_jobs_includes_hermes_watchdog():
    jobs = {j.job_id: j for j in default_jobs()}
    assert "hermes_watchdog" in jobs
    j = jobs["hermes_watchdog"]
    assert j.schedule == "every_30_minutes"
    assert j.enabled is True
    assert "hermes_watchdog" in j.command


def test_default_jobs_includes_hermes_daily_brief():
    jobs = {j.job_id: j for j in default_jobs()}
    assert "hermes_daily_brief" in jobs
    j = jobs["hermes_daily_brief"]
    assert j.schedule == "daily_07:00"
    assert j.enabled is True
    assert "hermes_daily_brief" in j.command


def test_default_jobs_preserves_homework_weekly():
    """Don't drop the existing default."""
    jobs = {j.job_id: j for j in default_jobs()}
    assert "homework_weekly" in jobs


def test_brain_caps_includes_hermes_watchdog():
    c = caps()
    assert "hermes_watchdog.md" in c
    hard_cap, warn_ratio = c["hermes_watchdog.md"]
    assert hard_cap == 8000
    assert warn_ratio == 1.15
```

- [ ] **Step 2: Run — confirm failure**

Run: `pytest tests/test_hermes_jobs_default.py -v`
Expected: 3 failures on the hermes_* assertions (KeyError or `in` check fails); the brain_caps test fails with `KeyError`.

- [ ] **Step 3: Extend default_jobs()**

In `src/scanner/automation/scheduled_jobs.py`, modify the `default_jobs()` function:

```python
def default_jobs() -> list[JobConfig]:
    """Built-in jobs surfaced when `.claude/jobs.json` is missing.

    All defaults are intentional — operator can disable via `.claude/jobs.json`
    or pause via the F9 Jobs screen (T1) at runtime.
    """
    return [
        JobConfig(
            job_id="homework_weekly",
            name="Weekly homework batch",
            schedule="weekly_SUN_02:00",
            command="python buddy_scanner.py homework --generate-batch --last 100",
            enabled=False,
            description="Regenerate the last 100 trade homework entries every Sunday 02:00 UTC.",
        ),
        JobConfig(
            job_id="hermes_watchdog",
            name="Hermes Watchdog — anomaly detection",
            schedule="every_30_minutes",
            command="python -m src.scanner.automation.hermes_watchdog",
            enabled=True,
            description=(
                "Reads scanner state files every 30 min; writes structured "
                "observations to .claude/brain/hermes_watchdog.md when "
                "anomalies (active alerts, heartbeat staleness, job failures) "
                "detected."
            ),
        ),
        JobConfig(
            job_id="hermes_daily_brief",
            name="Hermes Daily Brief — 24h structured summary",
            schedule="daily_07:00",
            command="python -m src.scanner.automation.hermes_daily_brief",
            enabled=True,
            description=(
                "Composes daily fixed-schema summary; appends to "
                ".claude/brain/hermes_watchdog.md at 07:00 UTC."
            ),
        ),
    ]
```

- [ ] **Step 4: Extend brain_caps**

In `src/scanner/automation/brain_caps.py`, modify the `_DEFAULT_CAPS` dict:

```python
_DEFAULT_CAPS: dict[str, tuple[int, float]] = {
    "briefing.md":         (3_000, 1.20),
    "session_handoff.md":  (2_000, 1.20),
    "open_questions.md":   (1_500, 1.20),
    "strategic_log.md":    (8_000, 1.15),
    "trade_narrative.md":  (5_000, 1.15),
    "hermes_watchdog.md":  (8_000, 1.15),  # Hermes watchdog digest — same shape as strategic_log
}
```

- [ ] **Step 5: Run tests — confirm pass**

Run: `pytest tests/test_hermes_jobs_default.py -v`
Expected: ALL 4 tests pass.

- [ ] **Step 6: Run existing scheduled_jobs + brain_caps tests for no regression**

Run: `pytest tests/test_scheduled_jobs.py tests/test_brain_caps.py -q`
Expected: All existing tests still pass.

- [ ] **Step 7: Commit**

```bash
git add src/scanner/automation/scheduled_jobs.py src/scanner/automation/brain_caps.py tests/test_hermes_jobs_default.py
git commit -m "feat(hermes): foundation — default_jobs adds hermes_watchdog + hermes_daily_brief, brain_caps adds hermes_watchdog.md (8K)"
```

---

## Task 2: HermesState persistence helper

**Files:**
- Create: `src/scanner/automation/hermes_state.py`
- Create: `tests/test_hermes_state.py`

Single-responsibility module for the dedup + brief-delta state that both watchdog and brief scripts persist between runs. One file (`.claude/hermes_state.json`), atomic writes via `safe_json_write`.

- [ ] **Step 1: Write failing tests**

Create `tests/test_hermes_state.py`:

```python
"""HermesState persistence — load/save round-trip + defaults."""
from __future__ import annotations

from pathlib import Path

from src.scanner.automation.hermes_state import HermesState


def test_default_state_fields(tmp_path: Path):
    p = tmp_path / "hermes_state.json"
    s = HermesState.from_path(p)
    assert s.last_silent_at_iso is None
    assert s.last_alert_keys == {}
    assert s.last_job_failure_keys == {}
    assert s.last_brief_at_iso is None
    assert s.cycle_count_at_last_brief is None


def test_save_then_load_round_trip(tmp_path: Path):
    p = tmp_path / "hermes_state.json"
    s = HermesState(
        last_silent_at_iso="2026-05-12T07:00:00+00:00",
        last_alert_keys={"consecutive_losses": "2026-05-12T07:30:00+00:00"},
        last_job_failure_keys={"nightly_audit": "2026-05-12T22:00:00+00:00"},
        last_brief_at_iso="2026-05-12T07:00:00+00:00",
        cycle_count_at_last_brief=12345,
    )
    s.save_to(p)
    s2 = HermesState.from_path(p)
    assert s2 == s


def test_missing_file_returns_defaults(tmp_path: Path):
    p = tmp_path / "does_not_exist.json"
    s = HermesState.from_path(p)
    assert s == HermesState()


def test_corrupt_file_returns_defaults(tmp_path: Path):
    p = tmp_path / "hermes_state.json"
    p.write_text("not valid json {{{")
    s = HermesState.from_path(p)
    assert s == HermesState()


def test_partial_file_loads_missing_fields_as_defaults(tmp_path: Path):
    """Forward-compatibility: old state files load with new fields defaulted."""
    p = tmp_path / "hermes_state.json"
    p.write_text('{"last_brief_at_iso": "2026-05-10T07:00:00+00:00"}')
    s = HermesState.from_path(p)
    assert s.last_brief_at_iso == "2026-05-10T07:00:00+00:00"
    assert s.last_alert_keys == {}
    assert s.cycle_count_at_last_brief is None
```

- [ ] **Step 2: Run — confirm failure**

Run: `pytest tests/test_hermes_state.py -v`
Expected: `ModuleNotFoundError: No module named 'src.scanner.automation.hermes_state'`.

- [ ] **Step 3: Implement HermesState**

Create `src/scanner/automation/hermes_state.py`:

```python
"""Hermes watchdog + daily brief state persistence.

One on-disk JSON at .claude/hermes_state.json. Atomic writes via the
cloud-branch safe_json utilities. Forward-compatible: missing fields
load with their dataclass defaults so adding fields in future patches
never breaks an older state file.

Buddy policies honored:
- safe_json_read with try/except + graceful fallback to defaults
- safe_json_write (atomic temp+rename + fcntl lock under the hood)
- No bare except: clauses
- Default values are sensible (None / empty dict) so first-run is safe
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Optional

from src.scanner.automation.safe_json import safe_json_read, safe_json_write


_DEFAULT_STATE_PATH = (
    Path(__file__).resolve().parents[3] / ".claude" / "hermes_state.json"
)


@dataclass
class HermesState:
    """Cross-tick state for the watchdog + daily brief."""

    # Watchdog dedup — rate-limit silent entries to once per 4h
    last_silent_at_iso: Optional[str] = None

    # Watchdog dedup — alert_type → ISO of last watch entry written for it
    # Prevents the same active alert from re-firing every 30 min
    last_alert_keys: Dict[str, str] = field(default_factory=dict)

    # Watchdog dedup — job_id → ISO of last watch entry written for it
    # Prevents repeated job failures from spamming the digest
    last_job_failure_keys: Dict[str, str] = field(default_factory=dict)

    # Brief — for cycle_count delta computation
    last_brief_at_iso: Optional[str] = None
    cycle_count_at_last_brief: Optional[int] = None

    # ── persistence ────────────────────────────────────────────────────

    @classmethod
    def from_path(cls, path: Path = _DEFAULT_STATE_PATH) -> "HermesState":
        """Load from disk; return defaults on missing/corrupt/parse-error.

        Forward-compatible — unknown keys in the file are ignored, missing
        keys load as their dataclass defaults.
        """
        raw = safe_json_read(path, default=None)
        if not isinstance(raw, dict):
            return cls()
        known = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in raw.items() if k in known}
        try:
            return cls(**filtered)
        except (TypeError, ValueError):
            # Schema drift in the file — start fresh rather than crash the
            # watchdog. Old state is overwritten on next save.
            return cls()

    def save_to(self, path: Path = _DEFAULT_STATE_PATH) -> bool:
        """Atomic write. Returns True on success, False on I/O failure."""
        return bool(safe_json_write(path, asdict(self), sort_keys=True))
```

- [ ] **Step 4: Run tests — confirm pass**

Run: `pytest tests/test_hermes_state.py -v`
Expected: ALL 5 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/scanner/automation/hermes_state.py tests/test_hermes_state.py
git commit -m "feat(hermes): HermesState persistence helper for watchdog + brief dedup state"
```

---

## Task 3: Digest writer + rotation helper

**Files:**
- Create: `src/scanner/automation/hermes_digest.py`
- Create: `tests/test_hermes_digest.py`

Shared utility imported by both watchdog and brief. Owns the markdown append, day-header insertion, and rotation logic.

- [ ] **Step 1: Write failing tests**

Create `tests/test_hermes_digest.py`:

```python
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


def test_atomic_write_does_not_corrupt_on_failure(tmp_path: Path):
    """If the append writes a temp file then renames, a crash mid-write
    leaves the original intact. We verify the post-append state is valid
    UTF-8 and parseable by our own day-header scanner."""
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
```

- [ ] **Step 2: Run — confirm failure**

Run: `pytest tests/test_hermes_digest.py -v`
Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Implement digest writer + rotation**

Create `src/scanner/automation/hermes_digest.py`:

```python
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
                while (alt := self._archive_dir / f"hermes_watchdog_{iso_year}_{iso_week:02d}_{i}.md").exists():
                    i += 1
                target = alt
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
```

- [ ] **Step 4: Run tests — confirm pass**

Run: `pytest tests/test_hermes_digest.py -v`
Expected: ALL 7 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/scanner/automation/hermes_digest.py tests/test_hermes_digest.py
git commit -m "feat(hermes): digest writer + weekly rotation helper (atomic markdown append)"
```

---

## Task 4: Watchdog script

**Files:**
- Create: `src/scanner/automation/hermes_watchdog.py`
- Create: `tests/test_hermes_watchdog.py`

The subprocess entry point fired by `ScheduledJobsRegistry` every 30 min. Reads state files, decides watch/silent/no-op, writes via `HermesDigest`, persists dedup state via `HermesState`.

- [ ] **Step 1: Write failing tests**

Create `tests/test_hermes_watchdog.py`:

```python
"""Hermes watchdog — decision tree + dedup."""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from src.scanner.automation.hermes_watchdog import WatchdogContext, run_once


UTC = timezone.utc
FIXED_NOW = datetime(2026, 5, 12, 14, 30, 0, tzinfo=UTC)


def _seed_clean_state(root: Path) -> None:
    """Plant a fully-healthy set of input files under root."""
    claude = root / ".claude"
    claude.mkdir(parents=True, exist_ok=True)
    (claude / "heartbeat.json").write_text(json.dumps({
        "scanner_alive": True,
        "ts_iso": (FIXED_NOW - timedelta(seconds=2)).isoformat(),
        "cycle_count": 100,
        "pid": 1234,
        "mode": "live",
        "last_error_ts": None,
    }))
    (claude / "state.json").write_text(json.dumps({"halted": False, "mode": "live"}))
    (claude / "alert_state.json").write_text(json.dumps({
        "active_alerts": [],
        "last_fired": {},
        "last_updated": FIXED_NOW.isoformat(),
    }))
    (root / "trained_data").mkdir(parents=True, exist_ok=True)
    (root / "trained_data" / "jobs_runtime_state.json").write_text(json.dumps({}))


def _ctx(root: Path) -> WatchdogContext:
    return WatchdogContext(
        repo_root=root,
        now=FIXED_NOW,
    )


def test_clean_state_with_no_recent_silent_writes_silent(tmp_path: Path):
    _seed_clean_state(tmp_path)
    result = run_once(_ctx(tmp_path))
    assert result.entries_written == ["silent"]
    digest = (tmp_path / ".claude" / "brain" / "hermes_watchdog.md").read_text()
    assert "### 14:30Z — silent" in digest


def test_clean_state_with_recent_silent_writes_nothing(tmp_path: Path):
    _seed_clean_state(tmp_path)
    # Mark a recent silent (2h ago) in the state file.
    state_path = tmp_path / ".claude" / "hermes_state.json"
    state_path.write_text(json.dumps({
        "last_silent_at_iso": (FIXED_NOW - timedelta(hours=2)).isoformat(),
    }))
    result = run_once(_ctx(tmp_path))
    assert result.entries_written == []
    # No digest file created at all.
    digest_path = tmp_path / ".claude" / "brain" / "hermes_watchdog.md"
    assert not digest_path.exists()


def test_active_unacknowledged_alert_writes_watch(tmp_path: Path):
    _seed_clean_state(tmp_path)
    alert_state = json.loads((tmp_path / ".claude" / "alert_state.json").read_text())
    alert_state["active_alerts"] = [{
        "alert_type": "consecutive_losses",
        "severity": "WARNING",
        "message": "3 consecutive losses (threshold: 3)",
        "timestamp": FIXED_NOW.isoformat(),
        "value": 3.0,
        "threshold": 3.0,
        "pair": "",
        "acknowledged": False,
    }]
    (tmp_path / ".claude" / "alert_state.json").write_text(json.dumps(alert_state))
    result = run_once(_ctx(tmp_path))
    assert "watch" in result.entries_written
    digest = (tmp_path / ".claude" / "brain" / "hermes_watchdog.md").read_text()
    assert "consecutive_losses" in digest
    assert "3 consecutive losses" in digest


def test_acknowledged_alert_does_not_write_watch(tmp_path: Path):
    _seed_clean_state(tmp_path)
    alert_state = json.loads((tmp_path / ".claude" / "alert_state.json").read_text())
    alert_state["active_alerts"] = [{
        "alert_type": "consecutive_losses",
        "severity": "WARNING",
        "message": "3 consecutive losses",
        "timestamp": FIXED_NOW.isoformat(),
        "value": 3.0,
        "threshold": 3.0,
        "pair": "",
        "acknowledged": True,   # ← key difference
    }]
    (tmp_path / ".claude" / "alert_state.json").write_text(json.dumps(alert_state))
    result = run_once(_ctx(tmp_path))
    assert "watch" not in result.entries_written


def test_same_alert_within_dedup_window_does_not_re_fire(tmp_path: Path):
    _seed_clean_state(tmp_path)
    alert_state = json.loads((tmp_path / ".claude" / "alert_state.json").read_text())
    alert_state["active_alerts"] = [{
        "alert_type": "consecutive_losses",
        "severity": "WARNING",
        "message": "3 consecutive losses",
        "timestamp": FIXED_NOW.isoformat(),
        "value": 3.0, "threshold": 3.0, "pair": "", "acknowledged": False,
    }]
    (tmp_path / ".claude" / "alert_state.json").write_text(json.dumps(alert_state))
    # Mark a previous watch for this alert key 10 min ago.
    state_path = tmp_path / ".claude" / "hermes_state.json"
    state_path.write_text(json.dumps({
        "last_alert_keys": {
            "consecutive_losses": (FIXED_NOW - timedelta(minutes=10)).isoformat(),
        },
    }))
    result = run_once(_ctx(tmp_path))
    assert "watch" not in result.entries_written


def test_heartbeat_stale_writes_watch(tmp_path: Path):
    _seed_clean_state(tmp_path)
    hb = json.loads((tmp_path / ".claude" / "heartbeat.json").read_text())
    hb["ts_iso"] = (FIXED_NOW - timedelta(seconds=90)).isoformat()  # > 60s old
    (tmp_path / ".claude" / "heartbeat.json").write_text(json.dumps(hb))
    result = run_once(_ctx(tmp_path))
    assert "watch" in result.entries_written
    digest = (tmp_path / ".claude" / "brain" / "hermes_watchdog.md").read_text()
    assert "heartbeat_stale" in digest


def test_job_failure_writes_watch_excluding_self(tmp_path: Path):
    """A failing job triggers an alert — but the watchdog must skip itself."""
    _seed_clean_state(tmp_path)
    (tmp_path / "trained_data" / "jobs_runtime_state.json").write_text(json.dumps({
        "nightly_audit": {
            "state": "active",
            "last_status": "failure",
            "last_error": "exit 1: disk full at /tmp",
            "last_status_at": FIXED_NOW.isoformat(),
            "run_count": 5,
        },
        "hermes_watchdog": {   # ← this one is OUR job; must be skipped
            "state": "active",
            "last_status": "failure",
            "last_error": "recursive failure",
            "last_status_at": FIXED_NOW.isoformat(),
            "run_count": 1,
        },
    }))
    result = run_once(_ctx(tmp_path))
    assert "watch" in result.entries_written
    digest = (tmp_path / ".claude" / "brain" / "hermes_watchdog.md").read_text()
    assert "nightly_audit" in digest
    # The watchdog must NEVER alert on its own failure (would be recursive).
    assert "hermes_watchdog" not in digest.split("### ")[-1]


def test_paused_failing_job_does_not_alert(tmp_path: Path):
    _seed_clean_state(tmp_path)
    (tmp_path / "trained_data" / "jobs_runtime_state.json").write_text(json.dumps({
        "nightly_audit": {
            "state": "paused",  # ← paused jobs are not actively failing
            "last_status": "failure",
            "last_error": "old error",
            "last_status_at": (FIXED_NOW - timedelta(hours=24)).isoformat(),
            "run_count": 5,
        },
    }))
    result = run_once(_ctx(tmp_path))
    assert "watch" not in result.entries_written


def test_corrupt_alert_state_does_not_crash(tmp_path: Path):
    _seed_clean_state(tmp_path)
    (tmp_path / ".claude" / "alert_state.json").write_text("not valid json {{{")
    # Should not raise. Should still complete (silent or no-op).
    result = run_once(_ctx(tmp_path))
    # Don't assert on entry kind — just that we didn't crash.
    assert result is not None
```

- [ ] **Step 2: Run — confirm failure**

Run: `pytest tests/test_hermes_watchdog.py -v`
Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Implement watchdog**

Create `src/scanner/automation/hermes_watchdog.py`:

```python
"""Hermes watchdog — 30-min in-process anomaly detector.

Subprocess entry point fired by ScheduledJobsRegistry. Reads scanner
state files, decides watch/silent/no-op, appends to digest, persists
dedup state.

Buddy policies honored (per CLAUDE.md + .claude/rules/improvement.md):
- Claude-free: no LLM, no external network calls.
- JSON reads wrapped in try/except, graceful fallback to defaults.
- Atomic writes via HermesDigest (markdown) + HermesState (JSON).
- Specific exception types (OSError, json.JSONDecodeError) — no bare except.
- Watchdog NEVER alerts on its own job_id (recursive-failure guard).
- Dedup: same alert_type or job_id within 30 min → no re-write.
- Silent rate-limit: max one silent entry per 4h.
"""
from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional

from src.scanner.automation.brain_caps import caps as brain_caps
from src.scanner.automation.hermes_digest import DigestEntry, HermesDigest
from src.scanner.automation.hermes_state import HermesState
from src.scanner.automation.safe_json import safe_json_read


logger = logging.getLogger(__name__)


_OWN_JOB_ID = "hermes_watchdog"
_HEARTBEAT_STALE_THRESHOLD_SEC = 60.0
_SILENT_RATE_LIMIT_HOURS = 4.0
_ALERT_DEDUP_WINDOW_MIN = 30.0
_JOB_FAILURE_DEDUP_WINDOW_MIN = 30.0


@dataclass(frozen=True)
class WatchdogContext:
    repo_root: Path
    now: datetime  # tz-aware UTC


@dataclass
class WatchdogResult:
    entries_written: List[str] = field(default_factory=list)  # "watch" | "silent"


def run_once(ctx: WatchdogContext) -> WatchdogResult:
    """One tick. Returns what got written.

    Never raises — failures are logged and the next tick (30 min later)
    has a fresh chance. This is intentional: the watchdog must NOT take
    the scan loop down by itself.
    """
    try:
        return _run_once_impl(ctx)
    except Exception as e:  # noqa: BLE001 — top-level safety net only
        logger.error("hermes_watchdog crashed: %s", e, exc_info=True)
        return WatchdogResult(entries_written=[])


def _run_once_impl(ctx: WatchdogContext) -> WatchdogResult:
    state_path = ctx.repo_root / ".claude" / "hermes_state.json"
    state = HermesState.from_path(state_path)

    alert_state = _read_alert_state(ctx.repo_root)
    heartbeat = _read_heartbeat(ctx.repo_root)
    jobs_state = _read_jobs_state(ctx.repo_root)

    triggers: List[_Trigger] = []

    # ── 1. Active unacknowledged alerts (from AlertManager) ────────────
    for alert in alert_state.get("active_alerts", []):
        if not isinstance(alert, dict):
            continue
        if alert.get("acknowledged"):
            continue
        alert_type = str(alert.get("alert_type", "unknown"))
        if _within_dedup_window(state.last_alert_keys.get(alert_type),
                                ctx.now, _ALERT_DEDUP_WINDOW_MIN):
            continue
        message = str(alert.get("message", ""))[:200]
        severity = str(alert.get("severity", ""))
        triggers.append(_Trigger(
            kind="watch",
            body=[
                f"trigger: alert_type={alert_type}  severity={severity}",
                f"detail: {message}",
            ],
            dedup_alert_key=alert_type,
        ))

    # ── 2. Heartbeat staleness ─────────────────────────────────────────
    hb_age = _heartbeat_age_sec(heartbeat, ctx.now)
    if hb_age is not None and hb_age > _HEARTBEAT_STALE_THRESHOLD_SEC:
        scanner_alive = bool(heartbeat.get("scanner_alive", False))
        key = "heartbeat_stale"
        if not _within_dedup_window(state.last_alert_keys.get(key),
                                    ctx.now, _ALERT_DEDUP_WINDOW_MIN):
            triggers.append(_Trigger(
                kind="watch",
                body=[
                    f"trigger: heartbeat_stale  age={hb_age:.0f}s  scanner_alive_flag={scanner_alive}",
                    f"detail: heartbeat.ts_iso older than {int(_HEARTBEAT_STALE_THRESHOLD_SEC)}s; scanner process likely dead",
                ],
                dedup_alert_key=key,
            ))

    # ── 3. Job failures (T1 surface) — exclude self ────────────────────
    for job_id, jstate in jobs_state.items():
        if job_id == _OWN_JOB_ID:
            # Recursive-failure guard — never alert on our own job.
            continue
        if not isinstance(jstate, dict):
            continue
        if jstate.get("state") != "active":
            continue
        if jstate.get("last_status") != "failure":
            continue
        if _within_dedup_window(state.last_job_failure_keys.get(job_id),
                                ctx.now, _JOB_FAILURE_DEDUP_WINDOW_MIN):
            continue
        last_error = str(jstate.get("last_error", ""))[:200]
        last_status_at = jstate.get("last_status_at", "")
        triggers.append(_Trigger(
            kind="watch",
            body=[
                f"trigger: scheduled_jobs failure  job_id={job_id}",
                f"detail: last_status_at={last_status_at}  last_error={last_error}",
            ],
            dedup_job_key=job_id,
        ))

    # ── 4. Silent (if no triggers and rate-limit passed) ───────────────
    entries_written: List[str] = []
    if not triggers:
        last_silent = _parse_iso(state.last_silent_at_iso)
        elapsed_h = ((ctx.now - last_silent).total_seconds() / 3600.0
                     if last_silent else float("inf"))
        if elapsed_h >= _SILENT_RATE_LIMIT_HOURS:
            triggers.append(_Trigger(
                kind="silent",
                body=[
                    "no alert; logged for completeness (proof-of-life)",
                ],
            ))
            state.last_silent_at_iso = ctx.now.isoformat()

    if not triggers:
        # Nothing to write this tick.
        return WatchdogResult(entries_written=[])

    # ── Write all triggers ─────────────────────────────────────────────
    digest = _make_digest(ctx)
    iso_now = ctx.now.isoformat()
    for t in triggers:
        digest.append(DigestEntry(at=ctx.now, kind=t.kind, body=t.body))
        entries_written.append(t.kind)
        if t.dedup_alert_key:
            state.last_alert_keys[t.dedup_alert_key] = iso_now
        if t.dedup_job_key:
            state.last_job_failure_keys[t.dedup_job_key] = iso_now

    state.save_to(state_path)
    return WatchdogResult(entries_written=entries_written)


# ── helpers ────────────────────────────────────────────────────────────


@dataclass
class _Trigger:
    kind: str
    body: List[str]
    dedup_alert_key: Optional[str] = None
    dedup_job_key: Optional[str] = None


def _read_alert_state(root: Path) -> dict:
    p = root / ".claude" / "alert_state.json"
    raw = safe_json_read(p, default=None)
    return raw if isinstance(raw, dict) else {}


def _read_heartbeat(root: Path) -> dict:
    p = root / ".claude" / "heartbeat.json"
    raw = safe_json_read(p, default=None)
    return raw if isinstance(raw, dict) else {}


def _read_jobs_state(root: Path) -> dict:
    p = root / "trained_data" / "jobs_runtime_state.json"
    raw = safe_json_read(p, default=None)
    return raw if isinstance(raw, dict) else {}


def _heartbeat_age_sec(heartbeat: dict, now: datetime) -> Optional[float]:
    ts_iso = heartbeat.get("ts_iso")
    ts = _parse_iso(ts_iso)
    if ts is None:
        return None
    return (now - ts).total_seconds()


def _parse_iso(s: Optional[str]) -> Optional[datetime]:
    if not s:
        return None
    try:
        dt = datetime.fromisoformat(s)
    except (TypeError, ValueError):
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _within_dedup_window(last_iso: Optional[str], now: datetime,
                         minutes: float) -> bool:
    last = _parse_iso(last_iso)
    if last is None:
        return False
    return (now - last).total_seconds() < minutes * 60


def _make_digest(ctx: WatchdogContext) -> HermesDigest:
    digest_path = ctx.repo_root / ".claude" / "brain" / "hermes_watchdog.md"
    archive_dir = ctx.repo_root / ".claude" / "brain" / ".archive"
    hard_cap, _warn_ratio = brain_caps().get("hermes_watchdog.md", (8000, 1.15))
    return HermesDigest(
        digest_path=digest_path,
        archive_dir=archive_dir,
        hard_cap=hard_cap,
    )


def main() -> int:
    """Subprocess entry — called by `python -m src.scanner.automation.hermes_watchdog`."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    repo_root = Path(__file__).resolve().parents[3]
    ctx = WatchdogContext(repo_root=repo_root, now=datetime.now(timezone.utc))
    result = run_once(ctx)
    if result.entries_written:
        logger.info("hermes_watchdog wrote: %s", result.entries_written)
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run tests — confirm pass**

Run: `pytest tests/test_hermes_watchdog.py -v`
Expected: ALL 9 tests pass.

- [ ] **Step 5: Module-import smoke test (verify entry point works)**

Run: `python -m src.scanner.automation.hermes_watchdog`
Expected: exit code 0. If state files are clean, may write a `silent` entry to `.claude/brain/hermes_watchdog.md` (first run since `hermes_state.json` is missing). Verify:

```bash
ls .claude/brain/hermes_watchdog.md && cat .claude/hermes_state.json
```

- [ ] **Step 6: Commit**

```bash
git add src/scanner/automation/hermes_watchdog.py tests/test_hermes_watchdog.py
git commit -m "feat(hermes): 30-min watchdog with active-alert / heartbeat / job-failure triggers + dedup"
```

---

## Task 5: Daily brief script

**Files:**
- Create: `src/scanner/automation/hermes_daily_brief.py`
- Create: `tests/test_hermes_daily_brief.py`

The daily entry point. Composes a fixed-schema brief from `state.json`, `heartbeat.json`, `trade_journal_rl.json`, per-pair model meta sidecars, `jobs_runtime_state.json`, and the digest's last-24h `watch` count.

- [ ] **Step 1: Write failing tests**

Create `tests/test_hermes_daily_brief.py`:

```python
"""Hermes daily brief — fixed-schema composition."""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from src.scanner.automation.hermes_daily_brief import BriefContext, run_once


UTC = timezone.utc
NOW = datetime(2026, 5, 13, 7, 0, 0, tzinfo=UTC)
MIDNIGHT_TODAY = datetime(2026, 5, 13, 0, 0, tzinfo=UTC)


def _seed_minimal(root: Path) -> None:
    """Plant minimal inputs for a clean brief composition."""
    claude = root / ".claude"
    claude.mkdir(parents=True, exist_ok=True)
    (claude / "heartbeat.json").write_text(json.dumps({
        "scanner_alive": True,
        "ts_iso": (NOW - timedelta(seconds=2)).isoformat(),
        "cycle_count": 12500,
        "pid": 1234,
        "mode": "live",
    }))
    (claude / "state.json").write_text(json.dumps({
        "halted": False, "mode": "live",
    }))
    (claude / "alert_state.json").write_text(json.dumps({
        "active_alerts": [],
    }))
    (root / "trained_data").mkdir(parents=True, exist_ok=True)
    (root / "trained_data" / "trade_journal_rl.json").write_text(json.dumps([]))
    (root / "trained_data" / "jobs_runtime_state.json").write_text(json.dumps({}))
    (root / "trained_data" / "models").mkdir(parents=True, exist_ok=True)


def _ctx(root: Path) -> BriefContext:
    return BriefContext(repo_root=root, now=NOW)


def test_minimal_brief_composes_with_nominal_notable(tmp_path: Path):
    _seed_minimal(tmp_path)
    run_once(_ctx(tmp_path))
    digest = (tmp_path / ".claude" / "brain" / "hermes_watchdog.md").read_text()
    assert "### 07:00Z — brief" in digest
    assert "halted: false" in digest
    assert "mode: live" in digest
    assert "all systems nominal" in digest


def test_cycle_count_delta_uses_last_brief_state(tmp_path: Path):
    _seed_minimal(tmp_path)
    state_path = tmp_path / ".claude" / "hermes_state.json"
    # Yesterday's brief recorded cycle_count=12400; current is 12500.
    state_path.write_text(json.dumps({
        "last_brief_at_iso": (NOW - timedelta(days=1)).isoformat(),
        "cycle_count_at_last_brief": 12400,
    }))
    run_once(_ctx(tmp_path))
    digest = (tmp_path / ".claude" / "brain" / "hermes_watchdog.md").read_text()
    assert "cycles_today: 100" in digest  # 12500 - 12400


def test_first_brief_reports_unknown_cycles_today(tmp_path: Path):
    _seed_minimal(tmp_path)
    # No hermes_state.json yet.
    run_once(_ctx(tmp_path))
    digest = (tmp_path / ".claude" / "brain" / "hermes_watchdog.md").read_text()
    assert "cycles_today: unknown" in digest


def test_brief_persists_cycle_count_for_next_run(tmp_path: Path):
    _seed_minimal(tmp_path)
    run_once(_ctx(tmp_path))
    state = json.loads((tmp_path / ".claude" / "hermes_state.json").read_text())
    assert state["cycle_count_at_last_brief"] == 12500
    assert state["last_brief_at_iso"] == NOW.isoformat()


def test_trades_24h_counts_only_today_closes(tmp_path: Path):
    _seed_minimal(tmp_path)
    trades = [
        # Yesterday — must NOT count
        {"pair": "EUR_USD", "direction": "LONG", "pnl": 50.0,
         "close_time": (MIDNIGHT_TODAY - timedelta(hours=2)).isoformat()},
        # Today — must count
        {"pair": "GBP_USD", "direction": "SHORT", "pnl": -23.0,
         "close_time": (MIDNIGHT_TODAY + timedelta(hours=3)).isoformat()},
        {"pair": "USD_JPY", "direction": "LONG", "pnl": 12.5,
         "close_time": (MIDNIGHT_TODAY + timedelta(hours=5)).isoformat()},
    ]
    (tmp_path / "trained_data" / "trade_journal_rl.json").write_text(json.dumps(trades))
    run_once(_ctx(tmp_path))
    digest = (tmp_path / ".claude" / "brain" / "hermes_watchdog.md").read_text()
    # 2 closes today, sum = -23.0 + 12.5 = -10.5
    assert "trades_24h: 2 trades" in digest
    assert "-10.50" in digest or "$-10.50" in digest


def test_notable_priority_halted_loss_streak(tmp_path: Path):
    _seed_minimal(tmp_path)
    (tmp_path / ".claude" / "state.json").write_text(json.dumps({
        "halted": True, "mode": "live",
    }))
    (tmp_path / ".claude" / "alert_state.json").write_text(json.dumps({
        "active_alerts": [{
            "alert_type": "consecutive_losses",
            "value": 5.0, "threshold": 3.0,
            "acknowledged": False,
            "message": "5 consecutive losses",
        }],
    }))
    run_once(_ctx(tmp_path))
    digest = (tmp_path / ".claude" / "brain" / "hermes_watchdog.md").read_text()
    assert "halted on loss streak" in digest


def test_notable_priority_halted_alone(tmp_path: Path):
    _seed_minimal(tmp_path)
    (tmp_path / ".claude" / "state.json").write_text(json.dumps({
        "halted": True, "mode": "live",
    }))
    run_once(_ctx(tmp_path))
    digest = (tmp_path / ".claude" / "brain" / "hermes_watchdog.md").read_text()
    assert "halted; operator un-halt required to resume" in digest


def test_notable_priority_job_failure(tmp_path: Path):
    _seed_minimal(tmp_path)
    (tmp_path / "trained_data" / "jobs_runtime_state.json").write_text(json.dumps({
        "nightly_audit": {
            "state": "active",
            "last_status": "failure",
            "last_error": "exit 1: out of disk",
            "last_status_at": (NOW - timedelta(hours=2)).isoformat(),
            "run_count": 7,
        },
    }))
    run_once(_ctx(tmp_path))
    digest = (tmp_path / ".claude" / "brain" / "hermes_watchdog.md").read_text()
    assert "scheduled job nightly_audit failed" in digest


def test_alerts_24h_counts_watch_entries(tmp_path: Path):
    _seed_minimal(tmp_path)
    # Pre-populate digest with two watch entries inside the 24h window
    # and one outside (yesterday before midnight).
    digest_path = tmp_path / ".claude" / "brain" / "hermes_watchdog.md"
    digest_path.parent.mkdir(parents=True, exist_ok=True)
    digest_path.write_text(
        "# Hermes Watchdog Digest\n\n"
        "## 2026-05-12\n\n"
        "### 14:30Z — watch\n- trigger: consecutive_losses\n\n"
        "### 18:00Z — watch\n- trigger: drawdown\n\n"
        "## 2026-05-11\n\n"
        "### 23:00Z — watch\n- trigger: stale_models\n\n"  # > 24h ago
    )
    run_once(_ctx(tmp_path))
    digest = digest_path.read_text()
    # Brief sees 2 watches in the trailing 24h.
    assert "alerts_24h: 2 watch entries" in digest
```

- [ ] **Step 2: Run — confirm failure**

Run: `pytest tests/test_hermes_daily_brief.py -v`
Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Implement daily brief**

Create `src/scanner/automation/hermes_daily_brief.py`:

```python
"""Hermes daily brief — fixed-schema 07:00 UTC summary.

Subprocess entry point fired by ScheduledJobsRegistry daily at 07:00 UTC.
Composes a structured brief (no LLM, deterministic priority for the
`notable` line) and appends it to .claude/brain/hermes_watchdog.md.

Buddy policies honored:
- No LLM in the loop (deterministic priority for `notable`).
- JSON reads via safe_json_read with try/except + graceful fallback.
- Atomic markdown append via HermesDigest.
- State persistence via HermesState (cycle_count_at_last_brief for delta).
- Specific exception types — no bare except.
- Trade-journal load handles list shape (current contract) gracefully;
  unexpected shapes degrade to empty results, not crashes.
"""
from __future__ import annotations

import json
import logging
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.scanner.automation.brain_caps import caps as brain_caps
from src.scanner.automation.hermes_digest import DigestEntry, HermesDigest
from src.scanner.automation.hermes_state import HermesState
from src.scanner.automation.safe_json import safe_json_read

logger = logging.getLogger(__name__)


_OWN_JOB_ID = "hermes_daily_brief"
_DAY_HEADER_RE = re.compile(r"^## (\d{4}-\d{2}-\d{2})\s*$", re.MULTILINE)
_WATCH_ENTRY_RE = re.compile(r"^### (\d{2}):(\d{2})Z — watch\s*$", re.MULTILINE)


@dataclass(frozen=True)
class BriefContext:
    repo_root: Path
    now: datetime  # tz-aware UTC


def run_once(ctx: BriefContext) -> None:
    """Compose + append the daily brief. Never raises."""
    try:
        _run_once_impl(ctx)
    except Exception as e:  # noqa: BLE001 — top-level safety net
        logger.error("hermes_daily_brief crashed: %s", e, exc_info=True)


def _run_once_impl(ctx: BriefContext) -> None:
    state_path = ctx.repo_root / ".claude" / "hermes_state.json"
    state = HermesState.from_path(state_path)

    inputs = _gather_inputs(ctx)
    body = _compose_body(ctx, inputs, state)

    digest = _make_digest(ctx)
    digest.append(DigestEntry(at=ctx.now, kind="brief", body=body))

    # Persist the cycle_count for tomorrow's delta.
    state.last_brief_at_iso = ctx.now.isoformat()
    state.cycle_count_at_last_brief = inputs.cycle_count
    state.save_to(state_path)


# ── inputs ─────────────────────────────────────────────────────────────


@dataclass
class _Inputs:
    halted: bool
    mode: str
    cycle_count: Optional[int]
    trades_today_count: int
    trades_today_pnl: float
    active_unack_alerts: List[dict]
    failing_jobs_24h: List[Tuple[str, str]]  # (job_id, last_error_snippet)
    model_ages: List[Tuple[str, float, str]]  # (pair, age_days, granularity)
    alerts_24h_count: int


def _gather_inputs(ctx: BriefContext) -> _Inputs:
    root = ctx.repo_root
    state_json = _read_json(root / ".claude" / "state.json")
    heartbeat = _read_json(root / ".claude" / "heartbeat.json")
    alert_state = _read_json(root / ".claude" / "alert_state.json")
    journal = _read_journal(root / "trained_data" / "trade_journal_rl.json")
    jobs = _read_json(root / "trained_data" / "jobs_runtime_state.json")

    halted = bool(state_json.get("halted", False)) if isinstance(state_json, dict) else False
    mode = str(state_json.get("mode", "unknown")) if isinstance(state_json, dict) else "unknown"
    cycle_count = (
        int(heartbeat.get("cycle_count")) if isinstance(heartbeat, dict)
        and isinstance(heartbeat.get("cycle_count"), int) else None
    )

    trades_today_count, trades_today_pnl = _trades_in_today(journal, ctx.now)
    active_unack = _active_unacknowledged_alerts(alert_state)
    failing_jobs_24h = _failing_jobs_in_last_24h(jobs, ctx.now, exclude={_OWN_JOB_ID})
    model_ages = _model_ages(root, ctx.now)
    alerts_24h_count = _count_watch_entries_in_last_24h(
        root / ".claude" / "brain" / "hermes_watchdog.md", ctx.now,
    )

    return _Inputs(
        halted=halted,
        mode=mode,
        cycle_count=cycle_count,
        trades_today_count=trades_today_count,
        trades_today_pnl=trades_today_pnl,
        active_unack_alerts=active_unack,
        failing_jobs_24h=failing_jobs_24h,
        model_ages=model_ages,
        alerts_24h_count=alerts_24h_count,
    )


# ── composition ────────────────────────────────────────────────────────


def _compose_body(ctx: BriefContext, inp: _Inputs, state: HermesState) -> List[str]:
    cycles_today_str = _format_cycles_today(inp.cycle_count, state)
    pnl_sign = "" if inp.trades_today_pnl < 0 else ""
    pnl_str = f"P&L {pnl_sign}${inp.trades_today_pnl:.2f}"
    model_str = _format_model_ages(inp.model_ages)
    job_counts = _job_counts_summary(ctx, inp.failing_jobs_24h)
    notable = _select_notable(inp)

    return [
        f"halted: {str(inp.halted).lower()} · mode: {inp.mode} · cycles_today: {cycles_today_str}",
        f"trades_24h: {inp.trades_today_count} trades · {pnl_str}",
        f"model ages: {model_str}",
        f"jobs: {job_counts}",
        f"alerts_24h: {inp.alerts_24h_count} watch entries",
        f"notable: {notable}",
    ]


def _select_notable(inp: _Inputs) -> str:
    """Fixed priority — first match wins."""
    has_loss_streak = any(
        a.get("alert_type") == "consecutive_losses"
        for a in inp.active_unack_alerts
    )
    if inp.halted and has_loss_streak:
        return "halted on loss streak — operator review required"
    if inp.halted:
        return "halted; operator un-halt required to resume"
    if inp.failing_jobs_24h:
        job_id, err_snippet = inp.failing_jobs_24h[0]
        return f"scheduled job {job_id} failed: {err_snippet[:80]}"
    stale = [(p, age, g) for p, age, g in inp.model_ages if age > 30.0]
    if stale:
        p, age, _g = stale[0]
        return f"model staleness — {p} is {age:.1f}d old"
    return "all systems nominal"


# ── helpers ────────────────────────────────────────────────────────────


def _read_json(path: Path) -> Any:
    raw = safe_json_read(path, default=None)
    return raw if raw is not None else {}


def _read_journal(path: Path) -> List[dict]:
    raw = safe_json_read(path, default=None)
    if isinstance(raw, list):
        return [r for r in raw if isinstance(r, dict)]
    if isinstance(raw, dict) and isinstance(raw.get("trades"), list):
        return [r for r in raw["trades"] if isinstance(r, dict)]
    return []


def _parse_iso(s: Optional[str]) -> Optional[datetime]:
    if not s:
        return None
    try:
        dt = datetime.fromisoformat(s)
    except (TypeError, ValueError):
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _midnight_utc(now: datetime) -> datetime:
    return now.replace(hour=0, minute=0, second=0, microsecond=0)


def _trades_in_today(journal: List[dict], now: datetime) -> Tuple[int, float]:
    midnight = _midnight_utc(now)
    count, pnl_sum = 0, 0.0
    for t in journal:
        ts = _parse_iso(t.get("close_time"))
        if ts is None or ts < midnight or ts >= now:
            continue
        count += 1
        try:
            pnl_sum += float(t.get("pnl", 0.0))
        except (TypeError, ValueError):
            pass
    return count, pnl_sum


def _active_unacknowledged_alerts(alert_state: Any) -> List[dict]:
    if not isinstance(alert_state, dict):
        return []
    alerts = alert_state.get("active_alerts", [])
    if not isinstance(alerts, list):
        return []
    return [a for a in alerts if isinstance(a, dict) and not a.get("acknowledged")]


def _failing_jobs_in_last_24h(
    jobs: Any, now: datetime, *, exclude: set,
) -> List[Tuple[str, str]]:
    if not isinstance(jobs, dict):
        return []
    cutoff = now - timedelta(hours=24)
    out: List[Tuple[str, str]] = []
    for job_id, jstate in jobs.items():
        if job_id in exclude or not isinstance(jstate, dict):
            continue
        if jstate.get("state") != "active":
            continue
        if jstate.get("last_status") != "failure":
            continue
        ts = _parse_iso(jstate.get("last_status_at"))
        if ts is None or ts < cutoff:
            continue
        err = str(jstate.get("last_error", ""))[:200]
        out.append((job_id, err))
    return out


def _model_ages(root: Path, now: datetime) -> List[Tuple[str, float, str]]:
    """Walk trained_data/models/<PAIR>/transformer_direction.meta.pkl.

    Uses joblib.load (sklearn convention) — the meta files are written by
    the trainer using joblib.dump. Returns empty list on any failure;
    operator sees model_ages: (unknown) and audits via the T10 Diagnostics
    panel (when shipped) or by hand.
    """
    out: List[Tuple[str, float, str]] = []
    models_root = root / "trained_data" / "models"
    if not models_root.exists():
        return out
    try:
        import joblib
    except ImportError:
        return out
    for pair_dir in sorted(models_root.iterdir()):
        if not pair_dir.is_dir():
            continue
        meta_path = pair_dir / "transformer_direction.meta.pkl"
        if not meta_path.exists():
            continue
        try:
            payload = joblib.load(meta_path)
        except (OSError, EOFError, KeyError, ValueError):
            continue
        if not isinstance(payload, dict):
            continue
        trained_at = _parse_iso(payload.get("trained_at"))
        if trained_at is None:
            continue
        age_days = (now - trained_at).total_seconds() / 86400.0
        granularity = str(payload.get("granularity", ""))
        out.append((pair_dir.name, age_days, granularity))
    return out


def _format_model_ages(ages: List[Tuple[str, float, str]]) -> str:
    if not ages:
        return "(unknown — no meta sidecars found)"
    parts = [f"{p} {a:.1f}d" for p, a, _g in ages]
    granularities = sorted({g for _p, _a, g in ages if g})
    suffix = f" ({'/'.join(granularities)})" if granularities else ""
    return " · ".join(parts) + suffix


def _format_cycles_today(cycle_count: Optional[int], state: HermesState) -> str:
    if cycle_count is None:
        return "unknown"
    last = state.cycle_count_at_last_brief
    if last is None:
        return "unknown"
    delta = cycle_count - int(last)
    return f"{delta}"


def _job_counts_summary(ctx: BriefContext, failing: List[Tuple[str, str]]) -> str:
    jobs = _read_json(ctx.repo_root / "trained_data" / "jobs_runtime_state.json")
    if not isinstance(jobs, dict):
        return "0 active, 0 paused, 0 failures in last 24h"
    active = sum(1 for j in jobs.values() if isinstance(j, dict) and j.get("state") == "active")
    paused = sum(1 for j in jobs.values() if isinstance(j, dict) and j.get("state") == "paused")
    return f"{active} active, {paused} paused, {len(failing)} failures in last 24h"


def _count_watch_entries_in_last_24h(digest_path: Path, now: datetime) -> int:
    if not digest_path.exists():
        return 0
    try:
        text = digest_path.read_text(encoding="utf-8")
    except OSError:
        return 0
    cutoff = now - timedelta(hours=24)
    # Walk top-to-bottom, tracking current day-header context.
    count = 0
    current_date: Optional[datetime] = None
    for line in text.splitlines():
        m_day = _DAY_HEADER_RE.match(line)
        if m_day:
            try:
                current_date = datetime.strptime(m_day.group(1), "%Y-%m-%d").replace(tzinfo=timezone.utc)
            except ValueError:
                current_date = None
            continue
        m_watch = _WATCH_ENTRY_RE.match(line)
        if m_watch and current_date is not None:
            hh, mm = int(m_watch.group(1)), int(m_watch.group(2))
            entry_at = current_date.replace(hour=hh, minute=mm)
            if entry_at >= cutoff:
                count += 1
    return count


def _make_digest(ctx: BriefContext) -> HermesDigest:
    digest_path = ctx.repo_root / ".claude" / "brain" / "hermes_watchdog.md"
    archive_dir = ctx.repo_root / ".claude" / "brain" / ".archive"
    hard_cap, _ = brain_caps().get("hermes_watchdog.md", (8000, 1.15))
    return HermesDigest(
        digest_path=digest_path,
        archive_dir=archive_dir,
        hard_cap=hard_cap,
    )


def main() -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    repo_root = Path(__file__).resolve().parents[3]
    ctx = BriefContext(repo_root=repo_root, now=datetime.now(timezone.utc))
    run_once(ctx)
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run tests — confirm pass**

Run: `pytest tests/test_hermes_daily_brief.py -v`
Expected: ALL 9 tests pass.

- [ ] **Step 5: Subprocess smoke test**

Run: `python -m src.scanner.automation.hermes_daily_brief`
Expected: exit 0; new `### HH:MMZ — brief` entry in `.claude/brain/hermes_watchdog.md`. Inspect:

```bash
tail -15 .claude/brain/hermes_watchdog.md
cat .claude/hermes_state.json
```

- [ ] **Step 6: Run full Hermes-watchdog test suite + check for regressions**

```bash
pytest tests/test_hermes_jobs_default.py tests/test_hermes_state.py \
       tests/test_hermes_digest.py tests/test_hermes_watchdog.py \
       tests/test_hermes_daily_brief.py -v
```

Expected: All tests pass (estimated 34 total: 4 + 5 + 7 + 9 + 9).

Also run the Tier 1 + cloud-branch tests to verify nothing regressed:

```bash
pytest tests/test_scheduled_jobs.py tests/test_brain_caps.py \
       tests/test_scheduled_jobs_observability.py -q
```

Expected: all green.

- [ ] **Step 7: Commit**

```bash
git add src/scanner/automation/hermes_daily_brief.py tests/test_hermes_daily_brief.py
git commit -m "feat(hermes): daily brief at 07:00 UTC — fixed-schema 24h summary (no LLM)"
```

---

## Self-review checklist

1. **Spec coverage:** every section of `docs/superpowers/specs/2026-05-12-hermes-watchdog-brief-design.md` has a corresponding implementation task above. ✓
   - Schedule grammar entries → Task 1
   - brain_caps entry → Task 1
   - State persistence → Task 2
   - Digest writer + rotation → Task 3
   - Watchdog decision tree → Task 4
   - Daily brief composition → Task 5

2. **Type / symbol consistency:**
   - `HermesState` (Task 2) used in Tasks 4 and 5 with the same field names. ✓
   - `HermesDigest` + `DigestEntry` (Task 3) used in Tasks 4 and 5 with the same constructor signature. ✓
   - `WatchdogContext` (Task 4) and `BriefContext` (Task 5) both have `repo_root: Path` and `now: datetime`. ✓

3. **No placeholders:** no "TBD", "TODO", "implement later", or comments without code. ✓

4. **No mocks:** every test uses real classes against real disk via `tmp_path`. ✓ (Inspect each `Step 1` — no `unittest.mock`, no `MagicMock`.)

5. **CLAUDE.md alignment:**
   - Claude-free runtime ✓ (subprocess entry points; no LLM call sites)
   - Atomic writes ✓ (`safe_json_write` for JSON, temp+rename for markdown)
   - JSON safety gates ✓ (`safe_json_read` with try/except + defaults)
   - Specific exception types ✓ (OSError, JSONDecodeError; no bare except)
   - Recursive-failure guard ✓ (watchdog excludes own `job_id`)
   - Config validation ✓ (real `brain_caps()` lookup, not hardcoded constants in two places)

6. **Spec D5 fix:** uses `alert_state.json:active_alerts` directly (which AlertManager already populates with thresholds via the `value`/`threshold` fields). The spec's "use ScannerConfig.alert_drawdown_threshold else 0.05" prose was incorrect; the recon-verified contract has the threshold baked into the alert record. No hardcoded fallback constant needed.

7. **Scope:** five tasks, each landable as its own PR. Dependencies: T2 → T4 and T5; T3 → T4 and T5; T1 is independent. Suggested merge order: T1, T2, T3, T4, T5.

## Execution handoff

Plan complete and saved. Two execution options per `superpowers:writing-plans`:

1. **Subagent-Driven (recommended)** — dispatch a fresh subagent per task (T1–T5), review between tasks, 5 independent PRs. Worktree isolation discouraged this round given the path-pollution lessons from Tier 1; inline execution may be cleaner.

2. **Inline Execution** — execute tasks sequentially in this session via `superpowers:executing-plans`, batched with operator checkpoints.

**Recommendation:** **Inline Execution** for the watchdog plan. Five tasks, all single-author, all narrow. The Tier 1 parallel-agent dispatch was justified by 6 truly-independent surfaces; this plan's tasks have linear dependencies (T1 → T2 → T3 → T4, T5) so serial execution avoids the worktree path-pollution issue that bit us during Tier 1. After the inline tasks land, the operator runs the manual smoke (boot `./buddy --demo`, press F9 to confirm both new jobs appear and tick).

"""Bounded-memory check for `.claude/brain/*.md` files (pick #6).

Buddy's brain files are read on every Claude/operator invocation
(CLAUDE.md "Read First on Every Invocation"). Letting them grow unbounded
slows session startup and rots context relevance. This module:

  * Defines soft char caps per file (hard cap + a 1.20x warn ceiling)
  * Exposes ``check_brain_caps()`` for both the TUI boot path and CI
  * Returns structured warnings — caller decides whether to log, banner, or fail

This is intentionally **non-blocking**: caps are advisory at v1. A future
iteration can rotate over-sized files into `.claude/brain/.archive/`.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[3]
_BRAIN_DIR = _REPO_ROOT / ".claude" / "brain"

# (hard_cap_chars, warn_ratio).
# Sized per the agent survey: caps are conservative — they allow normal
# growth but flag bloat. Files that exceed `hard_cap * warn_ratio` warn.
_DEFAULT_CAPS: dict[str, tuple[int, float]] = {
    "briefing.md":         (3_000, 1.20),
    "session_handoff.md":  (2_000, 1.20),
    "open_questions.md":   (1_500, 1.20),
    "strategic_log.md":    (8_000, 1.15),
    "trade_narrative.md":  (5_000, 1.15),
    "hermes_watchdog.md":  (8_000, 1.15),  # Hermes watchdog digest — same shape as strategic_log
}

# Severity bands above warn threshold.
_HIGH_SEVERITY_RATIO = 1.50   # ≥ 1.5x hard cap → "high" severity


@dataclass(frozen=True)
class BrainCapWarning:
    """One file's overage report."""

    filename: str
    char_count: int
    hard_cap: int
    warn_threshold: int
    severity: str  # "warn" | "high"

    @property
    def pct(self) -> int:
        return int(round(self.char_count / self.hard_cap * 100))

    def message(self) -> str:
        return (
            f".claude/brain/{self.filename} is {self.char_count:,} / "
            f"{self.hard_cap:,} chars ({self.pct}% of cap) — "
            "trim or archive outdated sections."
        )


def check_brain_caps(
    *,
    brain_dir: Path = _BRAIN_DIR,
    caps: Optional[dict[str, tuple[int, float]]] = None,
) -> list[BrainCapWarning]:
    """Return one warning per file that exceeds its warn threshold.

    Missing files are silently skipped (they may not exist yet in fresh
    checkouts). Unreadable files log a debug line but never raise — this
    function must be safe to call during boot.
    """
    caps = caps if caps is not None else _DEFAULT_CAPS
    out: list[BrainCapWarning] = []

    if not brain_dir.exists():
        return out

    for filename, (hard_cap, warn_ratio) in caps.items():
        path = brain_dir / filename
        if not path.exists() or not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as e:
            logger.debug("brain_caps: cannot read %s: %s", path, e)
            continue
        char_count = len(text)
        warn_threshold = int(hard_cap * warn_ratio)
        if char_count <= warn_threshold:
            continue
        severity = "high" if char_count >= int(hard_cap * _HIGH_SEVERITY_RATIO) else "warn"
        out.append(BrainCapWarning(
            filename=filename,
            char_count=char_count,
            hard_cap=hard_cap,
            warn_threshold=warn_threshold,
            severity=severity,
        ))
    return out


def format_warnings(warnings: Iterable[BrainCapWarning]) -> str:
    """Human-friendly multi-line summary; empty string when no warnings."""
    items = list(warnings)
    if not items:
        return ""
    lines = ["BrainCaps: %d file(s) over cap" % len(items)]
    for w in items:
        lines.append(f"  [{w.severity.upper()}] {w.message()}")
    return "\n".join(lines)


def caps() -> dict[str, tuple[int, float]]:
    """Read-only view of the default cap table — for tests + diagnostics."""
    return dict(_DEFAULT_CAPS)

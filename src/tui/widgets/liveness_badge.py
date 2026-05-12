"""LivenessBadge — at-a-glance health indicator over heartbeat.json + state.json.

Reads two files from the project's `.claude/` directory:
- ``heartbeat.json`` — written every ~10s by the TUI's heartbeat writer.
  Schema: ``ts_iso`` (UTC ISO8601), ``cycle_count``, ``scanner_alive``, …
- ``state.json`` — top-level ``halted`` boolean (StateEngine source of truth).

Renders one of five states:

=======  ======  ==========================================================
state    color   condition
=======  ======  ==========================================================
LIVE     green   fresh heartbeat (<15s) AND scanner_alive AND NOT halted
HALTED   red     state.json halted=true (overrides everything else)
STALE    yellow  heartbeat exists but >15s old OR scanner_alive=false
INIT     dim     heartbeat.json does not exist yet
ERROR    red     heartbeat.json or state.json present but unreadable
=======  ======  ==========================================================

Refresh cadence: every 5s via ``set_interval``. Pure function
``compute_liveness(claude_dir)`` is used by both the widget and the tests so
test behaviour is exactly what runs in the live TUI.

JSON reads use try/except with graceful fallback (NEVER crash on corrupt
state files) per ``.claude/rules/improvement.md`` "JSON Safety Gates".
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Tuple

from rich.text import Text
from textual.reactive import reactive
from textual.widgets import Static

logger = logging.getLogger(__name__)

# Freshness threshold: heartbeat older than this is STALE. Matches the
# CLAUDE.md "ts_iso ≤ 15s = alive" rule.
STALE_THRESHOLD_SECONDS: float = 15.0

# Auto-refresh cadence for the widget. Heartbeat writes every ~10s, so 5s
# gives us at least one observation per interval without thrashing.
REFRESH_INTERVAL_SECONDS: float = 5.0


@dataclass(frozen=True)
class LivenessState:
    """Result of evaluating heartbeat.json + state.json at one instant."""

    label: str          # one of: LIVE, HALTED, STALE, INIT, ERROR
    color: str          # rich color name / hex (used as label foreground)
    detail: str         # short suffix shown next to label (e.g. "3.2s ago")


# Color palette — kept consistent with the cyberpunk theme in theme.tcss.
_COLOR_LIVE = "#00ff88"      # neon green
_COLOR_HALTED = "#ff1744"    # alert red
_COLOR_STALE = "#ffb300"     # warning amber
_COLOR_INIT = "#888888"      # dim grey
_COLOR_ERROR = "#ff1744"     # alert red (same as HALTED — both are bad)


def _safe_read_json(path: Path) -> Tuple[Optional[dict], bool]:
    """Read JSON file, returning (data, file_existed).

    Returns ``(dict, True)`` on success, ``(None, True)`` if the file exists
    but is corrupt/unreadable, and ``(None, False)`` if the file is absent.
    Never raises — corrupt files become a None payload that the caller
    promotes to ERROR. Bare ``except`` is forbidden by improvement.md, so we
    catch the specific failure modes we expect.
    """
    try:
        with path.open("r") as f:
            data = json.load(f)
    except FileNotFoundError:
        return None, False
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("liveness_badge: failed to read %s: %r", path, exc)
        return None, True
    if not isinstance(data, dict):
        logger.warning(
            "liveness_badge: %s did not contain a JSON object (got %s)",
            path,
            type(data).__name__,
        )
        return None, True
    return data, True


def _parse_iso_utc(ts: object) -> Optional[datetime]:
    """Parse an ISO8601 timestamp; return None on any failure."""
    if not isinstance(ts, str):
        return None
    try:
        parsed = datetime.fromisoformat(ts)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        # Treat naive timestamps as UTC — matches the heartbeat writer.
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def compute_liveness(claude_dir: Path, *, now: Optional[datetime] = None) -> LivenessState:
    """Pure function: read ``.claude/heartbeat.json`` + ``.claude/state.json``
    and return the current LivenessState.

    ``now`` is injectable for deterministic tests; defaults to UTC now.
    ``claude_dir`` is the directory containing the two JSON files (typically
    ``<repo_root>/.claude``).
    """
    claude_dir = Path(claude_dir)
    if now is None:
        now = datetime.now(timezone.utc)

    heartbeat_path = claude_dir / "heartbeat.json"
    state_path = claude_dir / "state.json"

    heartbeat_data, heartbeat_existed = _safe_read_json(heartbeat_path)
    state_data, state_existed = _safe_read_json(state_path)

    # ERROR: a file is present but unreadable. This is a real fault — the
    # writer crashed mid-write, or someone hand-edited the JSON. Surface it.
    if (heartbeat_existed and heartbeat_data is None) or (
        state_existed and state_data is None
    ):
        return LivenessState(label="ERROR", color=_COLOR_ERROR, detail="unreadable JSON")

    # HALTED takes priority over freshness — even a fresh heartbeat from a
    # halted scanner is "halted", not "live". This matches the runtime
    # halt-check in EmbeddedScanner.run_one_cycle().
    if state_data is not None and bool(state_data.get("halted", False)):
        return LivenessState(label="HALTED", color=_COLOR_HALTED, detail="state.json halted=true")

    # INIT: no heartbeat written yet (first ~10s after launch).
    if not heartbeat_existed or heartbeat_data is None:
        return LivenessState(label="INIT", color=_COLOR_INIT, detail="no heartbeat yet")

    ts_parsed = _parse_iso_utc(heartbeat_data.get("ts_iso"))
    if ts_parsed is None:
        # Heartbeat file has no usable ts_iso — treat as corrupt.
        return LivenessState(label="ERROR", color=_COLOR_ERROR, detail="bad ts_iso")

    age_seconds = (now - ts_parsed).total_seconds()
    scanner_alive = bool(heartbeat_data.get("scanner_alive", False))

    if age_seconds > STALE_THRESHOLD_SECONDS:
        return LivenessState(
            label="STALE",
            color=_COLOR_STALE,
            detail=f"{age_seconds:.0f}s since heartbeat",
        )
    if not scanner_alive:
        return LivenessState(
            label="STALE",
            color=_COLOR_STALE,
            detail="scanner_alive=false",
        )

    cycle = heartbeat_data.get("cycle_count", "?")
    return LivenessState(
        label="LIVE",
        color=_COLOR_LIVE,
        detail=f"cycle {cycle} • {age_seconds:.1f}s ago",
    )


class LivenessBadge(Static):
    """One-line badge that auto-refreshes every 5s.

    Mount near the Footer in the main App. Reads `<repo_root>/.claude/*.json`
    on each tick and re-renders. Failures are surfaced as ERROR rather than
    crashing the TUI — the badge is a health indicator, it must never be
    the cause of a TUI crash itself.
    """

    DEFAULT_CSS = """
    LivenessBadge {
        height: 1;
        background: #0a0a14;
        padding: 0 1;
        text-style: bold;
    }
    """

    state: reactive[LivenessState] = reactive(
        LivenessState(label="INIT", color=_COLOR_INIT, detail="warming up")
    )

    def __init__(
        self,
        claude_dir: Path,
        *,
        refresh_interval: float = REFRESH_INTERVAL_SECONDS,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._claude_dir = Path(claude_dir)
        self._refresh_interval = float(refresh_interval)

    def on_mount(self) -> None:
        # Initial tick so the operator sees the real state immediately.
        self._tick()
        self.set_interval(self._refresh_interval, self._tick)

    def _tick(self) -> None:
        try:
            new_state = compute_liveness(self._claude_dir)
        except Exception as exc:  # noqa: BLE001 — surface as ERROR, never crash
            logger.error("liveness_badge: compute_liveness raised %r", exc)
            new_state = LivenessState(
                label="ERROR",
                color=_COLOR_ERROR,
                detail=f"{type(exc).__name__}",
            )
        self.state = new_state
        self.refresh()

    def render(self) -> Text:
        st = self.state
        t = Text()
        t.append(f"● {st.label}", style=f"bold {st.color}")
        if st.detail:
            t.append(f"  {st.detail}", style="dim")
        return t

"""Tier 1 T4: Transient phase indicator — "currently doing X" widget.

A lightweight inline status tag that names the active scan-loop phase
(scanning, gate-check, executing, idle) plus an optional detail string.
The widget polls a shared PhaseState dataclass every 0.5s — chosen
faster than the 3s overall refresh because phase transitions are
short-lived and the operator wants confirmation that "the bot is doing
something right now", not "the bot did something 3s ago".

Wiring:
- `EmbeddedScanner.__init__` creates `self.phase_state = PhaseState()`.
- `run_one_cycle` calls `.set("scanning", ...)`, `.set("gate-check", ...)`,
  `.set("executing", ...)`, and `.clear()` at natural phase boundaries.
- `tui/app.py` mounts `PhaseIndicator(state=self._scanner.phase_state)`
  near the F1 brain feed.
"""
from __future__ import annotations

from dataclasses import dataclass

from textual.reactive import reactive
from textual.widgets import Static


@dataclass
class PhaseState:
    """Shared mutable state between scanner thread and the indicator widget.

    The scanner thread writes via `set` / `clear`; the widget reads via
    `format`. Plain attribute writes are atomic for str in CPython —
    no lock needed because the widget only reads the strings, never
    mutates, and a torn read produces a transient stale label, not a
    crash.
    """

    phase: str = "idle"
    detail: str = ""

    def set(self, phase: str, detail: str = "") -> None:
        self.phase = phase
        self.detail = detail

    def clear(self) -> None:
        self.phase = "idle"
        self.detail = ""

    def format(self) -> str:
        if self.phase == "idle":
            return "[dim]— idle —[/]"
        body = f"[cyan]▸ {self.phase}[/]"
        if self.detail:
            body += f" · {self.detail}"
        return body


class PhaseIndicator(Static):
    """Inline transient phase tag; reads PhaseState every 0.5s."""

    text: reactive[str] = reactive("[dim]— idle —[/]")

    def __init__(self, *, state: PhaseState, refresh_sec: float = 0.5, **kwargs) -> None:
        super().__init__(**kwargs)
        self._state = state
        self._refresh_sec = refresh_sec

    def on_mount(self) -> None:
        self._tick()
        self.set_interval(self._refresh_sec, self._tick)

    def _tick(self) -> None:
        new_text = self._state.format()
        if new_text != self.text:
            self.text = new_text
            self.update(self.text)

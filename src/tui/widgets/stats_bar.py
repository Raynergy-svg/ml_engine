"""Tier 1 T3: Cumulative work-unit counter widget.

ScanCounters is a lifetime-of-session counter for cycles, pairs scanned,
gate evaluations, and trades executed. StatsBar is a Textual `Static`
widget that polls a ScanCounters instance every `refresh_sec` seconds
and renders a compact one-line summary plus a detailed tooltip.

Owned by EmbeddedScanner.counters; bumped inline at the natural phase
boundaries in run_one_cycle. Read-only from the TUI side.
"""
from __future__ import annotations

from dataclasses import dataclass

from textual.reactive import reactive
from textual.widgets import Static


@dataclass
class ScanCounters:
    """Cumulative work-unit counters, lifetime of TUI session.

    Owned by EmbeddedScanner.counters. Bumped inline at phase
    boundaries in run_one_cycle (cycle start, post-scan, gate-check,
    post-execution). Read by StatsBar via shared reference.
    """

    cycles: int = 0
    pairs_scanned: int = 0
    gates_checked: int = 0
    trades_executed: int = 0

    def bump_cycle(self, n: int = 1) -> None:
        self.cycles += n

    def bump_pair(self, n: int = 1) -> None:
        self.pairs_scanned += n

    def bump_gates_checked(self, n: int = 1) -> None:
        self.gates_checked += n

    def bump_trade(self, n: int = 1) -> None:
        self.trades_executed += n

    def format_compact(self) -> str:
        return (
            f"cycles {self.cycles} · pairs {self.pairs_scanned} · "
            f"gates {self.gates_checked} · trades {self.trades_executed}"
        )

    def format_detailed(self) -> str:
        return (
            f"cycles: {self.cycles}\n"
            f"pairs scanned: {self.pairs_scanned}\n"
            f"gates checked: {self.gates_checked}\n"
            f"trades executed: {self.trades_executed}"
        )


class StatsBar(Static):
    """Compact stats line. Tooltip shows detailed breakdown.

    Polls the shared ScanCounters every `refresh_sec` seconds and
    re-renders. No state of its own; the source-of-truth is the
    ScanCounters reference passed in at construction.
    """

    text: reactive[str] = reactive("")

    def __init__(
        self,
        *,
        counters: ScanCounters,
        refresh_sec: float = 2.0,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._counters = counters
        self._refresh_sec = refresh_sec
        self.tooltip = ""

    def on_mount(self) -> None:
        self._tick()
        self.set_interval(self._refresh_sec, self._tick)

    def _tick(self) -> None:
        self.text = self._counters.format_compact()
        self.tooltip = self._counters.format_detailed()
        self.update(self.text)

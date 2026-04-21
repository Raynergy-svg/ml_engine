"""StateStrip — persistent supervisor state bar for the Overview tab.

Shows scanner running/paused/halted status, dry-run vs live mode,
and model staleness in days. Updated every refresh cycle via update_from_snapshot().
"""
from __future__ import annotations

from rich.text import Text
from textual.reactive import reactive
from textual.widgets import Static

from src.tui.data_provider import DashboardSnapshot


class StateStrip(Static):
    """One-row status strip showing scanner state, mode, and model staleness."""

    scanner_status: reactive[str] = reactive("RUNNING")
    mode: reactive[str] = reactive("DRY-RUN")
    model_staleness_days: reactive[int] = reactive(0)

    def update_from_snapshot(self, snap: DashboardSnapshot) -> None:
        if snap.halted:
            self.scanner_status = "HALTED"
        elif snap.scanner_paused:
            self.scanner_status = "PAUSED"
        else:
            self.scanner_status = "RUNNING"

        raw = (snap.mode or "dry_run").lower()
        self.mode = "LIVE" if raw == "live" else "DRY-RUN"
        self.model_staleness_days = int(snap.max_component_age_days or 0)
        self.refresh()

    def render(self) -> Text:
        t = Text()

        status = self.scanner_status
        if status == "RUNNING":
            status_style = "bold #00ff41"
        elif status == "PAUSED":
            status_style = "bold #ffab00"
        else:
            status_style = "bold #ff1744"

        t.append("  ◈ SCANNER ", style="#6666aa")
        t.append(status, style=status_style)
        t.append("  │  ", style="#2a2a4a")

        mode_style = "bold #ff00ff" if self.mode == "LIVE" else "bold #00ffcc"
        t.append("MODE ", style="#6666aa")
        t.append(self.mode, style=mode_style)
        t.append("  │  ", style="#2a2a4a")

        days = self.model_staleness_days
        stale_style = "#00ff41" if days <= 7 else "#ffab00" if days <= 14 else "#ff1744"
        t.append("MODELS ", style="#6666aa")
        t.append(f"{days}d old", style=f"bold {stale_style}")

        return t

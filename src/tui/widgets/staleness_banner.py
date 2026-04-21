"""StalenessBanner — red alert above the Overview state strip when any
ensemble component is older than the staleness threshold.

Surfaces the .claude/rules/trading.md (2026-04-15) rule: when models are
> 7 days old the uncertainty hard-block tightens from 0.45 -> 0.35 and
operators need to retrain. Hidden when models are fresh.
"""
from __future__ import annotations

from rich.text import Text
from textual.reactive import reactive
from textual.widgets import Static

from src.tui.data_provider import DashboardSnapshot

STALENESS_THRESHOLD_DAYS: float = 7.0


class StalenessBanner(Static):
    """One-row red alert. display=False until staleness > 7d."""

    DEFAULT_CSS = """
    StalenessBanner {
        height: 1;
        background: #ff1744;
        color: #ffffff;
        text-style: bold;
        padding: 0 1;
    }
    """

    staleness_days: reactive[float] = reactive(0.0)

    def on_mount(self) -> None:
        self.display = False

    def update_from_snapshot(self, snap: DashboardSnapshot) -> None:
        days = float(getattr(snap, "max_component_age_days", 0.0) or 0.0)
        self.staleness_days = days
        self.display = days > STALENESS_THRESHOLD_DAYS
        self.refresh()

    def render(self) -> Text:
        days = self.staleness_days
        t = Text()
        t.append("  ⚠ MODEL STALENESS ALERT ", style="bold #ffffff on #ff1744")
        t.append(
            f" oldest component {days:.0f}d old (> {STALENESS_THRESHOLD_DAYS:.0f}d) — "
            f"uncertainty hard-block tightened to 0.35. RETRAIN REQUIRED.",
            style="bold #ffffff on #ff1744",
        )
        return t

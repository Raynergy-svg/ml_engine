"""
JOURNAL SCREEN (F4) — "The War Room"

Trade history, learning extraction, expectancy stats, P/L curve.
"""
from __future__ import annotations

import json
import random
from datetime import datetime, timezone
from pathlib import Path

from rich.text import Text

from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.widgets import DataTable, Label, RichLog, Sparkline, Static

from src.tui.data_provider import DashboardSnapshot


class ExpectancyPanel(Static):
    """Shows key trading statistics: win rate, expectancy, profit factor."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._stats: dict = {}

    def set_stats(self, stats: dict) -> None:
        self._stats = stats
        self.refresh()

    def render(self) -> Text:
        s = self._stats
        t = Text()
        _m = "#6666aa"

        total = s.get("total_trades", 0)
        wins = s.get("wins", 0)
        losses = s.get("losses", 0)
        wr = s.get("win_rate", 0)
        avg_win = s.get("avg_win", 0)
        avg_loss = s.get("avg_loss", 0)
        expectancy = s.get("expectancy", 0)
        pf = s.get("profit_factor", 0)
        streak = s.get("current_streak", 0)
        streak_type = s.get("streak_type", "—")
        best = s.get("best_trade", 0)
        worst = s.get("worst_trade", 0)

        wr_color = "#00ff41" if wr >= 55 else "#ffab00" if wr >= 45 else "#ff1744"
        exp_color = "#00ff41" if expectancy > 0 else "#ff1744"
        pf_color = "#00ff41" if pf > 1.0 else "#ff1744"
        no_data = total == 0

        t.append("  Total Trades:   ", style=_m)
        t.append(f"{total}\n", style="bold #e0e0ff")
        t.append("  Wins / Losses:  ", style=_m)
        t.append(f"{wins}", style="bold #00ff41")
        t.append(" / ", style=_m)
        t.append(f"{losses}\n", style="bold #ff1744")
        t.append("  Win Rate:       ", style=_m)
        t.append(f"{wr:.1f}%\n", style=f"bold {wr_color}")
        t.append("  ─────────────────────────\n", style="#2a2a4a")
        t.append("  Avg Win:        ", style=_m)
        t.append("—\n" if no_data else f"${avg_win:,.2f}\n",
                 style=_m if no_data else "bold #00ff41")
        t.append("  Avg Loss:       ", style=_m)
        t.append("—\n" if no_data else f"${abs(avg_loss):,.2f}\n",
                 style=_m if no_data else "bold #ff1744")
        t.append("  Expectancy:     ", style=_m)
        t.append("—\n" if no_data else f"${expectancy:,.2f}\n",
                 style=_m if no_data else f"bold {exp_color}")
        t.append("  Profit Factor:  ", style=_m)
        t.append(f"{pf:.2f}\n", style=f"bold {pf_color}")
        t.append("  ─────────────────────────\n", style="#2a2a4a")
        t.append("  Best Trade:     ", style=_m)
        t.append("—\n" if no_data else f"${best:,.2f}\n",
                 style=_m if no_data else "bold #00ff41")
        t.append("  Worst Trade:    ", style=_m)
        t.append("—\n" if no_data else f"${worst:,.2f}\n",
                 style=_m if no_data else "bold #ff1744")
        t.append("  Current Streak: ", style=_m)
        streak_color = "#00ff41" if streak_type == "WIN" else "#ff1744" if streak_type == "LOSS" else _m
        t.append(f"{streak} {streak_type}\n", style=f"bold {streak_color}")

        return t


class LearningsPanel(Static):
    """Shows recent learnings extracted from trade outcomes."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._learnings: list[str] = []

    def load_learnings(self, project_root: Path) -> None:
        """Load learnings from .claude/learnings.md."""
        path = project_root / ".claude" / "learnings.md"
        if not path.exists():
            self._learnings = ["No learnings file found"]
            return
        try:
            text = path.read_text()
            # Extract bullet points (lines starting with -)
            lines = [
                line.strip()
                for line in text.split("\n")
                if line.strip().startswith("- ")
            ]
            self._learnings = lines[-15:] if lines else ["No learnings yet"]
        except Exception:
            self._learnings = ["Error reading learnings"]
        self.refresh()

    def render(self) -> Text:
        t = Text()
        for learning in self._learnings:
            if "[PROMOTED]" in learning:
                t.append(f"  {learning}\n", style="#7c4dff")
            elif "loss" in learning.lower() or "fail" in learning.lower():
                t.append(f"  {learning}\n", style="#ff1744")
            elif "win" in learning.lower() or "improv" in learning.lower():
                t.append(f"  {learning}\n", style="#00ff41")
            else:
                t.append(f"  {learning}\n", style="#6666aa")
        return t


class JournalScreen(Container):
    """The War Room — trade history, stats, learnings, P/L curve."""

    DEFAULT_CSS = """
    JournalScreen {
        height: 1fr;
    }

    #journal-top {
        height: 60%;
        min-height: 12;
    }

    #journal-bottom {
        height: 40%;
        min-height: 8;
    }

    #journal-table-panel {
        width: 2fr;
        border: solid #2a2a4a;
        background: #131320;
        padding: 0 1;
    }

    #journal-stats-panel {
        width: 1fr;
        border: solid #ffab00;
        background: #131320;
        padding: 0 1;
    }

    #journal-learnings-panel {
        width: 1fr;
        border: solid #7c4dff;
        background: #131320;
        padding: 0 1;
    }

    #journal-pnl-panel {
        width: 1fr;
        border: solid #00ffcc;
        background: #131320;
        padding: 0 1;
    }

    #pnl-sparkline {
        min-height: 3;
    }
    """

    def __init__(self, project_root: str = "", **kwargs) -> None:
        super().__init__(**kwargs)
        self._project_root = Path(project_root) if project_root else Path.cwd()
        # US-512: trade_ids parallel to journal-table rows for gate-trace lookup
        self._row_trade_ids: list[str] = []
        # Auto-refresh: mtime of source files captured at last successful load.
        # update_from_snapshot stats both files each tick and only reloads when
        # mtime advances, so trade closures land in the table without restart.
        self._journal_mtime: float = 0.0
        self._learnings_mtime: float = 0.0

    def compose(self) -> ComposeResult:
        with Horizontal(id="journal-top"):
            with Vertical(id="journal-table-panel"):
                yield Label("⟨ TRADE JOURNAL ⟩", classes="panel-title")
                yield DataTable(id="journal-table")

            with Vertical(id="journal-stats-panel"):
                yield Label("⟨ EXPECTANCY STATS ⟩", classes="panel-title")
                yield ExpectancyPanel(id="expectancy-panel")

        with Horizontal(id="journal-bottom"):
            with Vertical(id="journal-learnings-panel"):
                yield Label("⟨ LEARNINGS ⟩  extracted from outcomes",
                           classes="panel-title")
                yield LearningsPanel(id="learnings-panel")

            with Vertical(id="journal-pnl-panel"):
                yield Label("⟨ CUMULATIVE P/L ⟩", classes="panel-title")
                yield Sparkline(data=[], id="pnl-sparkline")
                yield Static(
                    "no closed trades",
                    id="pnl-sparkline-placeholder",
                    classes="status-dim",
                )
                yield Label("", id="pnl-summary", classes="status-dim")

    def on_mount(self) -> None:
        """Load journal data on mount."""
        self._load_journal()
        self._load_learnings()

    def _load_journal(self) -> None:
        """Read trade journal and populate table + stats."""
        journal_path = self._project_root / "trained_data" / "trade_journal_rl.json"

        table = self.query_one("#journal-table", DataTable)
        table.cursor_type = "row"
        if not table.columns:
            table.add_columns("Pair", "Dir", "Conf", "Entry", "Exit", "P/L", "Result", "Time")
        # Drop prior rows before re-populating; otherwise auto-refresh duplicates.
        table.clear()
        self._row_trade_ids = []

        entries = []
        if journal_path.exists():
            try:
                raw = json.loads(journal_path.read_text())
                if isinstance(raw, list):
                    entries = raw
                self._journal_mtime = journal_path.stat().st_mtime
            except Exception:
                pass

        # Populate table with last 50 entries (most recent first)
        pnl_curve = []
        wins = 0
        losses = 0
        total_win_amt = 0.0
        total_loss_amt = 0.0
        best = 0.0
        worst = 0.0
        cumulative = 0.0
        streak = 0
        streak_type = "—"

        for entry in entries:
            outcome = entry.get("outcome") or {}
            if not isinstance(outcome, dict):
                outcome = {}
            # Skip open trades (no outcome) — don't count as losses
            if "realized_pl" not in outcome:
                continue
            pl = outcome.get("realized_pl", 0) or 0
            won = outcome.get("trade_won", False)
            cumulative += pl
            pnl_curve.append(cumulative)

            if won:
                wins += 1
                total_win_amt += pl
                if streak_type == "WIN":
                    streak += 1
                else:
                    streak = 1
                    streak_type = "WIN"
            else:
                losses += 1
                total_loss_amt += abs(pl)
                if streak_type == "LOSS":
                    streak += 1
                else:
                    streak = 1
                    streak_type = "LOSS"

            best = max(best, pl)
            worst = min(worst, pl)

        # Add recent entries to table (newest first)
        for entry in reversed(entries[-50:]):
            outcome = entry.get("outcome") or {}
            if not isinstance(outcome, dict):
                outcome = {}
            pl = outcome.get("realized_pl", 0) or 0
            won = outcome.get("trade_won", False)
            pair = entry.get("pair", "?").replace("_", "/")
            direction = entry.get("direction", "?")[:4]
            agents_data = entry.get("agents") or {}
            if not isinstance(agents_data, dict):
                agents_data = {}
            conf = agents_data.get("weighted_vote_score") or entry.get("confidence") or 0
            entry_p = entry.get("entry_price") or 0
            exit_p = entry.get("exit_price") or 0
            exit_reason = outcome.get("exit_reason", "?")
            entry_time = entry.get("entry_time") or entry.get("timestamp") or ""

            result_str = "WIN" if won else "LOSS"
            pl_str = f"+${pl:,.0f}" if pl >= 0 else f"-${abs(pl):,.0f}"
            time_str = entry_time[:16].replace("T", " ") if entry_time else "—"

            try:
                conf_str = f"{float(conf):.2f}" if conf else "—"
                entry_str = f"{float(entry_p):.5f}" if entry_p else "—"
                exit_str = f"{float(exit_p):.5f}" if exit_p else "—"
            except (ValueError, TypeError):
                conf_str = "—"
                entry_str = "—"
                exit_str = "—"

            table.add_row(
                pair, direction, conf_str,
                entry_str, exit_str,
                pl_str, result_str, time_str,
            )
            self._row_trade_ids.append(str(entry.get("trade_id") or ""))

        # Compute stats
        total = wins + losses
        win_rate = (wins / total * 100) if total > 0 else 0
        avg_win = (total_win_amt / wins) if wins > 0 else 0
        avg_loss = (total_loss_amt / losses) if losses > 0 else 0
        expectancy = (avg_win * win_rate / 100) - (avg_loss * (1 - win_rate / 100)) if total > 0 else 0
        profit_factor = (total_win_amt / total_loss_amt) if total_loss_amt > 0 else 0

        stats = {
            "total_trades": total,
            "wins": wins,
            "losses": losses,
            "win_rate": win_rate,
            "avg_win": avg_win,
            "avg_loss": -avg_loss,
            "expectancy": expectancy,
            "profit_factor": profit_factor,
            "best_trade": best,
            "worst_trade": worst,
            "current_streak": streak,
            "streak_type": streak_type,
        }

        self.query_one("#expectancy-panel", ExpectancyPanel).set_stats(stats)

        # P/L Curve sparkline — show placeholder when no closed trades yet
        pnl_spark = self.query_one("#pnl-sparkline", Sparkline)
        pnl_ph = self.query_one("#pnl-sparkline-placeholder", Static)
        if pnl_curve:
            pnl_spark.display = True
            pnl_ph.display = False
            pnl_spark.data = pnl_curve
            final = pnl_curve[-1]
            summary = f"  Total P/L: ${final:,.2f}  │  {total} trades  │  Win rate: {win_rate:.1f}%"
            self.query_one("#pnl-summary", Label).update(summary)
        else:
            pnl_spark.display = False
            pnl_ph.display = True

    def _load_learnings(self) -> None:
        """Load learnings from .claude/learnings.md."""
        self.query_one("#learnings-panel", LearningsPanel).load_learnings(self._project_root)
        learnings_path = self._project_root / ".claude" / "learnings.md"
        try:
            self._learnings_mtime = learnings_path.stat().st_mtime
        except OSError:
            pass

    def update_from_snapshot(self, snap: DashboardSnapshot) -> None:
        """Auto-refresh on snapshot tick when source files change on disk.

        Stats trade_journal_rl.json + learnings.md each tick; reloads only when
        mtime advances. Replaces the prior no-op stub (the "journal is mostly
        static" lie that caused the screen to stay stale until TUI restart).
        """
        journal_path = self._project_root / "trained_data" / "trade_journal_rl.json"
        learnings_path = self._project_root / ".claude" / "learnings.md"

        try:
            new_journal_mtime = journal_path.stat().st_mtime
        except OSError:
            new_journal_mtime = self._journal_mtime
        try:
            new_learnings_mtime = learnings_path.stat().st_mtime
        except OSError:
            new_learnings_mtime = self._learnings_mtime

        if new_journal_mtime > self._journal_mtime:
            self._load_journal()
        if new_learnings_mtime > self._learnings_mtime:
            self._load_learnings()

    # ------------------------------------------------------------------
    # US-512: Enter on a row -> GateTraceModal
    # ------------------------------------------------------------------

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        from src.tui.screens.gate_trace_modal import GateTraceModal

        row_idx = event.cursor_row
        tid = (
            self._row_trade_ids[row_idx]
            if 0 <= row_idx < len(self._row_trade_ids)
            else ""
        )
        if tid:
            self.app.push_screen(GateTraceModal(trade_id=tid))
        else:
            self.app.notify("gate trace unavailable for this trade", severity="warning")

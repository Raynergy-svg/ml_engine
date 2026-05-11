"""CommandPaletteModal — Vim-style ":" palette for jumping to any BuddyApp action.

Reduces menu-hunting across 8 tabs. Operator presses `:`, types a partial name
(e.g., "kill", "trades", "pause"), Up/Down to pick, Enter to dispatch. Escape
cancels.

The modal is purely a picker — it returns the chosen command id (str). The
BuddyApp.action_open_command_palette() handler routes the id to the matching
`action_*` method. Keeping dispatch in the App avoids coupling the modal to
the rest of the app surface.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Input, Label, ListItem, ListView, Static


@dataclass(frozen=True)
class Command:
    """One entry in the palette."""

    id: str
    label: str
    description: str
    keywords: tuple[str, ...] = ()  # additional aliases for fuzzy matching


def default_commands() -> list[Command]:
    """Static registry of palette commands.

    Mirrors the BINDINGS table in BuddyApp. Any change to BINDINGS should be
    reflected here so the palette stays in sync.
    """
    return [
        # Navigation (F1–F8 equivalents)
        Command("nav.overview", "Go to Overview", "Live trades + agent consensus", ("f1", "home")),
        Command("nav.inbox", "Go to Inbox", "Homework + adjustments queue", ("f2", "homework")),
        Command("nav.trades", "Go to Trades", "Open positions + closed journal", ("f3", "positions")),
        Command("nav.agents", "Go to Agents", "Agent weights + vote breakdown", ("f4", "weights")),
        Command("nav.journal", "Go to Journal", "Full closed-trade journal", ("f5",)),
        Command("nav.config", "Go to Config", "ScannerConfig tunables", ("f6", "settings", "tune")),
        Command("nav.rules", "Go to Rules", "trading.md / improvement.md viewer", ("f7",)),
        Command("nav.diag", "Go to Diagnostics", "Models, staleness, ensemble", ("f8", "diagnostics", "health")),
        # Supervisor
        Command("supervisor.pause", "Pause/Resume scanner", "Toggle StateEngine pause", ("space", "resume")),
        Command("supervisor.kill", "Kill switch — flatten all", "Type KILL to confirm", ("flatten", "stop", "panic")),
        Command("supervisor.mode", "Toggle Live/Demo mode", "Type LIVE to switch to live", ("live", "demo", "switch")),
        Command("supervisor.abort", "Abort pending signal", "Veto last pending signal", ("veto", "cancel")),
        # Broker
        Command("broker.cycle", "Cycle asset class", "FX → Futures → Hybrid", ("fx", "futures", "hybrid", "broker")),
        # App
        Command("app.quit", "Quit Buddy", "Clean shutdown", ("exit", "close")),
    ]


def filter_commands(query: str, commands: list[Command]) -> list[Command]:
    """Pure substring fuzzy filter — case-insensitive, all whitespace-separated
    tokens must appear in either the label or one of the keywords or the id.

    Empty query returns the full list unchanged.
    """
    q = query.strip().lower()
    if not q:
        return list(commands)
    tokens = q.split()
    out: list[Command] = []
    for cmd in commands:
        haystack = " ".join((cmd.id, cmd.label.lower(), *cmd.keywords))
        if all(t in haystack for t in tokens):
            out.append(cmd)
    return out


class CommandPaletteModal(ModalScreen[Optional[str]]):
    """Pop-up command palette. Dismisses with the chosen command id or None."""

    BINDINGS = [
        Binding("escape", "dismiss_cancel", "Cancel", show=True),
        Binding("up", "move_selection(-1)", "Up", show=False),
        Binding("down", "move_selection(1)", "Down", show=False),
        Binding("enter", "dispatch_selected", "Run", show=True),
    ]

    DEFAULT_CSS = """
    CommandPaletteModal {
        align: center middle;
    }

    #palette-dialog {
        width: 70;
        height: auto;
        max-height: 24;
        background: #0a0a0f;
        border: double #00ffcc;
        padding: 1 2;
    }

    #palette-title {
        text-align: center;
        color: #00ffcc;
        text-style: bold;
        padding: 0 0 1 0;
    }

    #palette-input {
        margin: 0 0 1 0;
        border: solid #00ffcc;
        background: #131320;
    }

    #palette-list {
        height: auto;
        max-height: 14;
        border: solid #2a2a3a;
        background: #07070b;
    }

    #palette-empty {
        color: #6666aa;
        text-align: center;
        padding: 1;
    }

    #palette-hint {
        color: #6666aa;
        text-align: center;
        padding: 1 0 0 0;
    }
    """

    def __init__(self, commands: Optional[list[Command]] = None) -> None:
        super().__init__()
        self._commands = commands if commands is not None else default_commands()
        self._filtered: list[Command] = list(self._commands)

    def compose(self) -> ComposeResult:
        with Vertical(id="palette-dialog"):
            yield Label("⌘  COMMAND PALETTE  ⌘", id="palette-title")
            yield Input(placeholder="Type to filter — Enter to run, Esc to cancel", id="palette-input")
            yield ListView(id="palette-list")
            yield Static("[dim]↑↓ move · Enter run · Esc cancel[/]", id="palette-hint", markup=True)

    def on_mount(self) -> None:
        self._rebuild_list("")
        self.query_one("#palette-input", Input).focus()

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id == "palette-input":
            self._rebuild_list(event.value)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id == "palette-input":
            self.action_dispatch_selected()

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        # Mouse click / double-Enter on a row.
        self.action_dispatch_selected()

    def _rebuild_list(self, query: str) -> None:
        self._filtered = filter_commands(query, self._commands)
        lv = self.query_one("#palette-list", ListView)
        lv.clear()
        if not self._filtered:
            lv.append(ListItem(Static("  no matches  ", id="palette-empty")))
            return
        for cmd in self._filtered:
            label = f"[b]{cmd.label}[/]  [dim]· {cmd.description}[/]"
            lv.append(ListItem(Static(label, markup=True)))
        lv.index = 0

    def action_move_selection(self, delta: int) -> None:
        lv = self.query_one("#palette-list", ListView)
        if not self._filtered:
            return
        new = (lv.index or 0) + delta
        new = max(0, min(len(self._filtered) - 1, new))
        lv.index = new

    def action_dispatch_selected(self) -> None:
        lv = self.query_one("#palette-list", ListView)
        if not self._filtered:
            return
        idx = lv.index or 0
        if 0 <= idx < len(self._filtered):
            self.dismiss(self._filtered[idx].id)

    def action_dismiss_cancel(self) -> None:
        self.dismiss(None)

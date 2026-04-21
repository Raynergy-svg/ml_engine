"""KillModal — double-confirm kill switch overlay for the Buddy TUI.

Requires the operator to type 'KILL' (case-sensitive) before the Confirm
button enables. Displays live trade count, unrealized P/L, and margin
exposure pulled directly from the current DashboardSnapshot.
"""
from __future__ import annotations

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Label, Static

from src.tui.data_provider import DashboardSnapshot


class KillModal(ModalScreen[bool]):
    """Double-confirm modal: operator must type KILL to enable the Confirm button.

    Returns True on confirm, False on cancel/escape.
    """

    BINDINGS = [Binding("escape", "dismiss_cancel", "Cancel", show=True)]

    DEFAULT_CSS = """
    KillModal {
        align: center middle;
    }

    #kill-dialog {
        width: 62;
        height: auto;
        background: #0a0a0f;
        border: double #ff1744;
        padding: 1 2;
    }

    #kill-title {
        text-align: center;
        color: #ff1744;
        text-style: bold;
        padding: 0 0 1 0;
    }

    #kill-warning {
        text-align: center;
        color: #ffab00;
        padding: 0 0 1 0;
    }

    #kill-stats {
        padding: 0 0 1 0;
        color: #6666aa;
    }

    #kill-prompt {
        color: #ffab00;
        padding: 1 0 0 0;
    }

    #kill-input {
        margin: 0 0 1 0;
        border: solid #ff1744;
        background: #131320;
    }

    #kill-buttons {
        align: center middle;
        height: 3;
    }

    #kill-confirm {
        margin: 0 1;
    }

    #kill-cancel {
        margin: 0 1;
    }
    """

    def __init__(self, snap: DashboardSnapshot) -> None:
        super().__init__()
        self._snap = snap

    def compose(self) -> ComposeResult:
        snap = self._snap
        trade_count = len(snap.trades)
        pnl = snap.unrealized_pnl
        pnl_str = f"+${pnl:,.2f}" if pnl >= 0 else f"-${abs(pnl):,.2f}"
        exposure_str = f"${snap.margin_used:,.2f}"

        with Vertical(id="kill-dialog"):
            yield Label("⚠  KILL SWITCH  ⚠", id="kill-title")
            yield Label(
                "This will FLATTEN ALL OPEN POSITIONS and HALT the scanner.",
                id="kill-warning",
            )
            yield Static(
                f"  Open trades:     {trade_count}\n"
                f"  Unrealized P/L:  {pnl_str}\n"
                f"  Margin exposure: {exposure_str}",
                id="kill-stats",
            )
            yield Label(
                "Type [bold red]KILL[/] (case-sensitive) to confirm:",
                id="kill-prompt",
                markup=True,
            )
            yield Input(placeholder="KILL", id="kill-input")
            with Horizontal(id="kill-buttons"):
                yield Button("⚡ Confirm", id="kill-confirm", disabled=True, variant="error")
                yield Button("✕ Cancel", id="kill-cancel", variant="default")

    def on_input_changed(self, event: Input.Changed) -> None:
        confirm_btn = self.query_one("#kill-confirm", Button)
        confirm_btn.disabled = event.value != "KILL"

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "kill-confirm" and not event.button.disabled:
            self.dismiss(True)
        elif event.button.id == "kill-cancel":
            self.dismiss(False)

    def action_dismiss_cancel(self) -> None:
        self.dismiss(False)

"""ModeConfirmModal — typed confirmation for dry_run → live mode upgrade.

The operator must type 'LIVE' (case-sensitive) to enable the Confirm button.
Credential pre-flight (OANDA_API_TOKEN/OANDA_API_KEY + OANDA_ACCOUNT_ID) is
checked by action_supervisor_mode before this modal is pushed; the modal
surfaces the same credential status for operator awareness.

Returns True on confirm, False on cancel/escape.
"""
from __future__ import annotations

import os

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Label, Static


def check_oanda_credentials() -> tuple[bool, str]:
    """Return (ok, message). Both API key and account ID must be present."""
    api_key = os.environ.get("OANDA_API_TOKEN") or os.environ.get("OANDA_API_KEY")
    account_id = os.environ.get("OANDA_ACCOUNT_ID")
    if not api_key and not account_id:
        return False, "OANDA_API_TOKEN and OANDA_ACCOUNT_ID not set"
    if not api_key:
        return False, "OANDA_API_TOKEN / OANDA_API_KEY not set"
    if not account_id:
        return False, "OANDA_ACCOUNT_ID not set"
    key_preview = api_key[:6] + "…" if len(api_key) > 6 else "…"
    return True, f"key …{key_preview}  account {account_id}"


class ModeConfirmModal(ModalScreen[bool]):
    """Dry-run → live upgrade confirmation. Operator must type 'LIVE'.

    Returns True on confirm, False on cancel/escape.
    """

    BINDINGS = [Binding("escape", "dismiss_cancel", "Cancel", show=True)]

    DEFAULT_CSS = """
    ModeConfirmModal {
        align: center middle;
    }

    #mode-dialog {
        width: 64;
        height: auto;
        background: #0a0a0f;
        border: double #ff00ff;
        padding: 1 2;
    }

    #mode-title {
        text-align: center;
        color: #ff00ff;
        text-style: bold;
        padding: 0 0 1 0;
    }

    #mode-warning {
        text-align: center;
        color: #ffab00;
        padding: 0 0 1 0;
    }

    #mode-stats {
        padding: 0 0 1 0;
        color: #6666aa;
    }

    #mode-prompt {
        color: #ffab00;
        padding: 1 0 0 0;
    }

    #mode-input {
        margin: 0 0 1 0;
        border: solid #ff00ff;
        background: #131320;
    }

    #mode-buttons {
        align: center middle;
        height: 3;
    }

    #mode-confirm {
        margin: 0 1;
    }

    #mode-cancel {
        margin: 0 1;
    }
    """

    def __init__(self, nav: float = 0.0, open_trades: int = 0) -> None:
        super().__init__()
        self._nav = nav
        self._open_trades = open_trades

    def compose(self) -> ComposeResult:
        cred_ok, cred_msg = check_oanda_credentials()
        cred_icon = "✓" if cred_ok else "✗"

        with Vertical(id="mode-dialog"):
            yield Label("⚡  ACTIVATE LIVE TRADING  ⚡", id="mode-title")
            yield Label(
                "Real capital at risk. OANDA execution will be enabled.",
                id="mode-warning",
            )
            yield Static(
                f"  {cred_icon} Credentials: {cred_msg}\n"
                f"  NAV: ${self._nav:,.2f}   Open positions: {self._open_trades}",
                id="mode-stats",
            )
            yield Label(
                "Type [bold magenta]LIVE[/] (case-sensitive) to confirm:",
                id="mode-prompt",
                markup=True,
            )
            yield Input(placeholder="LIVE", id="mode-input")
            with Horizontal(id="mode-buttons"):
                yield Button("⚡ Go Live", id="mode-confirm", disabled=True, variant="warning")
                yield Button("✕ Cancel", id="mode-cancel", variant="default")

    def on_input_changed(self, event: Input.Changed) -> None:
        confirm_btn = self.query_one("#mode-confirm", Button)
        confirm_btn.disabled = event.value != "LIVE"

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "mode-confirm" and not event.button.disabled:
            self.dismiss(True)
        elif event.button.id == "mode-cancel":
            self.dismiss(False)

    def action_dismiss_cancel(self) -> None:
        self.dismiss(False)

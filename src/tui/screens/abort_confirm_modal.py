"""AbortConfirmModal — single-confirm modal for pre-trade signal veto.

Per docs/supervisor_console_runbook.md §3.4: confirms the pending signal that
will be aborted (pair, direction, age in replay buffer). ENTER to confirm,
ESC to cancel. If no signal.pending exists in the replay buffer, the modal
shows 'no pending signal' and only ESC works.

Pattern mirrors KillModal but without the type-to-confirm requirement —
abort is reversible (next signal still gates), kill is not.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Label, Static


def _signal_age_seconds(event: dict) -> Optional[float]:
    """Compute age of a replay-buffer event in seconds.

    Returns None if the event has no parseable 'ts' field — caller should
    render '—' rather than guess.
    """
    ts = event.get("ts")
    if not isinstance(ts, str) or not ts:
        return None
    try:
        emitted = datetime.fromisoformat(ts)
    except (ValueError, TypeError):
        return None
    if emitted.tzinfo is None:
        emitted = emitted.replace(tzinfo=timezone.utc)
    delta = datetime.now(timezone.utc) - emitted
    return max(0.0, delta.total_seconds())


class AbortConfirmModal(ModalScreen[bool]):
    """Single-confirm modal for operator pre-trade signal abort.

    Returns True on confirm (ENTER or Confirm button), False on cancel
    (ESC or Cancel button). If constructed with event=None, only the
    cancel path is reachable — the confirm button is disabled.
    """

    BINDINGS = [
        Binding("escape", "dismiss_cancel", "Cancel", show=True),
        Binding("enter", "dismiss_confirm", "Confirm", show=True),
    ]

    DEFAULT_CSS = """
    AbortConfirmModal {
        align: center middle;
    }

    #abort-dialog {
        width: 60;
        height: auto;
        background: #0a0a0f;
        border: double #ffab00;
        padding: 1 2;
    }

    #abort-title {
        text-align: center;
        color: #ffab00;
        text-style: bold;
        padding: 0 0 1 0;
    }

    #abort-empty {
        text-align: center;
        color: #6666aa;
        padding: 1 0;
    }

    #abort-stats {
        padding: 0 0 1 0;
        color: #aaccff;
    }

    #abort-prompt {
        color: #ffab00;
        padding: 1 0 0 0;
        text-align: center;
    }

    #abort-buttons {
        align: center middle;
        height: 3;
    }

    #abort-confirm {
        margin: 0 1;
    }

    #abort-cancel {
        margin: 0 1;
    }
    """

    def __init__(self, event: Optional[dict]) -> None:
        """event: the full replay-buffer entry (with 'ts' + 'payload') or None.

        Passing None renders the empty-state layout — the operator can only
        dismiss; no veto event is publishable.
        """
        super().__init__()
        self._event = event

    def compose(self) -> ComposeResult:
        with Vertical(id="abort-dialog"):
            yield Label("⏸  ABORT PENDING SIGNAL  ⏸", id="abort-title")
            if self._event is None:
                yield Static("no pending signal", id="abort-empty")
                with Horizontal(id="abort-buttons"):
                    # Confirm disabled so ENTER cannot fire a veto on empty state.
                    yield Button("✓ Confirm", id="abort-confirm", disabled=True, variant="warning")
                    yield Button("✕ Cancel", id="abort-cancel", variant="default")
            else:
                payload = self._event.get("payload", {}) or {}
                pair = str(payload.get("pair", "?"))
                direction = str(payload.get("direction", "?"))
                signal_id = str(payload.get("signal_id", ""))
                age_s = _signal_age_seconds(self._event)
                age_str = f"{age_s:.1f}s" if age_s is not None else "—"

                stats_lines = [
                    f"  Pair:       {pair}",
                    f"  Direction:  {direction}",
                    f"  Age:        {age_str}",
                ]
                if signal_id:
                    stats_lines.append(f"  Signal ID:  {signal_id[:12]}…")
                yield Static("\n".join(stats_lines), id="abort-stats")
                yield Label(
                    "Press [bold yellow]ENTER[/] to abort, [dim]ESC[/] to cancel",
                    id="abort-prompt",
                    markup=True,
                )
                with Horizontal(id="abort-buttons"):
                    yield Button("✓ Confirm", id="abort-confirm", variant="warning")
                    yield Button("✕ Cancel", id="abort-cancel", variant="default")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "abort-confirm" and not event.button.disabled:
            self.dismiss(True)
        elif event.button.id == "abort-cancel":
            self.dismiss(False)

    def action_dismiss_cancel(self) -> None:
        self.dismiss(False)

    def action_dismiss_confirm(self) -> None:
        # ENTER must be inert on the empty-state layout — no pending event = no veto.
        if self._event is None:
            return
        self.dismiss(True)

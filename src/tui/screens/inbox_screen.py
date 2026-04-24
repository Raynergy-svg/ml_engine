"""
INBOX SCREEN (F2) — Adjustment Inbox

Morning operator dashboard: lists pending config proposals before Buddy runs.
A/R/S/V row actions let the operator approve, reject, snooze, or inspect each.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

from rich.text import Text

from textual import work
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, DataTable, Input, Label, Static, TabbedContent

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_PENDING_PATH = _PROJECT_ROOT / ".claude" / "pending_adjustments.json"
_APPROVED_PATH = _PROJECT_ROOT / ".claude" / "config_adjustments.json"

# Proposals in terminal states are not shown (approved already moved out, rejected is done)
_ACTIVE_STATUSES = ("pending", "snoozed", "invalid")


# ── Modals ───────────────────────────────────────────────────────────


class RejectModal(ModalScreen[Optional[str]]):
    """Small modal to capture rejection reason. Returns reason string or None on cancel."""

    BINDINGS = [Binding("escape", "cancel", "Cancel", show=True)]

    DEFAULT_CSS = """
    RejectModal {
        align: center middle;
    }

    #reject-dialog {
        width: 60;
        height: auto;
        background: #0a0a0f;
        border: double #ff1744;
        padding: 1 2;
    }

    #reject-title {
        text-align: center;
        color: #ff1744;
        text-style: bold;
        padding: 0 0 1 0;
    }

    #reject-prompt {
        color: #ffab00;
        padding: 0 0 0 0;
    }

    #reject-input {
        margin: 1 0 1 0;
        border: solid #ff1744;
        background: #131320;
    }

    #reject-buttons {
        align: center middle;
        height: 3;
    }

    #reject-confirm {
        margin: 0 1;
    }

    #reject-cancel-btn {
        margin: 0 1;
    }
    """

    def __init__(self, proposal_key: str) -> None:
        super().__init__()
        self._proposal_key = proposal_key

    def compose(self) -> ComposeResult:
        with Vertical(id="reject-dialog"):
            yield Label(f"✗ Reject '{self._proposal_key}'", id="reject-title")
            yield Label("Reason for rejection:", id="reject-prompt")
            yield Input(placeholder="e.g. bounds too aggressive", id="reject-input")
            with Horizontal(id="reject-buttons"):
                yield Button("✕ Reject", id="reject-confirm", variant="error")
                yield Button("← Back", id="reject-cancel-btn", variant="default")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "reject-confirm":
            reason = self.query_one("#reject-input", Input).value or "(no reason)"
            self.dismiss(reason)
        elif event.button.id == "reject-cancel-btn":
            self.dismiss(None)

    def action_cancel(self) -> None:
        self.dismiss(None)


class DetailModal(ModalScreen[None]):
    """Shows full proposal JSON detail."""

    BINDINGS = [Binding("escape", "close", "Close", show=True)]

    DEFAULT_CSS = """
    DetailModal {
        align: center middle;
    }

    #detail-dialog {
        width: 72;
        height: auto;
        max-height: 40;
        background: #0a0a0f;
        border: double #7c4dff;
        padding: 1 2;
        overflow-y: auto;
    }

    #detail-title {
        text-align: center;
        color: #7c4dff;
        text-style: bold;
        padding: 0 0 1 0;
    }

    #detail-body {
        color: #e0e0ff;
        padding: 0 0 1 0;
    }

    #detail-close {
        align: center middle;
        height: 3;
    }
    """

    def __init__(self, proposal: dict[str, Any]) -> None:
        super().__init__()
        self._proposal = proposal

    def compose(self) -> ComposeResult:
        p = self._proposal
        val = p.get("validation") or {}
        status = p.get("status", "?")
        snooze = p.get("snooze_until", "")
        ts = (p.get("timestamp") or "")[:19].replace("T", " ")
        detail_lines = [
            f"  ID:          {p.get('id', '?')[:16]}…",
            f"  Timestamp:   {ts}",
            f"  Field:       {p.get('key', '?')}",
            f"  Current:     {p.get('current_value', '—')}",
            f"  Proposed:    {p.get('proposed_value', '?')}",
            f"  Source:      {p.get('source', '?')}",
            f"  Status:      {status}",
            f"  Snooze:      {snooze or '—'}",
            "",
            "  ── Validation ─────────────────────────────",
            f"  Valid:       {'✓' if val.get('valid') else '✗'}",
            f"  Field exists:{' ✓' if val.get('field_exists') else ' ✗'}",
            f"  Type ok:     {'✓' if val.get('type_matches') else '✗'}",
            f"  In bounds:   {'✓' if val.get('within_bounds') else '✗'}",
            f"  Error:       {val.get('error_message') or '—'}",
            "",
            "  ── Reason ─────────────────────────────────",
            f"  {p.get('reason', '—')}",
        ]
        with Vertical(id="detail-dialog"):
            yield Label("◈ PROPOSAL DETAIL", id="detail-title")
            yield Static("\n".join(detail_lines), id="detail-body")
            with Horizontal(id="detail-close"):
                yield Button("← Close  [Esc]", id="detail-close-btn", variant="default")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        self.dismiss()

    def action_close(self) -> None:
        self.dismiss()


class PreviewModal(ModalScreen[None]):
    """Shows merged ScannerConfig diff if all pending valid proposals are approved."""

    BINDINGS = [Binding("escape", "close", "Close", show=True)]

    DEFAULT_CSS = """
    PreviewModal {
        align: center middle;
    }

    #preview-dialog {
        width: 76;
        height: auto;
        max-height: 42;
        background: #0a0a0f;
        border: double #00ff41;
        padding: 1 2;
        overflow-y: auto;
    }

    #preview-title {
        text-align: center;
        color: #00ff41;
        text-style: bold;
        padding: 0 0 1 0;
    }

    #preview-body {
        color: #e0e0ff;
        padding: 0 0 1 0;
    }

    #preview-close {
        align: center middle;
        height: 3;
    }
    """

    def __init__(self, proposals: list[dict[str, Any]]) -> None:
        super().__init__()
        self._proposals = proposals

    def compose(self) -> ComposeResult:
        lines = self._build_diff()
        with Vertical(id="preview-dialog"):
            yield Label("◈ NET CONFIG PREVIEW — if all pending approved", id="preview-title")
            yield Static("\n".join(lines), id="preview-body")
            with Horizontal(id="preview-close"):
                yield Button("← Close  [Esc]", id="preview-close-btn", variant="default")

    def _build_diff(self) -> list[str]:
        """Compute what ScannerConfig would look like if all pending valid proposals applied."""
        try:
            from src.scanner.config import ScannerConfig
            cfg = ScannerConfig()
        except Exception as e:
            return [f"  [Cannot load ScannerConfig: {e}]"]

        valid_pending = [
            p for p in self._proposals
            if p.get("status") == "pending"
            and (p.get("validation") or {}).get("valid")
        ]

        if not valid_pending:
            return ["  No valid pending proposals to preview."]

        lines = [f"  {'Field':<35} {'Current':<18} → {'Proposed':<18}"]
        lines.append("  " + "─" * 75)

        for p in valid_pending:
            key = p.get("key", "?")
            proposed = p.get("proposed_value")
            current = getattr(cfg, key, "?")
            line = f"  {key:<35} {str(current):<18} → {str(proposed):<18}"
            lines.append(line)

        lines.append("")
        lines.append(f"  {len(valid_pending)} field(s) would change.")
        return lines

    def on_button_pressed(self, event: Button.Pressed) -> None:
        self.dismiss()

    def action_close(self) -> None:
        self.dismiss()


# ── InboxScreen ──────────────────────────────────────────────────────


class InboxScreen(Container):
    """
    F2 — Adjustment Inbox: morning operator review before Buddy runs.

    Lists all actionable config proposals from pending_adjustments.json.
    Operator approves/rejects/snoozes each proposal. Approved proposals
    move to config_adjustments.json via AdjustmentApprover.approve().
    """

    BINDINGS = [
        Binding("a", "approve", "Approve", show=True),
        Binding("r", "reject", "Reject", show=True),
        Binding("s", "snooze", "Snooze 24h", show=True),
        Binding("v", "view_detail", "View Detail", show=True),
    ]

    DEFAULT_CSS = """
    InboxScreen {
        height: 1fr;
        padding: 0 1;
    }

    #inbox-header {
        height: 3;
        align: left middle;
    }

    #inbox-title {
        color: #ff00ff;
        text-style: bold;
        padding: 0 1;
    }

    #inbox-subtitle {
        color: #6666aa;
        padding: 0 2;
    }

    #inbox-table-panel {
        height: 1fr;
        border: solid #2a2a4a;
        background: #131320;
        padding: 0 1;
    }

    #inbox-actions {
        height: 3;
        align: center middle;
        padding: 0 1;
    }

    .inbox-action-btn {
        margin: 0 1;
    }

    #inbox-empty {
        height: 3;
        align: center middle;
        color: #6666aa;
        text-align: center;
        display: none;
    }

    #inbox-table {
        height: 1fr;
    }
    """

    def __init__(self, project_root: str = "", **kwargs) -> None:
        super().__init__(**kwargs)
        self._project_root = Path(project_root) if project_root else _PROJECT_ROOT
        self._pending_path = self._project_root / ".claude" / "pending_adjustments.json"
        self._approved_path = self._project_root / ".claude" / "config_adjustments.json"
        # Proposal list (all actionable statuses)
        self._proposals: list[dict[str, Any]] = []
        # Map from DataTable row_key → proposal dict
        self._row_to_proposal: dict[str, dict[str, Any]] = {}

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def compose(self) -> ComposeResult:
        with Horizontal(id="inbox-header"):
            yield Label("◈ ADJUSTMENT INBOX", id="inbox-title")
            yield Label("A approve  R reject  S snooze 24h  V detail", id="inbox-subtitle")

        with Vertical(id="inbox-table-panel"):
            yield DataTable(id="inbox-table", cursor_type="row")
            yield Static(
                "No pending adjustments. Buddy is running on current config.",
                id="inbox-empty",
            )

        with Horizontal(id="inbox-actions"):
            yield Button("◈ Preview net config", id="preview-btn", variant="default",
                         classes="inbox-action-btn")
            yield Button("✓ Approve all valid", id="approve-all-btn", variant="success",
                         classes="inbox-action-btn")
            yield Button("✕ Reject all", id="reject-all-btn", variant="error",
                         classes="inbox-action-btn")

    def on_mount(self) -> None:
        table = self.query_one("#inbox-table", DataTable)
        table.add_columns(
            "Timestamp", "Field", "Current", "Proposed",
            "Reason", "Source", "Validation", "Status",
        )
        self._load_proposals()
        self.set_interval(5.0, self._load_proposals)
        self._subscribe_events()

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def _load_proposals(self) -> None:
        """Read pending_adjustments.json and refresh the table."""
        proposals = self._read_proposals()
        self._proposals = proposals
        self._rebuild_table(proposals)
        self._update_tab_label(len([p for p in proposals if p.get("status") == "pending"]))

    def _read_proposals(self) -> list[dict[str, Any]]:
        """Load actionable proposals from pending_adjustments.json."""
        if not self._pending_path.exists():
            return []
        try:
            data = json.loads(self._pending_path.read_text())
            all_proposals = data.get("proposals", [])
            return [p for p in all_proposals if p.get("status") in _ACTIVE_STATUSES]
        except Exception as e:
            logger.warning("InboxScreen: failed to load proposals: %s", e)
            return []

    def _rebuild_table(self, proposals: list[dict[str, Any]]) -> None:
        """Repopulate the DataTable from the proposal list."""
        table = self.query_one("#inbox-table", DataTable)
        table.clear()
        self._row_to_proposal = {}

        try:
            empty_label = self.query_one("#inbox-empty", Static)
            empty_label.display = not bool(proposals)
            table.display = bool(proposals)
        except Exception:
            pass

        if not proposals:
            return

        # Try to read current ScannerConfig defaults for the "Current" column
        current_values = self._get_current_config_values()

        for proposal in proposals:
            proposal_id = proposal.get("id", "")
            key = proposal.get("key", "?")
            proposed_val = proposal.get("proposed_value")
            status = proposal.get("status", "?")
            val_dict = proposal.get("validation") or {}
            is_valid = val_dict.get("valid", False)
            ts = (proposal.get("timestamp") or "")[:16].replace("T", " ")
            reason = (proposal.get("reason") or "")[:40]
            source = proposal.get("source", "?")
            current_val = current_values.get(key, "?")

            # Validation display
            if status == "invalid" or not is_valid:
                val_display = "✗ INVALID"
            elif status == "snoozed":
                snooze_until = proposal.get("snooze_until", "")
                snooze_str = snooze_until[:16].replace("T", " ") if snooze_until else "?"
                val_display = f"⏸ snoozed → {snooze_str}"
            else:
                val_display = "✓ VALID"

            # Status display
            status_display = status.upper()

            row_key = proposal_id or f"row-{key}"
            table.add_row(
                ts,
                key,
                str(current_val),
                str(proposed_val),
                reason,
                source,
                val_display,
                status_display,
                key=row_key,
            )
            self._row_to_proposal[row_key] = proposal

        # Highlight invalid rows — apply style to validation column
        self._highlight_invalid_rows(proposals)

    def _highlight_invalid_rows(self, proposals: list[dict[str, Any]]) -> None:
        """Apply red-text styled cells to invalid rows via Rich Text."""
        table = self.query_one("#inbox-table", DataTable)
        for proposal in proposals:
            val_dict = proposal.get("validation") or {}
            status = proposal.get("status", "")
            is_invalid = status == "invalid" or not val_dict.get("valid", False)
            if not is_invalid:
                continue
            proposal_id = proposal.get("id", "")
            row_key = proposal_id or f"row-{proposal.get('key', '')}"
            key_col = 1  # "Field" column index
            try:
                table.update_cell(row_key, "Field", Text(proposal.get("key", "?"), style="bold #ff1744"))
                table.update_cell(row_key, "Status", Text("INVALID", style="bold #ff1744"))
                table.update_cell(row_key, "Validation", Text("✗ INVALID", style="bold #ff1744"))
            except Exception:
                pass

    def _get_current_config_values(self) -> dict[str, Any]:
        """Return {field_name: current_value} from a fresh ScannerConfig instance."""
        try:
            from src.scanner.config import ScannerConfig
            cfg = ScannerConfig()
            import dataclasses
            return {f.name: getattr(cfg, f.name) for f in dataclasses.fields(cfg)}
        except Exception:
            return {}

    # ------------------------------------------------------------------
    # Actions (BINDINGS)
    # ------------------------------------------------------------------

    def action_approve(self) -> None:
        """A — approve the focused proposal (only if validation.valid == True)."""
        proposal = self._focused_proposal()
        if proposal is None:
            self.notify("No proposal selected.", severity="warning")
            return

        val_dict = proposal.get("validation") or {}
        if not val_dict.get("valid"):
            self.notify(
                f"Cannot approve: {val_dict.get('error_message', 'invalid proposal')}",
                severity="error",
            )
            return

        if proposal.get("status") not in ("pending", "snoozed"):
            self.notify(f"Proposal is {proposal.get('status')} — cannot approve.", severity="warning")
            return

        proposal_id = proposal.get("id", "")
        try:
            from src.scanner.automation.adjustment_approver import AdjustmentApprover
            approver = AdjustmentApprover(
                pending_path=self._pending_path,
                approved_path=self._approved_path,
            )
            ok = approver.approve(proposal_id)
            if ok:
                key = proposal.get("key", "?")
                val = proposal.get("proposed_value")
                self.notify(f"✓ Approved: {key} = {val}", severity="information")
                self._load_proposals()
            else:
                self.notify("Approval failed — check logs.", severity="error")
        except Exception as e:
            self.notify(f"Approval error: {e}", severity="error")
            logger.exception("InboxScreen.action_approve failed")

    def action_reject(self) -> None:
        """R — open reject modal to capture reason, then reject."""
        proposal = self._focused_proposal()
        if proposal is None:
            self.notify("No proposal selected.", severity="warning")
            return

        key = proposal.get("key", "?")
        self.app.push_screen(RejectModal(key), callback=self._on_reject_result)

    def _on_reject_result(self, reason: Optional[str]) -> None:
        if reason is None:
            return
        proposal = self._focused_proposal()
        if proposal is None:
            return
        proposal_id = proposal.get("id", "")
        try:
            from src.scanner.automation.adjustment_approver import AdjustmentApprover
            approver = AdjustmentApprover(
                pending_path=self._pending_path,
                approved_path=self._approved_path,
            )
            ok = approver.reject(proposal_id, reason)
            if ok:
                self.notify(f"✗ Rejected: {proposal.get('key', '?')}", severity="information")
                self._load_proposals()
            else:
                self.notify("Reject failed — proposal not found.", severity="error")
        except Exception as e:
            self.notify(f"Reject error: {e}", severity="error")
            logger.exception("InboxScreen.action_reject failed")

    def action_snooze(self) -> None:
        """S — snooze the focused proposal for 24 hours."""
        proposal = self._focused_proposal()
        if proposal is None:
            self.notify("No proposal selected.", severity="warning")
            return

        proposal_id = proposal.get("id", "")
        try:
            from src.scanner.automation.adjustment_approver import AdjustmentApprover
            approver = AdjustmentApprover(
                pending_path=self._pending_path,
                approved_path=self._approved_path,
            )
            ok = approver.snooze(proposal_id, hours=24)
            if ok:
                key = proposal.get("key", "?")
                snooze_until = (datetime.now(timezone.utc) + timedelta(hours=24)).strftime("%Y-%m-%d %H:%M")
                self.notify(f"⏸ Snoozed '{key}' until {snooze_until} UTC", severity="information")
                self._load_proposals()
            else:
                self.notify("Snooze failed — proposal not found.", severity="error")
        except Exception as e:
            self.notify(f"Snooze error: {e}", severity="error")
            logger.exception("InboxScreen.action_snooze failed")

    def action_view_detail(self) -> None:
        """V — show full proposal detail in a modal."""
        proposal = self._focused_proposal()
        if proposal is None:
            self.notify("No proposal selected.", severity="warning")
            return
        self.app.push_screen(DetailModal(proposal))

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "preview-btn":
            self.app.push_screen(PreviewModal(self._proposals))
        elif event.button.id == "approve-all-btn":
            self.action_approve_all()
        elif event.button.id == "reject-all-btn":
            self.action_reject_all()

    def action_approve_all(self) -> None:
        """Approve every valid pending/snoozed proposal."""
        try:
            from src.scanner.automation.adjustment_approver import AdjustmentApprover
            approver = AdjustmentApprover(
                pending_path=self._pending_path,
                approved_path=self._approved_path,
            )
            result = approver.approve_all()
            approved = result.get("approved", 0)
            skipped = result.get("skipped", 0)
            failed = len(result.get("failed", []))
            if failed:
                self.notify(
                    f"Approved {approved}; {failed} failed; {skipped} invalid skipped.",
                    severity="warning",
                )
            else:
                self.notify(
                    f"Approved {approved} proposal(s); {skipped} invalid skipped.",
                    severity="information",
                )
            self._load_proposals()
        except Exception as e:
            self.notify(f"Approve all error: {e}", severity="error")
            logger.exception("InboxScreen.action_approve_all failed")

    def action_reject_all(self) -> None:
        """Reject every actionable proposal in the inbox."""
        try:
            from src.scanner.automation.adjustment_approver import AdjustmentApprover
            approver = AdjustmentApprover(
                pending_path=self._pending_path,
                approved_path=self._approved_path,
            )
            result = approver.reject_all("bulk reject from inbox")
            rejected = result.get("rejected", 0)
            failed = len(result.get("failed", []))
            if failed:
                self.notify(f"Rejected {rejected}; {failed} failed.", severity="warning")
            else:
                self.notify(f"Rejected {rejected} proposal(s).", severity="information")
            self._load_proposals()
        except Exception as e:
            self.notify(f"Reject all error: {e}", severity="error")
            logger.exception("InboxScreen.action_reject_all failed")

    # ------------------------------------------------------------------
    # Tab label update
    # ------------------------------------------------------------------

    def _update_tab_label(self, pending_count: int) -> None:
        """Update the Inbox tab label to show pending count."""
        try:
            tc = self.app.query_one("#main-tabs", TabbedContent)
            tab = tc.get_tab("inbox")
            if pending_count > 0:
                tab.label = f"◈ Inbox ({pending_count})"
            else:
                tab.label = "◈ Inbox"
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Event bus subscription (auto-refresh on adjustment.* events)
    # ------------------------------------------------------------------

    def _subscribe_events(self) -> None:
        """Start background listener for adjustment.* events from the trading EventBus."""
        self._watch_adjustment_events()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @work(thread=False)
    async def _watch_adjustment_events(self) -> None:
        """Async worker: subscribe to EventBus and reload proposals on adjustment.* events."""
        try:
            from src.scanner.automation.event_bus import get_event_bus
            bus = get_event_bus()
            async for event in bus.subscribe():
                if event.get("event_type", "").startswith("adjustment."):
                    self.call_later(self._load_proposals)
        except Exception:
            pass  # Event bus unavailable — 5s polling interval is the fallback

    def _focused_proposal(self) -> Optional[dict[str, Any]]:
        """Return the proposal dict for the currently highlighted DataTable row."""
        table = self.query_one("#inbox-table", DataTable)
        if table.row_count == 0:
            return None
        try:
            row_keys = list(table.rows.keys())
            cursor = table.cursor_row
            if cursor >= len(row_keys):
                return None
            rk = str(row_keys[cursor].value)
            return self._row_to_proposal.get(rk)
        except Exception:
            return None

    def _render_empty_state(self) -> None:
        """Called when no proposals exist — render empty message."""
        pass  # Handled by DataTable being empty; caller checks row_count

    def update_from_snapshot(self, snap: Any) -> None:
        """Called by app.py on every data refresh — re-checks proposal count."""
        self._load_proposals()

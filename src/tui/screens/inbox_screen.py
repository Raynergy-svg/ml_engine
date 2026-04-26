"""
INBOX SCREEN (F2) — Unified Inbox (Adjustments + Homework)

Morning operator dashboard. Two-pane layout:
- LEFT:  unified queue (config adjustment proposals + trade homework entries)
- RIGHT: live detail markdown that updates on arrow-key navigation

Filter pills cycle the queue between [All] [📚 Homework] [🔧 Adjustments].
A/R/E/S row actions dispatch by entry_type — homework uses HomeworkReviewer,
adjustments use the existing AdjustmentApprover path. Phase 96 Task 5.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

from rich.text import Text

from textual import work
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import (
    Button,
    DataTable,
    Input,
    Label,
    Markdown,
    Static,
    TabbedContent,
)

from src.scanner.automation.adjustment_approver import AdjustmentApprover
from src.scanner.automation.event_bus import get_event_bus
from src.scanner.automation.homework import HomeworkEntry, HomeworkStore
from src.scanner.automation.homework.reviewer import HomeworkReviewer
from src.scanner.config import ScannerConfig

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_PENDING_PATH = _PROJECT_ROOT / ".claude" / "pending_adjustments.json"
_APPROVED_PATH = _PROJECT_ROOT / ".claude" / "config_adjustments.json"

# Proposals in terminal states are not shown (approved already moved out, rejected is done)
_ACTIVE_STATUSES = ("pending", "snoozed", "invalid")


# ── Unified row adapter ──────────────────────────────────────────────


@dataclass
class UnifiedInboxRow:
    """Single row in the unified inbox.

    Wraps either a config-adjustment proposal dict or a HomeworkEntry. The
    `entry_type` discriminator lets the screen dispatch A/R/E/S to the
    appropriate backend (HomeworkReviewer vs AdjustmentApprover).
    """
    entry_type: str  # "homework" | "adjustment"
    timestamp: str
    subject: str
    detail_summary: str  # short summary for the queue column
    pl: Optional[float]  # for homework only — colors the row
    raw_entry: Any       # the actual HomeworkEntry or proposal dict


def _adjustment_to_row(p: dict) -> UnifiedInboxRow:
    return UnifiedInboxRow(
        entry_type="adjustment",
        timestamp=str(p.get("timestamp", ""))[:19],
        subject=str(p.get("key", "?")),
        detail_summary=f"{p.get('current_value', '?')} → {p.get('proposed_value', '?')}",
        pl=None,
        raw_entry=p,
    )


def _homework_to_row(e: HomeworkEntry) -> UnifiedInboxRow:
    return UnifiedInboxRow(
        entry_type="homework",
        timestamp=str(e.generated_at)[:19],
        subject=f"#{e.trade_id} {e.pair} {e.direction}",
        detail_summary=f"{e.close_reason}  {e.realized_pl:+.2f}",
        pl=e.realized_pl,
        raw_entry=e,
    )


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
            yield Input(placeholder="e.g. bounds too aggressive", id="reject-input", max_length=256)
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
            cfg = ScannerConfig()
        except (TypeError, ValueError) as e:
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


class HomeworkEditModal(ModalScreen[Optional[dict]]):
    """Capture an operator note while editing a homework entry's lesson.

    Returned dict shape: {"note": str, "lesson_override": str}. Returns
    None on cancel — the reviewer treats no-edit as a no-op.
    """

    BINDINGS = [Binding("escape", "cancel", "Cancel", show=True)]

    DEFAULT_CSS = """
    HomeworkEditModal {
        align: center middle;
    }

    #edit-dialog {
        width: 76;
        height: auto;
        background: #0a0a0f;
        border: double #00ff41;
        padding: 1 2;
    }

    #edit-title {
        text-align: center;
        color: #00ff41;
        text-style: bold;
        padding: 0 0 1 0;
    }

    #edit-prompt {
        color: #ffab00;
        padding: 0 0 0 0;
    }

    .edit-input {
        margin: 1 0 1 0;
        border: solid #00ff41;
        background: #131320;
    }

    #edit-buttons {
        align: center middle;
        height: 3;
    }

    #edit-confirm, #edit-cancel-btn {
        margin: 0 1;
    }
    """

    def __init__(self, homework_id: str, current_lesson: str = "") -> None:
        super().__init__()
        self._homework_id = homework_id
        self._current_lesson = current_lesson

    def compose(self) -> ComposeResult:
        with Vertical(id="edit-dialog"):
            yield Label(f"✎ Edit Homework {self._homework_id[:16]}…", id="edit-title")
            yield Label("Operator note (required):", id="edit-prompt")
            yield Input(placeholder="why this lesson is wrong/right…", id="edit-note", classes="edit-input")
            yield Label("Lesson override (optional):", id="edit-prompt")
            yield Input(
                placeholder=self._current_lesson[:60] or "(leave empty to keep proposed)",
                id="edit-lesson",
                classes="edit-input",
            )
            with Horizontal(id="edit-buttons"):
                yield Button("✓ Save", id="edit-confirm", variant="success")
                yield Button("← Cancel", id="edit-cancel-btn", variant="default")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "edit-confirm":
            note = self.query_one("#edit-note", Input).value or ""
            lesson = self.query_one("#edit-lesson", Input).value or ""
            if not note.strip():
                # operator note is required for an edit transition
                return
            self.dismiss({"note": note, "lesson_override": lesson})
        elif event.button.id == "edit-cancel-btn":
            self.dismiss(None)

    def action_cancel(self) -> None:
        self.dismiss(None)


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
        Binding("e", "edit", "Edit", show=True),
        Binding("s", "snooze", "Snooze 24h", show=True),
        Binding("v", "view_detail", "View Detail", show=True),
        Binding("tab", "cycle_filter", "Filter", show=True),
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

    #filter-bar {
        height: 3;
        align: left middle;
        padding: 0 1;
    }

    .filter-pill {
        margin: 0 1;
        background: #1a1a2a;
        color: #6666aa;
        border: solid #2a2a4a;
    }

    .filter-pill.active {
        background: #2a2a4a;
        color: #ff00ff;
        text-style: bold;
        border: solid #ff00ff;
    }

    #two-pane-row {
        height: 1fr;
    }

    #inbox-queue {
        width: 1fr;
        height: 1fr;
        border: solid #2a2a4a;
        background: #131320;
    }

    #detail-pane {
        width: 1fr;
        height: 1fr;
        border: solid #7c4dff;
        background: #0a0a0f;
        padding: 1 2;
    }

    #detail-pane-md {
        color: #e0e0ff;
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
    """

    def __init__(self, project_root: str = "", **kwargs) -> None:
        super().__init__(**kwargs)
        self._project_root = Path(project_root) if project_root else _PROJECT_ROOT
        self._pending_path = self._project_root / ".claude" / "pending_adjustments.json"
        self._approved_path = self._project_root / ".claude" / "config_adjustments.json"
        # Proposal list (all actionable statuses) — kept for PreviewModal/bulk actions
        self._proposals: list[dict[str, Any]] = []
        # Unified queue: adjustments + homework, currently visible (post-filter)
        self._current_rows: list[UnifiedInboxRow] = []
        # Map from DataTable row_key → row
        self._row_to_row: dict[str, UnifiedInboxRow] = {}
        # Active filter: "all" | "homework" | "adjustments"
        self._current_filter: str = "all"
        # Filter pill IDs in cycle order (for Tab key)
        self._filter_cycle = ["all", "homework", "adjustments"]
        # mtime-keyed read caches — fine <100 entries today, prevents pathological
        # re-parse at 10k+. Key: stat result tuple. Value: parsed payload.
        self._proposals_cache: tuple[int, list[dict[str, Any]]] | None = None
        self._homework_cache: tuple[tuple[int, int], list[Any]] | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def compose(self) -> ComposeResult:
        with Horizontal(id="inbox-header"):
            yield Label("◈ UNIFIED INBOX", id="inbox-title")
            yield Label(
                "A approve  R reject  E edit  S snooze 24h  V detail  Tab filter",
                id="inbox-subtitle",
            )

        with Horizontal(id="filter-bar"):
            yield Button("All", id="filter-all", classes="filter-pill active")
            yield Button("📚 Homework", id="filter-homework", classes="filter-pill")
            yield Button("🔧 Adjustments", id="filter-adjustments", classes="filter-pill")

        with Horizontal(id="two-pane-row"):
            yield DataTable(id="inbox-queue", cursor_type="row")
            with VerticalScroll(id="detail-pane"):
                yield Markdown(
                    "_Select an entry to view detail…_",
                    id="detail-pane-md",
                )

        yield Static(
            "No pending entries. Buddy is running on current config.",
            id="inbox-empty",
        )

        with Horizontal(id="inbox-actions"):
            yield Button("◈ Preview net config", id="preview-btn", variant="default",
                         classes="inbox-action-btn")
            yield Button("✓ Approve all", id="approve-all-btn", variant="success",
                         classes="inbox-action-btn")
            yield Button("✕ Reject all", id="reject-all-btn", variant="error",
                         classes="inbox-action-btn")

    def on_mount(self) -> None:
        table = self.query_one("#inbox-queue", DataTable)
        table.add_columns("Type", "When", "Subject", "Detail")
        self._load_proposals()
        self.set_interval(5.0, self._load_proposals)
        self._subscribe_events()

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def _load_proposals(self) -> None:
        """Read both adjustment proposals + homework entries, refresh the unified queue."""
        proposals = self._read_proposals()
        self._proposals = proposals
        rows = self._load_unified_rows()
        self._current_rows = rows
        self._rebuild_table(rows)
        self._update_tab_label(self._count_actionable(proposals, rows))

    @staticmethod
    def _stat_mtime_ns(path: Path) -> int:
        """Return mtime in nanoseconds, or 0 if path is missing/unreadable."""
        try:
            return path.stat().st_mtime_ns
        except OSError:
            return 0

    def _read_proposals(self) -> list[dict[str, Any]]:
        """Load actionable proposals from pending_adjustments.json (mtime-cached)."""
        if not self._pending_path.exists():
            self._proposals_cache = None
            return []
        mtime = self._stat_mtime_ns(self._pending_path)
        if self._proposals_cache is not None and self._proposals_cache[0] == mtime:
            return self._proposals_cache[1]
        try:
            data = json.loads(self._pending_path.read_text())
            all_proposals = data.get("proposals", [])
            result = [p for p in all_proposals if p.get("status") in _ACTIVE_STATUSES]
        except (OSError, json.JSONDecodeError) as e:
            logger.warning("InboxScreen: failed to load proposals: %s", e)
            return []
        self._proposals_cache = (mtime, result)
        return result

    def _load_unified_rows(self) -> list[UnifiedInboxRow]:
        """Read homework + adjustments, merge sorted by timestamp desc.

        Homework reads are mtime-cached against (pending_mtime, history_mtime)
        — both files participate because list_pending() dedupes against
        history. Either changing invalidates the cache.
        """
        rows: list[UnifiedInboxRow] = []

        # Homework — pending only, skip un-expired snoozed entries.
        # Key the cache on both file mtimes since list_pending dedupes on history.
        try:
            store = HomeworkStore()
            cache_key = (
                self._stat_mtime_ns(store.pending_path),
                self._stat_mtime_ns(store.history_path),
            )
            if self._homework_cache is not None and self._homework_cache[0] == cache_key:
                hw_entries = self._homework_cache[1]
            else:
                hw_entries = store.list_pending()
                self._homework_cache = (cache_key, hw_entries)
            for e in hw_entries:
                status = e.status or "pending"
                if status.startswith("snoozed_until_") and not self._snooze_expired(status):
                    continue
                rows.append(_homework_to_row(e))
        except (OSError, ValueError) as ex:
            logger.debug("InboxScreen homework read error: %s", ex)

        # Adjustments — reuse already-loaded list
        for p in self._proposals:
            rows.append(_adjustment_to_row(p))

        # Apply current filter
        if self._current_filter == "homework":
            rows = [r for r in rows if r.entry_type == "homework"]
        elif self._current_filter == "adjustments":
            rows = [r for r in rows if r.entry_type == "adjustment"]

        rows.sort(key=lambda r: r.timestamp, reverse=True)
        return rows

    @staticmethod
    def _snooze_expired(status: str) -> bool:
        """Return True if a 'snoozed_until_X' status's timestamp is in the past."""
        try:
            ts = status.replace("snoozed_until_", "")
            until = datetime.fromisoformat(ts)
            if until.tzinfo is None:
                until = until.replace(tzinfo=timezone.utc)
            return datetime.now(timezone.utc) >= until
        except (ValueError, TypeError):
            return False

    @staticmethod
    def _count_actionable(proposals: list[dict[str, Any]], rows: list[UnifiedInboxRow]) -> int:
        """Tab badge count = pending adjustments + visible homework entries."""
        adj_pending = len([p for p in proposals if p.get("status") == "pending"])
        hw_visible = len([r for r in rows if r.entry_type == "homework"])
        return adj_pending + hw_visible

    def _rebuild_table(self, rows: list[UnifiedInboxRow]) -> None:
        """Repopulate the DataTable from the unified row list."""
        table = self.query_one("#inbox-queue", DataTable)
        table.clear()
        self._row_to_row = {}

        try:
            empty_label = self.query_one("#inbox-empty", Static)
            empty_label.display = not bool(rows)
            table.display = bool(rows)
        except Exception:
            pass

        if not rows:
            self._render_detail(None)
            return

        for idx, r in enumerate(rows):
            type_glyph = "📚 HW" if r.entry_type == "homework" else "🔧 ADJ"
            ts = r.timestamp.replace("T", " ")[:16]
            subj = r.subject[:36]
            detail = r.detail_summary[:40]

            # Color homework rows by P/L sign
            if r.entry_type == "homework" and r.pl is not None:
                color = "#00ff41" if r.pl >= 0 else "#ff1744"
                detail_cell: Any = Text(detail, style=color)
            else:
                detail_cell = detail

            row_key = f"row-{idx}-{r.entry_type}"
            table.add_row(type_glyph, ts, subj, detail_cell, key=row_key)
            self._row_to_row[row_key] = r

        # Render detail for first row
        self._render_detail(rows[0])

    def _get_current_config_values(self) -> dict[str, Any]:
        """Return {field_name: current_value} from a fresh ScannerConfig instance."""
        import dataclasses
        try:
            cfg = ScannerConfig()
            return {f.name: getattr(cfg, f.name) for f in dataclasses.fields(cfg)}
        except (TypeError, ValueError) as exc:
            logger.warning("InboxScreen._get_current_config_values failed: %s", exc)
            return {}

    # ------------------------------------------------------------------
    # Detail pane rendering
    # ------------------------------------------------------------------

    def on_data_table_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        """When user navigates with arrow keys, update the detail pane."""
        try:
            cursor = event.cursor_row
        except AttributeError:
            cursor = getattr(event, "cursor_row", -1)
        if 0 <= cursor < len(self._current_rows):
            self._render_detail(self._current_rows[cursor])

    def _render_detail(self, row: Optional[UnifiedInboxRow]) -> None:
        """Update the right-pane Markdown widget for the focused row."""
        try:
            pane = self.query_one("#detail-pane-md", Markdown)
        except Exception:
            return

        if row is None:
            pane.update("_Inbox is empty._")
            return

        if row.entry_type == "homework":
            entry: HomeworkEntry = row.raw_entry
            md = entry.analysis_markdown or self._fallback_homework_md(entry)
            pane.update(md)
        else:
            p = row.raw_entry
            val = (p.get("validation") or {})
            valid_glyph = "✓" if val.get("valid") else "✗"
            md = (
                "## 🔧 Configuration Adjustment\n\n"
                f"**Field:** `{p.get('key', '?')}`  \n"
                f"**Current:** `{p.get('current_value', '—')}`  \n"
                f"**Proposed:** `{p.get('proposed_value', '?')}`  \n"
                f"**Source:** {p.get('source', 'unknown')}  \n"
                f"**Status:** {p.get('status', '?')}  \n"
                f"**Validation:** {valid_glyph} {val.get('error_message', '')}\n\n"
                "### Reason\n\n"
                f"{p.get('reason', '_(no reason)_')}\n\n"
                "---\n\n"
                "**Hotkeys:** `A` approve · `R` reject · `S` snooze · `V` detail modal\n"
            )
            pane.update(md)

    @staticmethod
    def _fallback_homework_md(entry: HomeworkEntry) -> str:
        """Used when analysis_markdown is empty — render a compact summary."""
        return (
            f"## 📚 Homework — Trade #{entry.trade_id} {entry.pair} {entry.direction}\n\n"
            f"**Outcome:** {entry.close_reason}  ({entry.realized_pl:+.2f})  \n"
            f"**Regime:** {entry.regime} | **R:R:** {entry.rr_ratio:.2f} | "
            f"**Conf:** {entry.confidence:.2f}\n\n"
            "### Proposed Lesson\n\n"
            f"{entry.proposed_lesson or '_(none)_'}\n\n"
            "**Hotkeys:** `A` approve · `R` reject · `E` edit · `S` snooze\n"
        )

    # ------------------------------------------------------------------
    # Actions (BINDINGS) — dispatch by entry_type
    # ------------------------------------------------------------------

    def action_approve(self) -> None:
        """A — approve the focused row (homework or adjustment)."""
        row = self._focused_row()
        if row is None:
            self.notify("No entry selected.", severity="warning")
            return
        if row.entry_type == "homework":
            self._approve_homework(row.raw_entry)
        else:
            self._approve_adjustment(row.raw_entry)

    def action_reject(self) -> None:
        """R — reject the focused row (modal captures reason)."""
        row = self._focused_row()
        if row is None:
            self.notify("No entry selected.", severity="warning")
            return
        if row.entry_type == "homework":
            entry: HomeworkEntry = row.raw_entry
            label = f"#{entry.trade_id} {entry.pair}"

            def _on_homework_reject(reason: Optional[str]) -> None:
                if not reason:
                    return
                self._reject_homework(entry, reason)

            self.app.push_screen(RejectModal(label), callback=_on_homework_reject)
        else:
            proposal = row.raw_entry
            key = proposal.get("key", "?")
            self.app.push_screen(RejectModal(key), callback=self._on_reject_result)

    def action_edit(self) -> None:
        """E — edit operator note + lesson override (homework only)."""
        row = self._focused_row()
        if row is None:
            self.notify("No entry selected.", severity="warning")
            return
        if row.entry_type != "homework":
            self.notify("Edit only applies to homework entries.", severity="warning")
            return

        entry: HomeworkEntry = row.raw_entry

        def _on_edit_result(payload: Optional[dict]) -> None:
            if not payload:
                return
            self._apply_homework_edit(entry, payload)

        self.app.push_screen(
            HomeworkEditModal(entry.homework_id, entry.proposed_lesson),
            callback=_on_edit_result,
        )

    def action_snooze(self) -> None:
        """S — snooze the focused row for 24 hours."""
        row = self._focused_row()
        if row is None:
            self.notify("No entry selected.", severity="warning")
            return
        if row.entry_type == "homework":
            self._snooze_homework(row.raw_entry)
        else:
            self._snooze_adjustment(row.raw_entry)

    def action_view_detail(self) -> None:
        """V — show full proposal detail in a modal (adjustments only — homework uses pane)."""
        row = self._focused_row()
        if row is None:
            self.notify("No entry selected.", severity="warning")
            return
        if row.entry_type == "homework":
            # Detail pane already shows homework markdown; no separate modal needed
            self.notify("Homework detail is in the right pane.", severity="information")
            return
        self.app.push_screen(DetailModal(row.raw_entry))

    def action_cycle_filter(self) -> None:
        """Tab — cycle filter pills All → Homework → Adjustments → All."""
        try:
            idx = self._filter_cycle.index(self._current_filter)
        except ValueError:
            idx = 0
        self._set_filter(self._filter_cycle[(idx + 1) % len(self._filter_cycle)])

    def _set_filter(self, name: str) -> None:
        """Switch active filter, refresh pill styling, reload rows."""
        if name not in self._filter_cycle:
            return
        self._current_filter = name
        for fid in self._filter_cycle:
            try:
                btn = self.query_one(f"#filter-{fid}", Button)
                if fid == name:
                    btn.add_class("active")
                else:
                    btn.remove_class("active")
            except Exception:
                pass
        self._load_proposals()

    # ------------------------------------------------------------------
    # Adjustment-side helpers
    # ------------------------------------------------------------------

    def _approve_adjustment(self, proposal: dict[str, Any]) -> None:
        val_dict = proposal.get("validation") or {}
        if not val_dict.get("valid"):
            self.notify(
                f"Cannot approve: {val_dict.get('error_message', 'invalid proposal')}",
                severity="error",
            )
            return
        if proposal.get("status") not in ("pending", "snoozed"):
            self.notify(
                f"Proposal is {proposal.get('status')} — cannot approve.",
                severity="warning",
            )
            return
        proposal_id = proposal.get("id", "")
        try:
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
            logger.exception("InboxScreen._approve_adjustment failed")

    def _on_reject_result(self, reason: Optional[str]) -> None:
        if reason is None:
            return
        row = self._focused_row()
        if row is None or row.entry_type != "adjustment":
            return
        proposal = row.raw_entry
        proposal_id = proposal.get("id", "")
        try:
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
            logger.exception("InboxScreen._on_reject_result failed")

    def _snooze_adjustment(self, proposal: dict[str, Any]) -> None:
        proposal_id = proposal.get("id", "")
        try:
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
            logger.exception("InboxScreen._snooze_adjustment failed")

    # ------------------------------------------------------------------
    # Homework-side helpers
    # ------------------------------------------------------------------

    def _approve_homework(self, entry: HomeworkEntry) -> None:
        try:
            signal = HomeworkReviewer(store=HomeworkStore()).approve(entry.homework_id)
            if signal:
                self.notify(
                    f"✓ Approved homework #{entry.trade_id} — {len(signal.agent_weight_deltas)} agent deltas",
                    severity="information",
                )
                self._load_proposals()
            else:
                self.notify("Approve failed — entry not found in pending.", severity="error")
        except Exception as e:
            self.notify(f"Homework approve error: {e}", severity="error")
            logger.exception("InboxScreen._approve_homework failed")

    def _reject_homework(self, entry: HomeworkEntry, note: str) -> None:
        try:
            signal = HomeworkReviewer(store=HomeworkStore()).reject(entry.homework_id, note=note)
            if signal:
                self.notify(f"✗ Rejected homework #{entry.trade_id}", severity="information")
                self._load_proposals()
            else:
                self.notify("Reject failed — entry not found.", severity="error")
        except Exception as e:
            self.notify(f"Homework reject error: {e}", severity="error")
            logger.exception("InboxScreen._reject_homework failed")

    def _apply_homework_edit(self, entry: HomeworkEntry, payload: dict) -> None:
        try:
            edits: dict[str, Any] = {"note": payload.get("note", "")}
            lesson = (payload.get("lesson_override") or "").strip()
            if lesson:
                edits["lesson_override"] = lesson
            signal = HomeworkReviewer(store=HomeworkStore()).edit(
                entry.homework_id, edits=edits
            )
            if signal:
                self.notify(f"✎ Edited homework #{entry.trade_id}", severity="information")
                self._load_proposals()
            else:
                self.notify("Edit failed — entry not found.", severity="error")
        except Exception as e:
            self.notify(f"Homework edit error: {e}", severity="error")
            logger.exception("InboxScreen._apply_homework_edit failed")

    def _snooze_homework(self, entry: HomeworkEntry) -> None:
        try:
            ok = HomeworkReviewer(store=HomeworkStore()).snooze(entry.homework_id, hours=24)
            if ok:
                self.notify(f"⏸ Snoozed homework #{entry.trade_id} 24h", severity="information")
                self._load_proposals()
            else:
                self.notify("Snooze failed — entry not found.", severity="error")
        except Exception as e:
            self.notify(f"Homework snooze error: {e}", severity="error")
            logger.exception("InboxScreen._snooze_homework failed")

    # ------------------------------------------------------------------
    # Buttons
    # ------------------------------------------------------------------

    def on_button_pressed(self, event: Button.Pressed) -> None:
        bid = event.button.id
        if bid == "preview-btn":
            self.app.push_screen(PreviewModal(self._proposals))
        elif bid == "approve-all-btn":
            self.action_approve_all()
        elif bid == "reject-all-btn":
            self.action_reject_all()
        elif bid in ("filter-all", "filter-homework", "filter-adjustments"):
            self._set_filter(bid.replace("filter-", ""))

    def action_approve_all(self) -> None:
        """Approve every pending homework + every valid pending/snoozed adjustment in the queue."""
        hw_approved = 0
        hw_failed = 0
        adj_approved = 0
        adj_skipped = 0
        adj_failed = 0

        # Homework: iterate visible rows in current filtered queue
        try:
            reviewer = HomeworkReviewer(store=HomeworkStore())
            hw_rows = [r for r in self._current_rows if r.entry_type == "homework"]
            for row in hw_rows:
                try:
                    entry = row.raw_entry
                    signal = reviewer.approve(entry.homework_id)
                    if signal is not None:
                        hw_approved += 1
                    else:
                        hw_failed += 1
                except Exception:
                    hw_failed += 1
                    logger.exception("InboxScreen.action_approve_all: homework approve failed")
        except Exception as e:
            logger.exception("InboxScreen.action_approve_all: homework loop init failed")
            self.notify(f"Homework approve init error: {e}", severity="error")

        # Adjustments: keep existing AdjustmentApprover.approve_all() behavior
        try:
            approver = AdjustmentApprover(
                pending_path=self._pending_path,
                approved_path=self._approved_path,
            )
            result = approver.approve_all()
            adj_approved = result.get("approved", 0)
            adj_skipped = result.get("skipped", 0)
            adj_failed = len(result.get("failed", []))
        except Exception as e:
            logger.exception("InboxScreen.action_approve_all: adjustment approve failed")
            self.notify(f"Adjustment approve error: {e}", severity="error")

        total_failed = hw_failed + adj_failed
        msg = (
            f"Approved {hw_approved} homework + {adj_approved} adjustments; "
            f"{total_failed} failed; {adj_skipped} invalid skipped."
        )
        self.notify(msg, severity="warning" if total_failed else "information")
        self._load_proposals()

    def action_reject_all(self) -> None:
        """Reject every pending homework + every actionable adjustment in the queue."""
        hw_rejected = 0
        hw_failed = 0
        adj_rejected = 0
        adj_failed = 0
        bulk_note = "bulk reject from inbox"

        # Homework: iterate visible rows; reject() requires a non-empty note
        try:
            reviewer = HomeworkReviewer(store=HomeworkStore())
            hw_rows = [r for r in self._current_rows if r.entry_type == "homework"]
            for row in hw_rows:
                try:
                    entry = row.raw_entry
                    signal = reviewer.reject(entry.homework_id, note=bulk_note)
                    if signal is not None:
                        hw_rejected += 1
                    else:
                        hw_failed += 1
                except Exception:
                    hw_failed += 1
                    logger.exception("InboxScreen.action_reject_all: homework reject failed")
        except Exception as e:
            logger.exception("InboxScreen.action_reject_all: homework loop init failed")
            self.notify(f"Homework reject init error: {e}", severity="error")

        # Adjustments: keep existing AdjustmentApprover.reject_all() behavior
        try:
            approver = AdjustmentApprover(
                pending_path=self._pending_path,
                approved_path=self._approved_path,
            )
            result = approver.reject_all(bulk_note)
            adj_rejected = result.get("rejected", 0)
            adj_failed = len(result.get("failed", []))
        except Exception as e:
            logger.exception("InboxScreen.action_reject_all: adjustment reject failed")
            self.notify(f"Adjustment reject error: {e}", severity="error")

        total_failed = hw_failed + adj_failed
        msg = (
            f"Rejected {hw_rejected} homework + {adj_rejected} adjustments; "
            f"{total_failed} failed."
        )
        self.notify(msg, severity="warning" if total_failed else "information")
        self._load_proposals()

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
            bus = get_event_bus()
            async for event in bus.subscribe():
                if event.get("event_type", "").startswith("adjustment."):
                    self.call_later(self._load_proposals)
        except Exception:
            pass  # Event bus unavailable — 5s polling interval is the fallback

    def _focused_row(self) -> Optional[UnifiedInboxRow]:
        """Return the UnifiedInboxRow for the currently highlighted DataTable row."""
        try:
            table = self.query_one("#inbox-queue", DataTable)
        except Exception:
            return None
        if table.row_count == 0:
            return None
        try:
            cursor = table.cursor_row
            if 0 <= cursor < len(self._current_rows):
                return self._current_rows[cursor]
            # Fallback: row_keys lookup if cursor is out of sync
            row_keys = list(table.rows.keys())
            if cursor >= len(row_keys):
                return None
            rk = str(row_keys[cursor].value)
            return self._row_to_row.get(rk)
        except Exception:
            return None

    def update_from_snapshot(self, snap: Any) -> None:
        """Called by app.py on every data refresh — re-checks proposal count."""
        self._load_proposals()

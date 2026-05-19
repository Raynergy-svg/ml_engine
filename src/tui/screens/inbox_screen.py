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
from textual.css.query import NoMatches
from textual.screen import ModalScreen
from textual.widgets import (
    Button,
    DataTable,
    Input,
    Label,
    Markdown,
    ProgressBar,
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

# Mythos audit 2026-04-29 — Meta-pipeline ChangePackages live here.
# The orchestrator routes per-cycle diagnostics to MetaManager when
# enable_meta_manager=True (commit f070d39); each routed incident
# produces a JSON file in this directory. Surfacing them in the inbox
# is step B of the Tier 7 visibility plan.
_META_CHANGES_DIR = _PROJECT_ROOT / ".claude" / "meta" / "changes"

# Proposals in terminal states are not shown (approved already moved out, rejected is done)
_ACTIVE_STATUSES = ("pending", "snoozed", "invalid")

# ChangePackage stages that count as "in the operator's queue" — terminal
# stages (closed/aborted/rejected) are filtered out so the inbox doesn't
# pile up forever.
_META_ACTIVE_STAGES = (
    "intake", "diagnosing", "proposing", "evaluating",
    "policy_check", "awaiting_approval", "approved",
    "deployed_shadow", "deployed_canary", "deployed_live",
    "post_deploy_review",
)


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


def _meta_package_to_row(blob: dict) -> Optional[UnifiedInboxRow]:
    """Convert a meta-pipeline ChangePackage JSON blob to an inbox row.

    Returns None for malformed packages. Subject = change_id + kind.
    Detail summary = stage + delta-key count + constitution status.
    """
    try:
        change_id = str(blob.get("change_id", ""))
        kind = str(blob.get("kind", "?"))
        stage = str(blob.get("stage", "?"))
    except Exception:
        return None
    if not change_id:
        return None

    proposal = blob.get("proposal") or {}
    delta = (proposal.get("config_delta") or {}) if isinstance(proposal, dict) else {}
    delta_keys = list(delta.keys()) if isinstance(delta, dict) else []

    attestation = blob.get("constitution_attestation") or {}
    if isinstance(attestation, dict):
        const_passed = attestation.get("passed")
    else:
        const_passed = None
    const_str = (
        "✓" if const_passed is True
        else ("✗" if const_passed is False else "?")
    )

    history = blob.get("history") or []
    last_ts = ""
    if isinstance(history, list) and history:
        last = history[-1]
        if isinstance(last, dict):
            last_ts = str(last.get("ts", ""))[:19]
    if not last_ts:
        last_ts = str(blob.get("created_at", ""))[:19]

    detail = f"{stage}  Δkeys={len(delta_keys)}  const={const_str}"
    if delta_keys:
        # Show first key + count rather than dumping all keys
        detail = f"{stage}  {delta_keys[0]}{'…' if len(delta_keys) > 1 else ''}  const={const_str}"

    return UnifiedInboxRow(
        entry_type="meta",
        timestamp=last_ts,
        subject=f"⟪{change_id[:8]}⟫ {kind}",
        detail_summary=detail,
        pl=None,
        raw_entry=blob,
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
        background: #05060a;
        border: double #ff3158;
        padding: 1 2;
    }

    #reject-title {
        text-align: center;
        color: #ff3158;
        text-style: bold;
        padding: 0 0 1 0;
    }

    #reject-prompt {
        color: #ffb000;
        padding: 0 0 0 0;
    }

    #reject-input {
        margin: 1 0 1 0;
        border: solid #ff3158;
        background: #090d18;
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

    #reject-validation {
        color: #ff3158;
        padding: 0 0 0 0;
        height: 1;
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
            yield Label("", id="reject-validation")
            with Horizontal(id="reject-buttons"):
                yield Button("✕ Reject", id="reject-confirm", variant="error", disabled=True)
                yield Button("← Back", id="reject-cancel-btn", variant="default")

    def on_input_changed(self, event: Input.Changed) -> None:
        btn = self.query_one("#reject-confirm", Button)
        btn.disabled = not event.value.strip()
        if event.value.strip():
            self.query_one("#reject-validation", Label).update("")

    def on_input_submitted(self, event: Input.Submitted) -> None:
        self._try_submit()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "reject-confirm":
            self._try_submit()
        elif event.button.id == "reject-cancel-btn":
            self.dismiss(None)

    def _try_submit(self) -> None:
        note = self.query_one("#reject-input", Input).value.strip()
        if not note:
            self.query_one("#reject-validation", Label).update("note required")
            return
        self.dismiss(note)

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
        background: #05060a;
        border: double #b84dff;
        padding: 1 2;
        overflow-y: auto;
    }

    #detail-title {
        text-align: center;
        color: #b84dff;
        text-style: bold;
        padding: 0 0 1 0;
    }

    #detail-body {
        color: #e8f7ff;
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
        background: #05060a;
        border: double #39ff14;
        padding: 1 2;
        overflow-y: auto;
    }

    #preview-title {
        text-align: center;
        color: #39ff14;
        text-style: bold;
        padding: 0 0 1 0;
    }

    #preview-body {
        color: #e8f7ff;
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
        background: #05060a;
        border: double #39ff14;
        padding: 1 2;
    }

    #edit-title {
        text-align: center;
        color: #39ff14;
        text-style: bold;
        padding: 0 0 1 0;
    }

    #edit-prompt {
        color: #ffb000;
        padding: 0 0 0 0;
    }

    .edit-input {
        margin: 1 0 1 0;
        border: solid #39ff14;
        background: #090d18;
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
        background: #05060a;
    }

    #inbox-header {
        height: 3;
        align: left middle;
        background: #090b14;
        border-bottom: solid #26304f;
    }

    #inbox-title {
        color: #ff2bd6;
        text-style: bold;
        padding: 0 1;
    }

    #inbox-subtitle {
        color: #7483b8;
        padding: 0 2;
    }

    #filter-bar {
        height: 3;
        align: left middle;
        padding: 0 1;
    }

    .filter-pill {
        margin: 0 1;
        background: #111827;
        color: #7483b8;
        border: solid #26304f;
    }

    .filter-pill.active {
        background: #1f2a44;
        color: #ffea00;
        text-style: bold;
        border: tall #ff2bd6;
    }

    .filter-pill:hover {
        color: #00f5ff;
        border: solid #00f5ff;
    }

    #two-pane-row {
        height: 1fr;
    }

    #inbox-queue {
        width: 1fr;
        height: 1fr;
        border: double #00f5ff;
        background: #090d18;
        color: #e8f7ff;
    }

    #detail-pane {
        width: 1fr;
        height: 1fr;
        border: double #b84dff;
        background: #070912;
        padding: 1 2;
    }

    #detail-pane-md {
        color: #e8f7ff;
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
        color: #7483b8;
        text-align: center;
        display: none;
        border: solid #26304f;
        background: #090d18;
    }

    /* US-013 — approve-all worker progress indicator.
       Hidden by default; the worker toggles display via set_styles
       through call_from_thread when approve-all starts/finishes. */
    #approve-all-progress-row {
        height: 3;
        align: center middle;
        padding: 0 1;
        background: #0a0e1a;
        border-bottom: solid #26304f;
        display: none;
    }

    #approve-all-progress-label {
        color: #00f5ff;
        padding: 0 2;
    }

    #approve-all-progress {
        width: 1fr;
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
        self._filter_cycle = ["all", "homework", "adjustments", "meta"]
        # mtime-keyed read caches — fine <100 entries today, prevents pathological
        # re-parse at 10k+. Key: stat result tuple. Value: parsed payload.
        self._proposals_cache: tuple[int, list[dict[str, Any]]] | None = None
        self._homework_cache: tuple[tuple[int, int], list[Any]] | None = None
        # Cache for _read_meta_packages: (dir_mtime_ns, rows)
        self._meta_cache: tuple[int, list[UnifiedInboxRow]] | None = None
        # US-013 — re-entrancy guard for approve-all worker. Prevents a
        # second click/keypress from spawning a parallel worker while the
        # first is still running.
        self._approve_all_running: bool = False
        # Latest auto-hide timer for the completion banner (cancelled on
        # any new approve-all to keep the banner state coherent).
        self._approve_all_hide_timer: Any = None

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

        # US-013 — approve-all worker progress indicator. Hidden until
        # the worker fires; the worker uses call_from_thread to flip
        # display + update progress, keeping the main thread responsive.
        with Horizontal(id="approve-all-progress-row"):
            yield Label("Approve all …", id="approve-all-progress-label")
            yield ProgressBar(
                total=100,
                show_eta=False,
                show_percentage=True,
                id="approve-all-progress",
            )

        with Horizontal(id="filter-bar"):
            yield Button("All", id="filter-all", classes="filter-pill active")
            yield Button("📚 Homework", id="filter-homework", classes="filter-pill")
            yield Button("🔧 Adjustments", id="filter-adjustments", classes="filter-pill")
            yield Button("🧠 Meta", id="filter-meta", classes="filter-pill")

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

        # Meta-pipeline ChangePackages — Mythos 2026-04-29 step B.
        # Each non-terminal package in .claude/meta/changes/ surfaces
        # as an inbox row so the operator can SEE what the cybernetic
        # loop produced this scan, not just what the legacy adjuster
        # queued.
        try:
            for meta_row in self._read_meta_packages():
                rows.append(meta_row)
        except (OSError, ValueError) as ex:
            logger.debug("InboxScreen meta read error: %s", ex)

        # Apply current filter
        if self._current_filter == "homework":
            rows = [r for r in rows if r.entry_type == "homework"]
        elif self._current_filter == "adjustments":
            rows = [r for r in rows if r.entry_type == "adjustment"]
        elif self._current_filter == "meta":
            rows = [r for r in rows if r.entry_type == "meta"]

        rows.sort(key=lambda r: r.timestamp, reverse=True)
        return rows

    def _read_meta_packages(self) -> list[UnifiedInboxRow]:
        """Scan .claude/meta/changes/*.json and emit non-terminal rows.

        Cached on the directory's mtime so we don't re-parse every
        5-second interval when nothing changed.
        """
        if not _META_CHANGES_DIR.exists():
            self._meta_cache = None
            return []
        try:
            dir_mtime = _META_CHANGES_DIR.stat().st_mtime_ns
        except OSError:
            return []
        if self._meta_cache is not None and self._meta_cache[0] == dir_mtime:
            return list(self._meta_cache[1])

        rows: list[UnifiedInboxRow] = []
        for path in _META_CHANGES_DIR.glob("*.json"):
            if path.name.endswith(".lock"):
                continue
            try:
                blob = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(blob, dict):
                continue
            stage = str(blob.get("stage", ""))
            if stage not in _META_ACTIVE_STAGES:
                continue
            row = _meta_package_to_row(blob)
            if row is not None:
                rows.append(row)
        self._meta_cache = (dir_mtime, rows)
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
        meta_visible = len([r for r in rows if r.entry_type == "meta"])
        return adj_pending + hw_visible + meta_visible

    def _rebuild_table(self, rows: list[UnifiedInboxRow]) -> None:
        """Repopulate the DataTable from the unified row list."""
        table = self.query_one("#inbox-queue", DataTable)
        table.clear()
        self._row_to_row = {}

        try:
            empty_label = self.query_one("#inbox-empty", Static)
            empty_label.display = not bool(rows)
            table.display = bool(rows)
        except NoMatches:
            logger.debug("InboxScreen._rebuild_table: #inbox-empty not mounted yet")

        if not rows:
            self._render_detail(None)
            return

        for idx, r in enumerate(rows):
            if r.entry_type == "homework":
                type_glyph = "📚 HW"
            elif r.entry_type == "meta":
                type_glyph = "🧠 META"
            else:
                type_glyph = "🔧 ADJ"
            ts = r.timestamp.replace("T", " ")[:16]
            subj = r.subject[:36]
            detail = r.detail_summary[:40]

            # Color homework rows by P/L sign
            if r.entry_type == "homework" and r.pl is not None:
                color = "#39ff14" if r.pl >= 0 else "#ff3158"
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
        except NoMatches:
            logger.debug("InboxScreen._render_detail: #detail-pane-md not mounted")
            return

        if row is None:
            pane.update("_Inbox is empty._")
            return

        if row.entry_type == "homework":
            entry: HomeworkEntry = row.raw_entry
            md = entry.analysis_markdown or self._fallback_homework_md(entry)
            pane.update(md)
        elif row.entry_type == "meta":
            pane.update(self._meta_detail_md(row.raw_entry))
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
    def _meta_detail_md(blob: dict) -> str:
        """Render a ChangePackage blob as readable Markdown for the detail pane."""
        change_id = str(blob.get("change_id", "?"))
        kind = str(blob.get("kind", "?"))
        stage = str(blob.get("stage", "?"))
        source = str(blob.get("source", "?"))
        created = str(blob.get("created_at", ""))[:19].replace("T", " ")

        proposal = blob.get("proposal") or {}
        delta = (proposal.get("config_delta") or {}) if isinstance(proposal, dict) else {}
        rationale = (proposal.get("rationale") or "") if isinstance(proposal, dict) else ""

        attestation = blob.get("constitution_attestation") or {}
        if isinstance(attestation, dict):
            const_passed = attestation.get("passed")
            const_notes = str(attestation.get("notes", ""))[:120]
        else:
            const_passed = None
            const_notes = ""
        const_str = "✓ passed" if const_passed is True else ("✗ failed" if const_passed is False else "? unknown")

        history = blob.get("history") or []
        history_lines = []
        if isinstance(history, list):
            for h in history[-5:]:
                if isinstance(h, dict):
                    ts = str(h.get("ts", ""))[:19].replace("T", " ")
                    ev = str(h.get("event", ""))
                    history_lines.append(f"  `{ts}` {ev}")

        delta_lines = []
        if isinstance(delta, dict):
            for k, v in list(delta.items())[:8]:
                delta_lines.append(f"  `{k}` = `{v}`")

        md = (
            f"## 🧠 Meta ChangePackage\n\n"
            f"**ID:** `{change_id[:16]}…`  \n"
            f"**Kind:** {kind}  \n"
            f"**Stage:** `{stage}`  \n"
            f"**Source:** {source}  \n"
            f"**Created:** {created}  \n"
            f"**Constitution:** {const_str}  \n"
        )
        if const_notes:
            md += f"**Notes:** {const_notes}  \n"
        if delta_lines:
            md += "\n### Config Delta\n\n" + "\n".join(delta_lines) + "\n"
        if rationale:
            md += f"\n### Rationale\n\n{rationale[:400]}\n"
        if history_lines:
            md += "\n### Recent History\n\n" + "\n".join(history_lines) + "\n"
        md += "\n---\n\n_Meta entries are read-only in the inbox — manage via MetaManager._\n"
        return md

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
            except NoMatches:
                logger.debug("InboxScreen filter pill #filter-%s not mounted", fid)
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
        elif bid in ("filter-all", "filter-homework", "filter-adjustments", "filter-meta"):
            self._set_filter(bid.replace("filter-", ""))

    def action_approve_all(self) -> None:
        """Dispatch the approve-all work onto a Textual worker thread.

        US-013 — Phase 96 batch homework can mount 17+ rows in the queue;
        the legacy main-thread loop (3 sub-loops + MetaManager.drain inline)
        locked the TUI for several seconds on a real run. Now we snapshot
        the visible row lists on the main thread, then hand them to a
        @work(thread=True) worker so the compositor stays responsive.

        Progress is published back to the main thread via
        ``self.app.call_from_thread(self._approve_all_progress, ...)``.
        On completion the worker calls ``_approve_all_complete`` which
        emits the summary notification, reloads the queue, and auto-hides
        the progress bar after 3 seconds.
        """
        if self._approve_all_running:
            # Second press while a worker is in flight — guard against
            # the parallel-spawn race. The progress bar is already
            # visible; the operator can wait for completion.
            self.notify("Approve-all already running…", severity="information")
            return

        # Snapshot visible rows on the main thread. The worker thread
        # must not read self._current_rows directly — the 5s auto-refresh
        # timer can mutate it under the worker, and Textual's reactive
        # internals are not thread-safe.
        hw_rows = [r for r in self._current_rows if r.entry_type == "homework"]
        meta_rows = [r for r in self._current_rows if r.entry_type == "meta"]
        total = len(hw_rows) + 1 + len(meta_rows)  # +1 for adjustment batch
        if total <= 0:
            self.notify("No entries to approve.", severity="information")
            return

        self._approve_all_running = True
        # Cancel any pending auto-hide from a prior approve-all so the
        # banner doesn't disappear mid-run.
        self._cancel_progress_autohide()
        # Show progress bar at 0/total before the worker starts.
        self._set_progress_visible(True)
        self._update_progress(0, total, "Starting…")

        self._approve_all_worker(hw_rows, meta_rows, total)

    @work(thread=True, exclusive=True, group="inbox-approve-all")
    def _approve_all_worker(
        self,
        hw_rows: list[UnifiedInboxRow],
        meta_rows: list[UnifiedInboxRow],
        total: int,
    ) -> None:
        """Run the three approval loops + drain on a worker thread.

        All UI mutations route through ``self.app.call_from_thread`` so
        the Textual event loop owns every widget update. The worker
        itself does only file I/O + module-level approver/reviewer calls.
        """
        hw_approved = 0
        hw_failed = 0
        adj_approved = 0
        adj_skipped = 0
        adj_failed = 0
        meta_approved = 0
        meta_failed = 0

        current = 0

        # ── 1. Homework loop ────────────────────────────────────────────
        try:
            reviewer = HomeworkReviewer(store=HomeworkStore())
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
                    logger.exception(
                        "InboxScreen._approve_all_worker: homework approve failed"
                    )
                current += 1
                self.app.call_from_thread(
                    self._update_progress,
                    current,
                    total,
                    f"Homework {hw_approved}/{len(hw_rows)} approved",
                )
        except Exception as e:
            logger.exception(
                "InboxScreen._approve_all_worker: homework loop init failed"
            )
            self.app.call_from_thread(
                self.notify, f"Homework approve init error: {e}", severity="error"
            )

        # ── 2. Adjustments batch (treated as a single step) ─────────────
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
            logger.exception(
                "InboxScreen._approve_all_worker: adjustment approve failed"
            )
            self.app.call_from_thread(
                self.notify, f"Adjustment approve error: {e}", severity="error"
            )
        current += 1
        self.app.call_from_thread(
            self._update_progress,
            current,
            total,
            f"Adjustments {adj_approved} approved",
        )

        # ── 3. Meta packages (uses existing helper; runs on worker thread).
        # _approve_meta_packages is file I/O + ApprovalQueue + drain — safe
        # to call from a worker so long as we don't touch widgets directly
        # inside it. We pre-snapshotted meta_rows, so route through a small
        # private variant that takes the snapshot.
        meta_approved, meta_failed = self._approve_meta_rows(meta_rows)
        current += len(meta_rows)
        self.app.call_from_thread(
            self._update_progress,
            current,
            total,
            f"Meta {meta_approved} approved",
        )

        # ── 4. Summary, reload, auto-hide ────────────────────────────────
        total_failed = hw_failed + adj_failed + meta_failed
        msg = (
            f"Approved {hw_approved} homework + {adj_approved} adjustments + "
            f"{meta_approved} meta packages; "
            f"{total_failed} failed; {adj_skipped} invalid skipped."
        )
        approved_count = hw_approved + adj_approved + meta_approved
        self.app.call_from_thread(
            self._approve_all_complete,
            msg,
            approved_count,
            bool(total_failed),
        )

    def _approve_meta_rows(
        self, meta_rows: list[UnifiedInboxRow]
    ) -> tuple[int, int]:
        """Worker-thread variant of _approve_meta_packages.

        Accepts a pre-snapshotted list of meta rows instead of reading
        ``self._current_rows`` (which is owned by the main thread).
        """
        if not meta_rows:
            return (0, 0)

        approved = 0
        failed = 0

        try:
            from src.scanner.automation.approval_queue import ApprovalQueue
            queue = ApprovalQueue()
        except Exception as e:
            logger.exception(
                "InboxScreen._approve_meta_rows: ApprovalQueue init failed"
            )
            self.app.call_from_thread(
                self.notify, f"Meta approve init error: {e}", severity="error"
            )
            return (0, len(meta_rows))

        for row in meta_rows:
            blob = row.raw_entry if isinstance(row.raw_entry, dict) else {}
            change_id = str(blob.get("change_id", ""))
            if not change_id:
                failed += 1
                continue
            try:
                pkg = queue.approve(change_id, reason="bulk approve from inbox")
                if pkg is not None:
                    approved += 1
                else:
                    failed += 1
                    logger.info(
                        "InboxScreen._approve_meta_rows: change_id=%s "
                        "had no pending entry — likely already approved or "
                        "orphan in changes_dir",
                        change_id,
                    )
            except Exception:
                failed += 1
                logger.exception(
                    "InboxScreen._approve_meta_rows: queue.approve raised "
                    "change_id=%s", change_id,
                )

        # Drain inline so packages walk to DEPLOYED_SHADOW immediately
        # rather than sitting in approved/ until a separate trigger fires.
        try:
            from src.scanner.automation.meta_manager import (
                is_enabled as _meta_enabled,
                _build_production_manager,
            )
            if _meta_enabled():
                from src.scanner.automation import meta_manager as _mm
                if _mm._PRODUCTION_MGR is None:
                    _mm._PRODUCTION_MGR = _build_production_manager()
                counters = _mm._PRODUCTION_MGR.drain(current_cycle=0)
                logger.info(
                    "InboxScreen._approve_meta_rows: drain counters=%s",
                    counters,
                )
        except Exception:
            logger.exception(
                "InboxScreen._approve_meta_rows: drain failed — packages "
                "approved but not yet deployed; next drain call will pick "
                "them up"
            )

        return (approved, failed)

    # ── ProgressBar helpers (main-thread only) ──────────────────────────

    def _set_progress_visible(self, visible: bool) -> None:
        """Toggle the progress row visibility. Main thread only."""
        try:
            row = self.query_one("#approve-all-progress-row", Horizontal)
            row.display = visible
        except NoMatches:
            logger.debug(
                "InboxScreen._set_progress_visible: row not mounted yet"
            )

    def _update_progress(self, current: int, total: int, message: str) -> None:
        """Update progress bar value + label. Routed from the worker via
        ``call_from_thread``; safe to call directly on the main thread too.
        """
        try:
            label = self.query_one("#approve-all-progress-label", Label)
            label.update(f"Approve all — {message} ({current}/{total})")
        except NoMatches:
            pass
        try:
            bar = self.query_one("#approve-all-progress", ProgressBar)
            # Keep total in sync in case the visible queue shrank between
            # snapshot and now (paranoia — total is fixed per worker run).
            bar.update(total=total, progress=current)
        except NoMatches:
            pass

    def _approve_all_progress(self, current: int, total: int) -> None:
        """Public progress callback (AC contract).

        US-013 acceptance criterion specifies the entry point as
        ``self._approve_all_progress(current, total)`` — keep this thin
        wrapper around ``_update_progress`` so tests + future callers can
        target the documented surface even though the worker uses the
        richer message-bearing variant.
        """
        self._update_progress(current, total, "")

    def _approve_all_complete(
        self,
        summary_msg: str,
        approved_count: int,
        had_failures: bool,
    ) -> None:
        """Called from the worker thread on completion. Main-thread only."""
        self._approve_all_running = False
        # Final progress bar state — surface "approved N items" then hide.
        try:
            label = self.query_one("#approve-all-progress-label", Label)
            label.update(f"approved {approved_count} items")
        except NoMatches:
            pass
        try:
            bar = self.query_one("#approve-all-progress", ProgressBar)
            # Force the bar to "complete" visually regardless of any
            # partial-fail counts the worker accumulated.
            current_total = bar.total or max(approved_count, 1)
            bar.update(progress=current_total)
        except NoMatches:
            pass

        self.notify(
            summary_msg, severity="warning" if had_failures else "information"
        )
        self._load_proposals()

        # Auto-hide after 3s — give the operator a moment to read the
        # completion banner before the row collapses.
        self._cancel_progress_autohide()
        self._approve_all_hide_timer = self.set_timer(
            3.0, lambda: self._set_progress_visible(False)
        )

    def _cancel_progress_autohide(self) -> None:
        """Cancel any pending auto-hide timer for the progress banner."""
        if self._approve_all_hide_timer is not None:
            try:
                self._approve_all_hide_timer.stop()
            except Exception:
                pass
            self._approve_all_hide_timer = None

    def _approve_meta_packages(self) -> tuple[int, int]:
        """Approve every visible meta row + drain so packages actuate.

        Returns ``(approved, failed)`` — the count of packages that
        successfully moved through ApprovalQueue.approve and the count
        that errored. Drain runs inline so the operator sees immediate
        DEPLOYED_SHADOW state in the inbox after pressing approve-all,
        not "approved but not deployed for an hour" silence.
        """
        meta_rows = [r for r in self._current_rows if r.entry_type == "meta"]
        if not meta_rows:
            return (0, 0)

        approved = 0
        failed = 0

        try:
            from src.scanner.automation.approval_queue import ApprovalQueue
            queue = ApprovalQueue()
        except Exception as e:
            logger.exception(
                "InboxScreen._approve_meta_packages: ApprovalQueue init failed"
            )
            self.notify(f"Meta approve init error: {e}", severity="error")
            return (0, len(meta_rows))

        for row in meta_rows:
            blob = row.raw_entry if isinstance(row.raw_entry, dict) else {}
            change_id = str(blob.get("change_id", ""))
            if not change_id:
                failed += 1
                continue
            try:
                pkg = queue.approve(change_id, reason="bulk approve from inbox")
                if pkg is not None:
                    approved += 1
                else:
                    # ApprovalQueue.approve returns None when pending file
                    # is missing (already approved / orphan in changes_dir
                    # without corresponding pending entry).
                    failed += 1
                    logger.info(
                        "InboxScreen._approve_meta_packages: change_id=%s "
                        "had no pending entry — likely already approved or "
                        "orphan in changes_dir",
                        change_id,
                    )
            except Exception:
                failed += 1
                logger.exception(
                    "InboxScreen._approve_meta_packages: queue.approve raised "
                    "change_id=%s", change_id,
                )

        # Drain inline so the deployment actually happens. Without this,
        # approved packages would sit in .claude/approval_queue/approved/
        # until the next caller of MetaManager.drain — which today is no
        # one in TUI runtime (only scripts/cybernetic_promote.py calls
        # drain). Operator would see "approved" notification but no
        # DEPLOYED_SHADOW state appear in the inbox.
        try:
            from src.scanner.automation.meta_manager import (
                is_enabled as _meta_enabled,
                _build_production_manager,
            )
            if _meta_enabled():
                # Reuse the already-built singleton so we don't reload
                # episodic memory + StagedDeployer on every approve-all.
                from src.scanner.automation import meta_manager as _mm
                if _mm._PRODUCTION_MGR is None:
                    _mm._PRODUCTION_MGR = _build_production_manager()
                counters = _mm._PRODUCTION_MGR.drain(current_cycle=0)
                logger.info(
                    "InboxScreen._approve_meta_packages: drain counters=%s",
                    counters,
                )
        except Exception:
            logger.exception(
                "InboxScreen._approve_meta_packages: drain failed — packages "
                "approved but not yet deployed; next drain call will pick "
                "them up"
            )

        return (approved, failed)

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
        except NoMatches:
            logger.debug("InboxScreen._update_tab_label: #main-tabs/inbox tab not mounted")

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
        except (RuntimeError, AttributeError, OSError) as e:
            # Event bus unavailable — 5s polling interval is the fallback
            logger.debug("InboxScreen event bus subscribe failed (using poll fallback): %s", e)

    def _focused_row(self) -> Optional[UnifiedInboxRow]:
        """Return the UnifiedInboxRow for the currently highlighted DataTable row."""
        try:
            table = self.query_one("#inbox-queue", DataTable)
        except NoMatches:
            logger.debug("InboxScreen._focused_row: #inbox-queue not mounted")
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
        except (IndexError, AttributeError, KeyError) as e:
            logger.debug("InboxScreen._focused_row cursor lookup failed: %s", e)
            return None

    def update_from_snapshot(self, snap: Any) -> None:
        """Called by app.py on every data refresh — re-checks proposal count."""
        self._load_proposals()

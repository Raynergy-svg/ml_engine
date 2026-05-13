"""
BUDDY — Cyberpunk Command Bridge TUI
Built on Textual. Inspired by Dolphie, Harlequin, Toad.

Dual-mode: --live (real OANDA data) or --demo (simulated data).
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import random
import re
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from rich.text import Text

from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.reactive import reactive
from textual.widgets import (
    DataTable,
    Footer,
    Label,
    RichLog,
    Sparkline,
    Static,
    TabbedContent,
    TabPane,
)

from src.tui.data_provider import DataProvider, DashboardSnapshot
from src.tui.heartbeat import write_heartbeat
from src.tui.screens.command_palette_modal import CommandPaletteModal
from src.tui.screens.abort_confirm_modal import AbortConfirmModal
from src.tui.screens.kill_modal import KillModal
from src.tui.screens.log_viewer_modal import LogViewerModal
from src.tui.screens.mode_modal import ModeConfirmModal, check_oanda_credentials
from src.tui.screens.trades_screen import TradesScreen
from src.tui.screens.agents_screen import AgentsScreen
from src.tui.screens.config_screen import ConfigScreen
from src.tui.screens.diagnostics_screen import DiagnosticsScreen
from src.tui.screens.journal_screen import JournalScreen
from src.tui.screens.inbox_screen import InboxScreen
from src.tui.screens.rules_screen import RulesScreen
from src.tui.widgets.state_strip import StateStrip
from src.tui.widgets.staleness_banner import StalenessBanner
from src.tui.widgets.liveness_badge import LivenessBadge
from src.tui.widgets.stats_bar import ScanCounters, StatsBar
from src.tui.widgets.phase_indicator import PhaseIndicator, PhaseState

logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
STRATEGIC_LOG_PATH = PROJECT_ROOT / ".claude" / "brain" / "strategic_log.md"


def _emit_signal_veto(
    app: Optional[App],
    bus: Any,
    pending_event: dict,
    log_path: Optional[Path] = None,
) -> bool:
    """Publish control.signal_veto and append the audit entry to strategic_log.md.

    Pure side-effect helper extracted for testability — the unit test drives
    it directly with a real EventBus + tmp_path log without mounting the App.

    Returns True if the veto was published. The audit log is best-effort:
    a failed disk write does not block the veto publish, but is logged.
    """
    payload = pending_event.get("payload", {}) or {}
    signal_id = str(payload.get("signal_id", ""))
    if not signal_id:
        logger.warning("_emit_signal_veto: refusing to publish — empty signal_id")
        return False
    pair = str(payload.get("pair", "?"))
    direction = str(payload.get("direction", "?"))

    bus.publish("control.signal_veto", {
        "signal_id": signal_id,
        "vetoed_by": "operator",
        "pair": pair,
        "direction": direction,
    })
    logger.info("control.signal_veto emitted for signal_id=%s pair=%s", signal_id, pair)

    target = log_path if log_path is not None else STRATEGIC_LOG_PATH
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).isoformat()
        entry = (
            f"\n### {ts} — Operator SUPERVISOR_ABORT\n"
            f"**actor:** TUI  **action:** supervisor_abort  "
            f"**pair:** {pair}  **signal_id:** {signal_id}\n"
        )
        with target.open("a", encoding="utf-8") as fh:
            fh.write(entry)
    except Exception as exc:
        logger.debug("_emit_signal_veto: strategic_log write failed: %s", exc)

    if app is not None:
        try:
            app.notify(
                f"Signal VETOED: {pair} {direction}",
                title="Operator Abort",
                severity="warning",
            )
        except Exception:
            pass
        try:
            app.query_one("#brain-log", RichLog).write(Text.from_markup(
                f"[bold red]◈ SIGNAL VETOED[/] — operator abort: [yellow]{pair} {direction}[/] "
                f"[dim]signal_id={signal_id[:8]}…[/]"
            ))
        except Exception:
            pass
    return True

# ── Simulated Data (demo mode) ──────────────────────────────────────

DEMO_AGENTS = [
    ("trend", 0.82, "BULLISH"), ("m_rev", 0.61, "NEUTRAL"),
    ("volat", 0.55, "LOW-VOL"), ("risk", 0.91, "SAFE"),
    ("uncrt", 0.31, "CLEAR"), ("momt", 0.74, "STRONG"),
    ("exec", 0.68, "READY"), ("news", 0.45, "CALM"),
    ("mtf", 0.72, "ALIGNED"), ("pair", 0.66, "GOOD"),
    ("sess", 0.58, "LONDON"), ("sr", 0.77, "ABOVE-S1"),
]

BRAIN_TEMPLATES = [
    "[cyan]▸ Scanning {pair} on H1...[/]",
    "[dim]  TCN pred: {v1:.2f} | Ridge: {v2:.2f} | RF: {v3:.2f}[/]",
    "[dim]  Ensemble confidence: {v1:.2f} ± {v4:.3f}[/]",
    "[magenta]▸ Agent team voting on {pair}...[/]",
    "[dim]  ▸ trend_agent: {dir} ({v1:.2f})[/]",
    "[dim]  ▸ risk_sentinel: PASS ({v1:.2f})[/]",
    "[yellow]▸ Gate check: CONF {g1} | MOM {g2} | RISK {g3}[/]",
    "[dim]  R:R = {rr:.1f}:1 — {action}[/]",
    "[green]✓ Signal: {pair} {dir} @ {price:.5f}[/]",
    "[cyan]▸ Drawdown guardian: portfolio risk {dd:.1f}% — {dd_status}[/]",
    "[dim]  RL weights updated from last 10 outcomes[/]",
    "[yellow]▸ Skipping {pair}: uncertainty {v1:.2f} > threshold[/]",
    "[red]✗ {pair} rejected: R:R {rr:.1f}:1 below minimum 1.2:1[/]",
    "[dim]  Scan cycle #{cycle} complete — {v1:.1f}s[/]",
    "[cyan]▸ Next scan in {cycle}s...[/]",
]

PAIRS = ["EUR/USD", "GBP/JPY", "USD/CAD", "AUD/NZD", "EUR/GBP", "USD/CHF"]


def _fake_brain_line() -> str:
    tmpl = random.choice(BRAIN_TEMPLATES)
    return tmpl.format(
        pair=random.choice(PAIRS), dir=random.choice(["BUY", "SELL"]),
        v1=random.uniform(0.5, 0.9), v2=random.uniform(0.5, 0.8),
        v3=random.uniform(0.5, 0.8), v4=random.uniform(0.02, 0.08),
        g1=random.choice(["✓", "✗"]), g2=random.choice(["✓", "✗"]),
        g3=random.choice(["✓", "✗"]), rr=random.uniform(0.8, 2.5),
        action=random.choice(["EXECUTING...", "MONITORING", "QUEUED"]),
        price=1.0 + random.random() * 0.5,
        dd=random.uniform(1, 8), dd_status=random.choice(["OK ✓", "CAUTION ▲"]),
        cycle=random.randint(30, 999),
    )


# ── Widgets ─────────────────────────────────────────────────────────


_BRAIN_FEED_PATH = Path(".claude/brain/feed.jsonl")
_BRAIN_FEED_MAX_BYTES = 5 * 1024 * 1024  # 5 MB rotation ceiling
_BRAIN_MARKUP_RE = re.compile(r"\[/?[a-zA-Z0-9 _#,\.\-]*\]")


def _strip_rich_markup(markup: str) -> str:
    """Strip Rich/Textual ``[tag]...[/]`` markup so the disk-tee is
    plain text. Keeps the message content; drops only the formatting
    brackets. Conservative regex matches simple tag names + style
    attributes — anything fancier passes through unchanged."""
    return _BRAIN_MARKUP_RE.sub("", markup).strip()


def _tee_brain_to_disk(markup: str) -> None:
    """Append a brain-feed line to .claude/brain/feed.jsonl.

    Mythos audit 2026-04-30 — closes the verification gap operator
    flagged. Pre-fix the brain feed (F1 panel) was in-memory only;
    confirming "this commit fires" required scraping unrelated disk
    artifacts. Now every _write_brain call appends a JSONL line so:
        tail -f .claude/brain/feed.jsonl | grep "meta"
    works as the operator's verification surface.

    Schema: ``{"ts": iso, "msg": stripped_text, "raw": markup}``.
    Both stripped + raw included so grep-on-plain-text works AND
    the original Rich markup is recoverable for forensic playback.

    Rotation: when the file exceeds 5 MB, atomically rotate to
    feed.jsonl.1 and start fresh. Operator can `cat feed.jsonl.1
    feed.jsonl` for full history. Caller wraps this in try/except —
    any error here MUST NOT break TUI rendering.
    """
    try:
        _BRAIN_FEED_PATH.parent.mkdir(parents=True, exist_ok=True)
    except OSError:
        return

    # Cheap rotation check (no fsync, no lock — best-effort).
    try:
        if _BRAIN_FEED_PATH.exists() and _BRAIN_FEED_PATH.stat().st_size > _BRAIN_FEED_MAX_BYTES:
            rotated = _BRAIN_FEED_PATH.with_suffix(".jsonl.1")
            try:
                rotated.unlink()
            except FileNotFoundError:
                pass
            _BRAIN_FEED_PATH.rename(rotated)
    except OSError:
        pass

    line = json.dumps(
        {
            "ts": datetime.now(timezone.utc).isoformat(),
            "msg": _strip_rich_markup(markup),
            "raw": markup,
        },
        ensure_ascii=False,
    )
    try:
        with _BRAIN_FEED_PATH.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
    except OSError:
        return


def _scanner_alive_from_sources(snap: DashboardSnapshot | None, scanner: object | None) -> bool:
    """Return whether the runtime scanner is alive from real in-process sources."""
    cycle = int(getattr(snap, "scan_cycle_count", 0) or 0) if snap is not None else 0
    has_open_trades = bool(getattr(snap, "trades", []) or []) if snap is not None else False
    embedded_ready = bool(
        getattr(scanner, "is_ready", False)
        or getattr(scanner, "_running", False)
    )
    return bool(cycle > 0 or has_open_trades or embedded_ready)


def format_meta_brain_line(blob: dict) -> str:
    """Format a single meta-pipeline ledger entry into a brain-log line.

    Mythos audit 2026-04-29 Tier-7 step C — extracted as a free function
    so the tail-watcher's formatting logic is unit-testable without
    spinning up a Textual App. Returns Rich markup ready for
    ``RichLog.write(Text.from_markup(...))``.

    Stage → color/icon map:
      rejected/aborted          → red ✗
      approved/deployed_live    → green ✓
      deployed_shadow/canary    → magenta ◆
      everything else (intake,
      proposing, evaluating,
      awaiting_approval, ...)   → cyan ▸
    """
    change_id = str(blob.get("change_id", "?"))[:8]
    stage = str(blob.get("stage", "?"))
    event = blob.get("event")
    kind = blob.get("kind")

    if stage in ("rejected", "aborted"):
        color = "red"
        icon = "✗"
    elif stage in ("approved", "deployed_live"):
        color = "green"
        icon = "✓"
    elif stage in ("deployed_shadow", "deployed_canary"):
        color = "magenta"
        icon = "◆"
    else:
        color = "cyan"
        icon = "▸"

    parts = [f"⟪{change_id}⟫", stage]
    if event:
        parts.append(f"event={event}")
    if kind:
        parts.append(f"kind={kind}")
    return f"[{color}]  {icon} meta {' '.join(parts)}[/]"


class HeaderBar(Static):
    """Persistent header: NAV, P/L, connection status."""

    can_focus = True  # Click-to-focus for the 'c' copy hotkey
    nav = reactive(0.0)
    pnl = reactive(0.0)
    open_count = reactive(0)
    oanda_ok = reactive(False)
    scanner_ok = reactive(False)
    mode_label = reactive("DEMO")

    def render(self) -> Text:
        now = datetime.now(timezone.utc).strftime("%H:%M UTC")
        pnl_sign = "+" if self.pnl >= 0 else ""
        pnl_color = "#39ff14" if self.pnl >= 0 else "#ff3158"
        oanda_style = "bold #39ff14" if self.oanda_ok else "bold #ff3158"
        oanda_text = "● LIVE" if self.oanda_ok else "● OFF"
        scanner_style = "bold #39ff14" if self.scanner_ok else "bold #7483b8"
        scanner_text = "● ACTIVE" if self.scanner_ok else "● IDLE"

        t = Text()
        t.append("  ▰ ", style="bold #ff2bd6")
        t.append("BUDDY//2046", style="bold #00f5ff")
        t.append(
            f" [{self.mode_label}]",
            style="bold #b84dff" if self.mode_label == "DEMO" else "bold #39ff14",
        )
        t.append("  ║  ", style="#26304f")
        t.append("NAV ", style="#7483b8")
        t.append(f"${self.nav:,.0f}", style="bold #39ff14")
        t.append("  │  ", style="#26304f")
        t.append("P/L ", style="#7483b8")
        t.append(f"{pnl_sign}${self.pnl:,.0f}", style=f"bold {pnl_color}")
        t.append("  │  ", style="#26304f")
        t.append(f"Open: {self.open_count}", style="#7483b8")
        t.append("  │  ", style="#26304f")
        t.append("OANDA ", style="#7483b8")
        t.append(oanda_text, style=oanda_style)
        t.append("  │  ", style="#26304f")
        t.append("Scanner ", style="#7483b8")
        t.append(scanner_text, style=scanner_style)
        t.append("  ║  ", style="#26304f")
        t.append(f"{now}", style="#7483b8")
        return t


class AgentPanel(Static):
    """15-agent panel with colored bar indicators (see ``ScannerAgentTeam._BASE_WEIGHTS``). Reads from snapshot or demo."""

    can_focus = True  # Click-to-focus for the 'c' copy hotkey

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._agents: list[tuple[str, float, str]] = list(DEMO_AGENTS)

    def update_from_snapshot(self, snap: DashboardSnapshot) -> None:
        if snap.agents:
            self._agents = [(a.name, a.score, a.signal) for a in snap.agents]

    def render(self) -> Text:
        t = Text()
        for name, score, signal in self._agents:
            s = max(0.0, min(1.0, score))
            filled = int(s * 10)
            empty = 10 - filled
            color = "#39ff14" if s >= 0.7 else "#ffb000" if s >= 0.5 else "#ff3158"
            t.append(f"  {name:<6}", style="#7483b8")
            t.append(" █" * filled, style=color)
            t.append("░" * empty, style="#26304f")
            t.append(f" {s:.2f}", style=f"bold {color}")
            t.append(f"  {signal}", style="#7483b8")
            t.append("\n")

        if self._agents:
            avg = sum(s for _, s, _ in self._agents) / len(self._agents)
            bar_filled = int(avg * 20)
            bar_empty = 20 - bar_filled
            t.append("  ─────────────────────────────\n", style="#26304f")
            t.append("  WEIGHTED ", style="#7483b8")
            t.append(f"{avg:.2f}", style="bold #00f5ff")
            t.append("  [", style="#26304f")
            t.append("█" * bar_filled, style="#00f5ff")
            t.append("░" * bar_empty, style="#26304f")
            t.append("]", style="#26304f")
        return t


class ReflectionLogReader:
    """Tail logs/reflection_log.jsonl and stream formatted lines to #reflection-log.

    The Claude self-improvement subprocess (src/scanner/automation/claude_subprocess.py)
    appends one JSON line per spawn. This reader seeks from its last-read offset
    each poll cycle, parses new lines, and dispatches formatted markup via the
    provided thread-safe callback.

    Design:
      - File missing → emit "waiting" once per session, then no-op
      - Bad JSON line → log+skip, don't crash the reader loop
      - File truncated (offset > size) → rewind to 0
      - Idempotent: each entry formatted exactly once (offset persisted in memory)
    """

    def __init__(
        self,
        log_path: Path,
        callback,  # Callable[[str], None]
        stop_flag,  # threading.Event
        poll_interval: float = 2.0,
    ) -> None:
        self._log_path = log_path
        self._callback = callback
        self._stop = stop_flag
        self._poll_interval = poll_interval
        self._offset = 0
        self._announced_waiting = False
        # Mythos audit 2026-05-04 — make 159+ entries of history
        # SCANNABLE. Snapshot the file size at boot so we can render a
        # "▼ NEW SESSION ▼" marker once replay catches up; track the
        # last-rendered day so a yellow "── Mon May 04 2026 ──" line
        # gets injected on day rollovers.
        try:
            self._boot_eof = log_path.stat().st_size if log_path.exists() else 0
        except OSError:
            self._boot_eof = 0
        self._boot_marker_emitted = False
        self._last_day: str = ""

        # Mythos audit 2026-05-05 — operator: panel showed "old logs"
        # because Claude reflection_log.jsonl is dormant under no-LLM.
        # Multiplex two extra live "system thinking" sources into the
        # same panel: meta-pipeline ledger + autonomous_trainer log.
        # Each tracked with its own offset + boot_eof so day-rollover
        # / boot-marker logic reuses the primary source's _last_day.
        # The PANEL display now matches the per-box `c` snapshot
        # (commit a308b9d) — same merged stream, formatted live.
        from pathlib import Path as _P
        _root = _P(__file__).resolve().parents[2]
        self._aux_sources = [
            {
                "path": _root / ".claude" / "meta" / "changes.jsonl",
                "tag": "meta",
                "color": "cyan",
                "formatter": self._format_meta,
                "offset": 0,
                "announced_waiting": False,
            },
            {
                "path": _root / "logs" / "autonomous_trainer.jsonl",
                "tag": "train",
                "color": "yellow",
                "formatter": self._format_autotrain,
                "offset": 0,
                "announced_waiting": False,
            },
        ]
        for src in self._aux_sources:
            try:
                src["offset"] = (
                    src["path"].stat().st_size if src["path"].exists() else 0
                )
            except OSError:
                src["offset"] = 0

    def _format_meta(self, entry: dict) -> str:
        """Format a .claude/meta/changes.jsonl entry."""
        ts_local = self._local_dt(entry.get("updated_at") or entry.get("ts", ""))
        ts = ts_local.strftime("%H:%M") if ts_local else "--:--"
        cid = (entry.get("change_id", "") or "")[:12]
        stage = entry.get("stage", "?")
        event = entry.get("event") or ""
        kind = entry.get("kind", "?")
        # Color stage: green for promotions, red for rejections, dim otherwise
        color = "green" if "promoted" in event or stage in ("deployed_live", "closed") else \
                "red" if stage in ("rejected", "aborted") else "cyan"
        return (
            f"[{color}]  ◆ {ts} meta cid={cid} stage={stage} "
            f"event={event} kind={kind}[/]"
        )

    def _format_autotrain(self, entry: dict) -> str:
        """Format a logs/autonomous_trainer.jsonl entry."""
        ts_local = self._local_dt(entry.get("ts", ""))
        ts = ts_local.strftime("%H:%M") if ts_local else "--:--"
        tid = str(entry.get("trade_id", "?"))[:20]
        success = bool(entry.get("success"))
        hyp = str(entry.get("hypothesis") or "").strip()[:80]
        sym = "[green]✓[/]" if success else "[yellow]○[/]"
        return f"[yellow]  ▣ {ts} train {sym} {tid} → {hyp}[/]"

    @staticmethod
    def _local_dt(ts_raw):
        """Parse ISO timestamp → datetime in operator's local timezone."""
        from datetime import datetime as _dt
        try:
            ts_obj = _dt.fromisoformat(str(ts_raw).replace("Z", "+00:00"))
        except (ValueError, TypeError):
            return None
        try:
            return ts_obj.astimezone()
        except Exception:
            return ts_obj

    def _format(self, entry: dict) -> str:
        # Trade ID — keep first 12 chars (human-recognizable prefix)
        tid = str(entry.get("trade_id", "?"))[:12]
        mode = str(entry.get("mode", "?"))
        dur = float(entry.get("duration_seconds", 0) or 0)
        cost = float(entry.get("cost_usd", 0) or 0)
        success = bool(entry.get("success"))
        hyp = str(entry.get("hypothesis") or "").strip()[:70]
        err = str(entry.get("error") or "").strip()[:60]
        # Local-timezone HH:MM (was raw UTC). Lets operator correlate
        # entries with their wall-clock without mental TZ math.
        ts_local = self._local_dt(entry.get("ts", ""))
        ts = ts_local.strftime("%H:%M") if ts_local else "--:--"

        if not success and err:
            return f"[red]  ✗ {ts} {tid} {mode} FAILED: {err}[/]"
        if not success:
            return f"[dim]  ○ {ts} {tid} {mode} skipped (budget/lock/noop)[/]"
        if mode == "deep":
            return (
                f"[magenta]  ◆ {ts} {tid} DEEP {dur:.0f}s "
                f"${cost:.2f} → {hyp}[/]"
            )
        return f"[cyan]  ▸ {ts} {tid} light {dur:.0f}s → {hyp}[/]"

    def _maybe_day_separator(self, entry: dict):
        """Return a '── Day Date Year ──' line when the day rolls, else None.

        Sticky watermark — emitted once per day-change as we walk through
        the log. Lets the operator see the boundary between, say, April
        15's losing-streak entries and today's clean state. Schema-tolerant:
        meta entries use `updated_at`, claude entries use `ts`.
        """
        ts_local = self._local_dt(
            entry.get("ts") or entry.get("updated_at", "")
        )
        if ts_local is None:
            return None
        day = ts_local.strftime("%a %b %d %Y")
        if day != self._last_day:
            self._last_day = day
            return f"[bold yellow]── {day} ──[/]"
        return None

    def run(self) -> None:
        """Main loop. Call from a thread; exits when stop_flag is set."""
        import time as _time
        while not self._stop.is_set():
            try:
                if not self._log_path.exists():
                    if not self._announced_waiting:
                        self._announced_waiting = True
                        self._callback(
                            "[dim]  waiting for first reflection… (fires on trade close)[/]"
                        )
                    self._poll_aux_sources()
                    self._stop.wait(self._poll_interval)
                    continue

                size = self._log_path.stat().st_size
                if size < self._offset:
                    # File was truncated/rotated — reset
                    self._offset = 0
                if size == self._offset:
                    self._poll_aux_sources()
                    self._stop.wait(self._poll_interval)
                    continue

                # Once we have a real file, clear the waiting flag so a later
                # deletion + recreation re-announces.
                self._announced_waiting = False
                # Mythos audit 2026-05-05 — `for line in f:` uses the
                # iterator protocol which DISABLES f.tell() afterwards
                # ("telling position disabled by next() call" — visible
                # in buddy_debug.log after commit 26bdb88). Use a
                # readline() loop so f.tell() stays valid for offset
                # tracking + boot-marker boundary detection.
                with open(self._log_path, "r") as f:
                    f.seek(self._offset)
                    while True:
                        raw = f.readline()
                        if not raw:
                            break
                        if not raw.strip():
                            continue
                        try:
                            entry = json.loads(raw)
                        except json.JSONDecodeError:
                            continue
                        try:
                            sep = self._maybe_day_separator(entry)
                            if sep:
                                self._callback(sep)
                        except Exception:
                            pass
                        try:
                            self._callback(self._format(entry))
                        except Exception:
                            pass

                        # Boot marker: once we cross pre-session EOF.
                        if (
                            not self._boot_marker_emitted
                            and self._boot_eof > 0
                            and f.tell() >= self._boot_eof
                        ):
                            self._boot_marker_emitted = True
                            try:
                                from datetime import datetime as _dt2
                                now_local = _dt2.now().astimezone()
                                stamp = now_local.strftime("%H:%M:%S")
                                self._callback(
                                    f"[bold #ff2bd6]"
                                    f"━━━━━━ ▼ NEW SESSION @ {stamp} "
                                    f"(history above) ▼ ━━━━━━[/]"
                                )
                            except Exception:
                                pass
                    # Edge case: empty file or already at EOF.
                    if (
                        not self._boot_marker_emitted
                        and f.tell() >= self._boot_eof
                    ):
                        self._boot_marker_emitted = True
                        try:
                            from datetime import datetime as _dt2
                            now_local = _dt2.now().astimezone()
                            stamp = now_local.strftime("%H:%M:%S")
                            self._callback(
                                f"[bold #ff2bd6]"
                                f"━━━━━━ ▼ NEW SESSION @ {stamp} "
                                f"(history above) ▼ ━━━━━━[/]"
                            )
                        except Exception:
                            pass
                    self._offset = f.tell()
            except Exception as e:
                logger.debug("ReflectionLogReader error: %s", e)

            self._poll_aux_sources()

            self._stop.wait(self._poll_interval)

    def _poll_aux_sources(self) -> None:
        """Poll meta/autotrain side streams even when Claude log is idle."""
        # Multiplex aux sources (meta + autotrainer). Each reuses the
        # primary's day-separator state for chronological coherence.
        # readline() loop keeps f.tell() valid (see primary loop note).
        for src in self._aux_sources:
            try:
                p = src["path"]
                if not p.exists():
                    continue
                size = p.stat().st_size
                if size < src["offset"]:
                    src["offset"] = 0
                if size == src["offset"]:
                    continue
                with open(p, "r") as f:
                    f.seek(src["offset"])
                    while True:
                        raw = f.readline()
                        if not raw:
                            break
                        if not raw.strip():
                            continue
                        try:
                            entry = json.loads(raw)
                        except json.JSONDecodeError:
                            continue
                        try:
                            sep = self._maybe_day_separator(entry)
                            if sep:
                                self._callback(sep)
                        except Exception:
                            pass
                        try:
                            self._callback(src["formatter"](entry))
                        except Exception:
                            pass
                    src["offset"] = f.tell()
            except Exception as e:
                logger.debug(
                    "ReflectionLogReader aux source %s error: %s",
                    src.get("tag", "?"), e,
                )


class RiskPanel(Static):
    """Risk metrics panel. Reads from snapshot or uses defaults.

    Defense-in-depth: coerces any NaN/inf/None upstream value to a stable 0.0
    and renders "—" (em-dash) when no real data has arrived yet so the UI
    never shows literal "nan".
    """

    can_focus = True  # Click-to-focus for the 'c' copy hotkey

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._risk = 0.0
        self._dd = 0.0
        self._max_dd = 0.0
        self._corr_ok = True
        self._has_data = False  # True once we see at least one real snapshot

    @staticmethod
    def _coerce(v: float) -> float:
        import math as _m
        try:
            fv = float(v)
        except (TypeError, ValueError):
            return 0.0
        if _m.isnan(fv) or _m.isinf(fv) or fv < 0.0:
            return 0.0
        return fv

    def update_from_snapshot(self, snap: "DashboardSnapshot") -> None:
        self._risk = self._coerce(snap.portfolio_risk_pct)
        self._dd = self._coerce(snap.drawdown_pct)
        self._max_dd = self._coerce(snap.max_drawdown_pct)
        self._corr_ok = bool(snap.correlation_ok)
        # Consider data "present" once scanner has run at least once
        self._has_data = getattr(snap, "scan_cycle_count", 0) > 0 or (
            self._risk > 0 or self._dd > 0
        )

    def render(self) -> Text:
        pr, dd, mdd = self._risk, self._dd, self._max_dd
        rc = "#39ff14" if pr < 10 else "#ffb000" if pr < 13 else "#ff3158"
        dc = "#39ff14" if dd < 3 else "#ffb000" if dd < 5 else "#ff3158"
        corr_text = "OK ✓" if self._corr_ok else "WARN ▲"
        corr_style = "bold #39ff14" if self._corr_ok else "bold #ffb000"

        # Render em-dash placeholder while we're still waiting for first scan.
        def _fmt(val: float) -> str:
            if not self._has_data and val == 0.0:
                return "    —  "
            return f"{val:.1f}%"

        t = Text()
        t.append("  Portfolio Risk: ", style="#7483b8")
        t.append(f"{_fmt(pr)}\n", style=f"bold {rc}")
        t.append("  Drawdown:       ", style="#7483b8")
        t.append(f"{_fmt(dd)}\n", style=f"bold {dc}")
        t.append("  Max DD Today:   ", style="#7483b8")
        t.append(f"{_fmt(mdd)}\n", style="bold #ffb000")
        t.append("  Correlation:    ", style="#7483b8")
        t.append(f"{corr_text}\n", style=corr_style)
        return t


class MTFConfluencePanel(Static):
    """Multi-Timeframe Confluence visualization."""

    can_focus = True  # Click-to-focus for the 'c' copy hotkey

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._h4_closes: list[float] = []
        self._h1_closes: list[float] = []
        self._m15_closes: list[float] = []
        self._h4_score = 0.0
        self._h1_score = 0.0
        self._m15_score = 0.0
        self._confluence = 0.0

    def update_from_snapshot(self, snap: DashboardSnapshot) -> None:
        if snap.candles_h4:
            self._h4_closes = snap.candles_h4[-20:]
        if snap.candles_h1:
            self._h1_closes = snap.candles_h1[-20:]
        if snap.candles_m15:
            self._m15_closes = snap.candles_m15[-20:]
        self._h4_score = snap.mtf_h4_score
        self._h1_score = snap.mtf_h1_score
        self._m15_score = snap.mtf_m15_score
        self._confluence = snap.mtf_confluence_score

    def _sparkline_str(self, prices: list[float]) -> str:
        if not prices or len(prices) < 2:
            return "░" * 16
        mn, mx = min(prices), max(prices)
        rng = mx - mn if mx != mn else 0.001
        blocks = "▁▂▃▄▅▆▇█"
        return "".join(blocks[min(7, int((p - mn) / rng * 7))] for p in prices[-16:])

    def render(self) -> Text:
        t = Text()
        screens = [
            ("H4 TREND", self._h4_score, self._h4_closes, "SMA"),
            ("H1 WAVE", self._h1_score, self._h1_closes, "RSI"),
            ("M15 ENTRY", self._m15_score, self._m15_closes, "EMA"),
        ]
        weights = [0.50, 0.30, 0.20]

        for label, score, closes, indicator in screens:
            color = "#39ff14" if score >= 0.7 else "#ffb000" if score >= 0.5 else "#ff3158"
            signal = "BULLISH" if score >= 0.7 else "CAUTION" if score >= 0.5 else "WEAK"
            spark = self._sparkline_str(closes)

            t.append(f"  {label:<10}", style="bold #b84dff")
            t.append(f"  {spark}", style=color)
            t.append(f"  {score:.2f}", style=f"bold {color}")
            t.append(f"  {signal:<8}", style=color)
            t.append(f"  {indicator}", style="#7483b8")
            t.append("\n")

        conf = self._confluence or sum(s * w for (_, s, _, _), w in zip(screens, weights))
        conf_color = "#39ff14" if conf >= 0.65 else "#ffb000" if conf >= 0.45 else "#ff3158"
        conf_signal = "PROCEED ✓" if conf >= 0.65 else "CAUTION ▲" if conf >= 0.45 else "REJECT ✗"
        bar_filled = int(conf * 25)
        bar_empty = 25 - bar_filled

        t.append("  ──────────────────────────────────────────────────────\n", style="#26304f")
        t.append("  CONFLUENCE ", style="#7483b8")
        t.append(f"{conf:.2f}", style=f"bold {conf_color}")
        t.append("  [", style="#26304f")
        t.append("█" * bar_filled, style=conf_color)
        t.append("░" * bar_empty, style="#26304f")
        t.append("]  ", style="#26304f")
        t.append(conf_signal, style=f"bold {conf_color}")
        return t


class SystemHealthBar(Static):
    """Bottom status bar with system health metrics."""

    can_focus = True  # Click-to-focus for the 'c' copy hotkey

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._snap: DashboardSnapshot | None = None

    def update_from_snapshot(self, snap: DashboardSnapshot) -> None:
        self._snap = snap

    def render(self) -> Text:
        s = self._snap
        scan_ms = s.scan_duration_ms if s else 0
        api_ms = s.oanda_latency_ms if s else 0
        models = f"{s.models_loaded}/{s.models_total}" if s else "—/—"
        cpu = s.cpu_pct if s else 0
        mem = s.mem_mb if s else 0
        oanda_ok = s.oanda_connected if s else False
        oanda_style = "bold #39ff14" if oanda_ok else "bold #ff3158"
        if not s or not s.scanner_ready:
            scanner_text = "● OFFLINE"
            scanner_style = "bold #ff3158"
        elif s.halted:
            scanner_text = "● HALTED"
            scanner_style = "bold #ff3158"
        elif s.scanner_paused:
            scanner_text = "● PAUSED"
            scanner_style = "bold #ffb000"
        elif s.scan_cycle_count <= 0:
            scanner_text = "● READY"
            scanner_style = "bold #ffb000"
        else:
            scanner_text = f"● {scan_ms:.0f}ms"
            scanner_style = "bold #39ff14"

        t = Text()
        t.append("  Scanner: ", style="#7483b8")
        t.append(scanner_text, style=scanner_style)
        t.append("  │  ", style="#26304f")
        t.append("OANDA: ", style="#7483b8")
        t.append(f"● {api_ms:.0f}ms", style=oanda_style)
        t.append("  │  ", style="#26304f")
        t.append("Models: ", style="#7483b8")
        t.append(f"● {models}", style="bold #39ff14")
        t.append("  │  ", style="#26304f")
        t.append("RL Sync: ", style="#7483b8")
        t.append("● CURRENT", style="bold #39ff14")
        t.append("  │  ", style="#26304f")
        t.append(f"CPU: {cpu:.0f}%", style="#7483b8")
        t.append("  │  ", style="#26304f")
        t.append(f"MEM: {mem:.0f}MB", style="#7483b8")
        return t


# ── Main App ────────────────────────────────────────────────────────


class BuddyApp(App):
    """BUDDY — Cyberpunk Command Bridge."""

    CSS_PATH = "theme.tcss"
    TITLE = "BUDDY"

    BINDINGS = [
        Binding("f1", "switch_tab('overview')", "Overview", show=True),
        Binding("f2", "switch_tab('inbox')", "Inbox", show=True),
        Binding("f3", "switch_tab('trades')", "Trades", show=True),
        Binding("f4", "switch_tab('agents')", "Agents", show=True),
        Binding("f5", "switch_tab('journal')", "Journal", show=True),
        Binding("f6", "switch_tab('config')", "Config", show=True),
        Binding("f7", "switch_tab('rules')", "Rules", show=True),
        Binding("f8", "switch_tab('diag')", "Diagnostics", show=True),
        Binding("f9", "switch_tab('jobs')", "Jobs", show=True),
        Binding("ctrl+f", "cycle_asset_class", "FX⇄Futures", show=True),
        Binding("space", "supervisor_pause", "Pause", show=True),
        Binding("k", "supervisor_kill", "Kill", show=True),
        Binding("m", "supervisor_mode", "Mode", show=True),
        Binding("a", "supervisor_abort", "Abort Signal", show=True),
        # Mythos audit 2026-04-29 — Tier-7 step D: state-preserving
        # safe-restart. Replaces the manual `pkill python -m src.tui &&
        # ./buddy` dance every commit. Ctrl+R is unbound by default
        # in Textual; if the operator has terminal-history-search muscle
        # memory, alias it via `bind ctrl+shift+r` in their shell.
        Binding("ctrl+r", "safe_restart", "Restart", show=True),
        # Mythos audit 2026-05-04 — operator-reported: ctrl+r was being
        # intercepted by the shell (zsh reverse-history-search) before
        # Textual saw it. F12 is a function key reserved by no shell —
        # guaranteed to reach the TUI's binding handler.
        Binding("f12", "safe_restart", "Restart(F12)", show=True),
        Binding("u", "unhalt", "Unhalt", show=True),
        Binding("c", "copy_snapshot", "Copy", show=True),
        Binding("r", "action_refresh_rules", "Refresh Rules", show=True),
        Binding("q", "quit", "Quit", show=True),
        Binding("colon", "open_command_palette", "Commands", show=False),
        Binding("ctrl+l", "open_log_viewer", "Logs", show=False),
        # Tier 2 T7 — brain section editor (briefing.md CRUD with brain_caps
        # pre-write validation + reset-to-template). show=False to keep the
        # footer uncluttered; surfaces via command palette.
        Binding("ctrl+e", "edit_briefing", "Edit Briefing", show=False),
    ]

    # Asset class modes — F7 cycles through them
    _ASSET_CLASSES = ["fx", "futures", "hybrid"]

    def __init__(self, live: bool = False, **kwargs) -> None:
        super().__init__(**kwargs)
        self._live = live
        self._provider = DataProvider(project_root=str(PROJECT_ROOT))
        self._demo_nav = 101420.0
        self._scanner = None  # EmbeddedScanner (live mode only)
        self._reflection_stop = threading.Event()  # Signals ReflectionLogReader to exit
        self._asset_class = "fx"  # current asset class mode
        self._kill_in_progress = False  # True while flatten_all is running (disables hotkeys)
        # Plan C (2026-04-29) — file position into .claude/meta/changes.jsonl;
        # _tick_brain reads forward from here on each 0.8s tick.
        self._brain_ledger_pos: int = 0
        # T3: shared cumulative work-unit counters. Pre-built at app construct
        # so the F1 StatsBar can take a stable reference at compose time; the
        # EmbeddedScanner receives the same instance and bumps it inline.
        self._counters = ScanCounters()
        # Tier 1 T4: shared PhaseState reference. Created here so the
        # PhaseIndicator widget in compose() has a stable target even
        # before _start_scanner instantiates the EmbeddedScanner. The
        # scanner adopts this same instance (see _start_scanner).
        self._phase_state = PhaseState()
        # US-004: pre-trade veto countdown bookkeeping. _active_veto_countdown_cancel
        # is the bound stop-fn for the currently-ticking set_interval handle
        # (None when no countdown is active). A fresh signal.pending stops
        # the prior countdown before starting its own — see
        # _start_veto_countdown.
        self._active_veto_countdown_cancel: Optional[Any] = None
        self._active_veto_signal_id: str = ""

    def compose(self) -> ComposeResult:
        yield HeaderBar(id="header-bar")

        with TabbedContent(id="main-tabs"):
            with TabPane("◈ Overview", id="overview"):
                yield StalenessBanner(id="staleness-banner")
                yield StateStrip(id="state-strip")
                with Vertical(id="overview-container"):
                    with Horizontal(id="overview-top"):
                        with Vertical(classes="panel"):
                            yield Label("⟨ LIVE TRADES ⟩", classes="panel-title")
                            yield DataTable(id="trades-table")
                        with Vertical(classes="panel"):
                            yield Label("⟨ AGENT CONSENSUS ⟩", classes="panel-title")
                            yield AgentPanel(id="agent-panel")
                        with Vertical(id="mtf-panel"):
                            yield Label("⟨ MTF CONFLUENCE ⟩  H4│H1│M15", classes="panel-title")
                            yield MTFConfluencePanel(id="mtf-display")
                    with Horizontal(id="overview-bottom"):
                        with Vertical(classes="panel"):
                            yield Label("⟨ RISK DASHBOARD ⟩", classes="panel-title")
                            yield RiskPanel(id="risk-display")
                            yield Label("  NAV 30d", classes="status-dim")
                            yield Sparkline(data=[], id="nav-sparkline")
                        with Vertical(id="reflection-stream", classes="panel"):
                            yield Label(
                                "⟨ REFLECTION STREAM ⟩  [Claude self-improvement subprocess]",
                                classes="panel-title",
                            )
                            yield RichLog(
                                id="reflection-log",
                                highlight=True,
                                markup=True,
                                max_lines=200,
                                auto_scroll=True,
                            )
                        with Vertical(id="brain-stream"):
                            yield Label("⟨ BUDDY'S BRAIN ⟩  [stream of consciousness]",
                                       classes="panel-title")
                            # Tier 1 T4: transient phase indicator — tells the
                            # operator what the scanner is doing RIGHT NOW
                            # (scanning / gate-check / executing / idle). Sits
                            # above the cumulative counter and brain log so it's
                            # visible without scrolling.
                            yield PhaseIndicator(
                                state=self._phase_state,
                                id="phase-indicator",
                            )
                            # T3: cumulative work-unit counter (cycles · pairs · gates · trades).
                            # Shares ScanCounters instance with EmbeddedScanner.counters.
                            yield StatsBar(counters=self._counters, id="stats-bar")
                            yield RichLog(id="brain-log", highlight=True, markup=True,
                                         max_lines=500, auto_scroll=True)

            with TabPane("◈ Inbox", id="inbox"):
                yield InboxScreen(
                    project_root=str(PROJECT_ROOT),
                    id="inbox-screen",
                )
            with TabPane("◈ Trades", id="trades"):
                yield TradesScreen(id="trades-screen")
            with TabPane("◈ Agents", id="agents"):
                yield AgentsScreen(id="agents-screen")
            with TabPane("◈ Journal", id="journal"):
                yield JournalScreen(
                    project_root=str(PROJECT_ROOT),
                    id="journal-screen",
                )
            with TabPane("◈ Config", id="config"):
                yield ConfigScreen(
                    project_root=str(PROJECT_ROOT),
                    id="config-screen",
                )
            with TabPane("◈ Rules", id="rules"):
                yield RulesScreen(
                    project_root=str(PROJECT_ROOT),
                    id="rules-screen",
                )
            with TabPane("◈ Diagnostics", id="diag"):
                yield DiagnosticsScreen(
                    project_root=str(PROJECT_ROOT),
                    live=self._live,
                    id="diag-screen",
                )

        yield SystemHealthBar(id="system-health")
        yield LivenessBadge(
            claude_dir=Path(__file__).resolve().parent.parent.parent / ".claude",
            id="liveness-badge",
        )
        yield Footer()

    def on_mount(self) -> None:
        # Set up trades table columns
        table = self.query_one("#trades-table", DataTable)
        table.add_columns("Pair", "Dir", "P/L", "Entry", "SL", "TP", "Qty")

        # Boot brain stream
        self._write_boot_sequence()

        if self._live:
            self._connect_live()
            # Start embedded scanner (replaces need for separate main.py scan --watch)
            self._start_scanner()
            # Start the reflection stream reader (tails logs/reflection_log.jsonl)
            self._start_reflection_reader()
        else:
            self._init_demo()
            # In demo mode the reflection subprocess never spawns; tell the user so.
            try:
                rlog = self.query_one("#reflection-log", RichLog)
                rlog.write(Text.from_markup(
                    "[dim]  DEMO MODE — Claude reflections fire only in --live[/]"
                ))
            except Exception:
                pass

        # Start periodic refresh (both modes)
        self.set_interval(3.0, self._refresh_all)
        # Brain stream: fake lines in demo only; live uses real scanner output
        if not self._live:
            self.set_interval(0.8, self._tick_brain)
        # US-603: Layer-2 watchdog heartbeat (every 10s, atomic write).
        self._last_error_ts: str | None = None
        self._write_heartbeat_tick()
        self.set_interval(10.0, self._write_heartbeat_tick)
        # Mythos audit 2026-04-29 — Tier 7 step C: meta-pipeline events
        # to brain log. Tail .claude/meta/changes.jsonl since last seen
        # offset and emit one brain line per event so the operator sees
        # the cybernetic loop fire in real time. Cheap — file is
        # append-only, we track byte offset, no JSON re-parse for old
        # entries.
        self._meta_ledger_offset: int = self._initial_meta_ledger_offset()
        self.set_interval(1.0, self._tick_meta_events)
        # US-004: subscribe to event_bus signal.pending so the pre-trade
        # veto countdown banner fires automatically when ExecutionManager
        # publishes a pending signal. Replaces the dead-code path where
        # _start_veto_countdown was defined but never called.
        self._veto_subscription_worker()

    def on_unmount(self) -> None:
        """Signal background readers to exit cleanly."""
        try:
            self._reflection_stop.set()
        except Exception:
            pass

    def _initial_meta_ledger_offset(self) -> int:
        """Start tailing from the END of the existing ledger so we don't
        flood the brain log with hundreds of historical events on TUI
        boot. Future events get streamed live."""
        try:
            from pathlib import Path as _Path
            ledger = _Path(".claude/meta/changes.jsonl")
            if ledger.exists():
                return ledger.stat().st_size
        except Exception:
            pass
        return 0

    def _tick_meta_events(self) -> None:
        """Tail .claude/meta/changes.jsonl, emit one brain line per new
        event. Mythos Tier-7 step C — makes the cybernetic loop visible.
        """
        try:
            from pathlib import Path as _Path
            import json as _json
            ledger = _Path(".claude/meta/changes.jsonl")
            if not ledger.exists():
                return
            try:
                size = ledger.stat().st_size
            except OSError:
                return
            if size <= self._meta_ledger_offset:
                # File rotated or shrunk — restart from current end.
                if size < self._meta_ledger_offset:
                    self._meta_ledger_offset = size
                return
            with ledger.open("r", encoding="utf-8", errors="replace") as fh:
                fh.seek(self._meta_ledger_offset)
                new_text = fh.read()
                self._meta_ledger_offset = fh.tell()
            for line in new_text.splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    blob = _json.loads(line)
                except _json.JSONDecodeError:
                    continue
                if not isinstance(blob, dict):
                    continue
                self._emit_meta_brain_line(blob)
        except Exception as e:
            # Brain-log surfacing is observability — never crash the TUI on it
            logger.debug("_tick_meta_events: %s", e)

    def _emit_meta_brain_line(self, blob: dict) -> None:
        """Format a meta ledger entry into a single colored brain line."""
        msg = format_meta_brain_line(blob)
        try:
            self._write_brain(msg)
        except Exception:
            pass

    def _write_heartbeat_tick(self) -> None:
        """US-603: emit .claude/heartbeat.json for the external watchdog daemon."""
        try:
            repo_root = PROJECT_ROOT
            cycle = 0
            scanner_alive = False
            try:
                snap = self._provider.get_snapshot() if self._provider else None
                if snap is not None:
                    cycle = int(getattr(snap, "scan_cycle_count", 0) or 0)
            except Exception:
                snap = None
                pass
            if self._live:
                scanner = getattr(self, "_scanner", None)
                scanner_alive = _scanner_alive_from_sources(snap, scanner)
            write_heartbeat(
                repo_root,
                cycle_count=cycle,
                mode="live" if self._live else "demo",
                scanner_alive=scanner_alive,
                last_error_ts=self._last_error_ts,
            )
        except Exception as exc:  # heartbeat must never crash the TUI
            self._last_error_ts = datetime.now(timezone.utc).isoformat()
            logging.getLogger(__name__).debug("heartbeat write failed: %s", exc)

    def _write_boot_sequence(self) -> None:
        log = self.query_one("#brain-log", RichLog)
        header = self.query_one("#header-bar", HeaderBar)
        now = datetime.now(timezone.utc)
        mode = "LIVE" if self._live else "DEMO"
        header.mode_label = mode

        msgs = [
            f"[bold #ff2bd6]◈ BUDDY COMMAND BRIDGE — {mode} MODE[/]",
            f"[dim]  Session started {now.strftime('%Y-%m-%d %H:%M:%S UTC')}[/]",
        ]
        if self._live:
            msgs.append("[cyan]▸ Connecting to OANDA...[/]")
        else:
            msgs += [
                "[dim]  Simulated data — run with --live for real OANDA feed[/]",
                "[dim]  Loading demo models: TCN ✓ | Ridge ✓ | RF ✓[/]",
                "[cyan]▸ Starting demo scan cycle...[/]",
                "",
            ]
        for m in msgs:
            log.write(Text.from_markup(m))

    def _write_brain(self, markup: str) -> None:
        """Safely write to brain log (must be called on main thread).

        Mythos audit 2026-04-30 — also tees to .claude/brain/feed.jsonl
        as plain-text JSONL so verification can read what the operator
        sees in the F1 brain feed. Pre-fix the brain feed was in-memory
        only — auditing whether the cybernetic loop was firing required
        disk-artifact archaeology (state.json, meta package files,
        reflection_log.jsonl) instead of just `tail -f` on the actual
        feed. Now both surfaces show the same content.

        Disk write is best-effort — failures are swallowed so a degraded
        filesystem can never break the live TUI rendering.
        """
        self.query_one("#brain-log", RichLog).write(Text.from_markup(markup))
        try:
            _tee_brain_to_disk(markup)
        except Exception:  # noqa: BLE001 — observability is never load-bearing
            pass

    @work(thread=True)
    def _connect_live(self) -> None:
        """Connect to OANDA in a background thread.

        NOTE: query_one must NOT be called from this thread.
        All DOM access goes through call_from_thread.
        """
        ok = self._provider.connect()
        if ok:
            snap = self._provider.refresh()
            self.call_from_thread(self._apply_snapshot, snap)
            self.call_from_thread(
                self._write_brain,
                f"[green]✓ OANDA connected — NAV ${snap.nav:,.2f}[/]",
            )
            self.call_from_thread(
                self._write_brain,
                f"[dim]  Open trades: {len(snap.trades)}[/]",
            )
        else:
            self.call_from_thread(
                self._write_brain,
                "[yellow]▸ OANDA not available — falling back to cached data[/]",
            )
            # Still load agent weights and journal data
            snap = self._provider.refresh()
            self.call_from_thread(self._apply_snapshot, snap)

    def _start_scanner(self) -> None:
        """Initialize and schedule the embedded scanner (live mode only).

        Creates the EmbeddedScanner, kicks off initialization in a
        background thread (heavy imports), then schedules scan cycles:
        - First scan: 10s after boot (let OANDA connect first)
        - Subsequent scans: every 5 minutes
        """
        from src.tui.embedded_scanner import EmbeddedScanner

        project_root = str(PROJECT_ROOT)
        self._scanner = EmbeddedScanner(
            project_root=project_root,
            brain_callback=self._scanner_brain_bridge,
            auto_execute=True,
            interval_minutes=5,
            counters=self._counters,  # T3: shared with F1 StatsBar
        )
        # Tier 1 T4: adopt the App's PhaseState so the F1 PhaseIndicator
        # (mounted in compose()) and the scanner thread share the same
        # mutable instance. Otherwise the widget polls the App's
        # default-idle state forever while the scanner mutates its own.
        self._scanner.phase_state = self._phase_state
        self._init_scanner_worker()

    def _scanner_brain_bridge(self, markup: str) -> None:
        """Thread-safe bridge: scanner thread → main thread → brain log.

        The EmbeddedScanner calls this from its worker thread.
        We use call_from_thread to safely write to the Textual DOM.
        """
        try:
            self.call_from_thread(self._write_brain, markup)
        except Exception:
            # App might be shutting down — swallow safely
            pass

    def _reflection_bridge(self, markup: str) -> None:
        """Thread-safe bridge: reflection reader thread → main thread → reflection log."""
        try:
            self.call_from_thread(self._write_reflection, markup)
        except Exception:
            pass

    def _write_reflection(self, markup: str) -> None:
        """Safely write to reflection log (must be called on main thread)."""
        try:
            log = self.query_one("#reflection-log", RichLog)
            log.write(Text.from_markup(markup))
        except Exception:
            pass

    @work(thread=True)
    def _start_reflection_reader(self) -> None:
        """Background worker: tails logs/reflection_log.jsonl and streams to TUI."""
        project_root = PROJECT_ROOT
        log_path = project_root / "logs" / "reflection_log.jsonl"
        reader = ReflectionLogReader(
            log_path=log_path,
            callback=self._reflection_bridge,
            stop_flag=self._reflection_stop,
            poll_interval=2.0,
        )
        reader.run()

    @work(thread=True)
    def _init_scanner_worker(self) -> None:
        """Initialize scanner in background thread (heavy imports)."""
        ok = self._scanner.initialize()
        if ok:
            # Schedule scan cycles on the main thread
            self.call_from_thread(self._schedule_scan_timer)
        else:
            self.call_from_thread(
                self._write_brain,
                "[yellow]▸ Scanner init failed — running in display-only mode[/]",
            )

    def _schedule_scan_timer(self) -> None:
        """Schedule scan cycle timers (must be called on main thread)."""
        try:
            snap = self._provider.refresh()
            if self._scanner is not None and self._scanner.is_ready:
                snap.scanner_ready = True
                snap.scan_cycle_count = max(
                    int(getattr(snap, "scan_cycle_count", 0) or 0),
                    int(getattr(self._scanner, "_scan_count", 0) or 0),
                )
            self._apply_snapshot(snap)
        except Exception as exc:
            logger.debug("_schedule_scan_timer readiness refresh failed: %s", exc)
        # First scan after 10s (let OANDA connect + models load)
        self.set_timer(10.0, self._trigger_scan_cycle)
        # Subsequent scans every 5 minutes (300s)
        self.set_interval(300.0, self._trigger_scan_cycle)

    def _trigger_scan_cycle(self) -> None:
        """Timer callback — triggers a scan cycle in a background worker."""
        if self._scanner is not None and self._scanner.is_ready:
            self._do_scan_cycle()

    @work(thread=True, exclusive=True)
    def _do_scan_cycle(self) -> None:
        """Run one scan cycle in a background thread.

        exclusive=True ensures only one scan runs at a time.
        If a scan takes >5 minutes, the next timer fire is ignored
        rather than stacking up workers.
        """
        enrichment = self._scanner.run_one_cycle()
        if enrichment is not None:
            # Store enrichment — next _do_live_refresh (every 3s) picks it up
            self._provider.apply_scan_enrichment(enrichment)

    def _init_demo(self) -> None:
        """Initialize demo mode with fake data."""
        header = self.query_one("#header-bar", HeaderBar)
        header.nav = self._demo_nav
        header.pnl = self._demo_nav - 100000
        header.open_count = 3

        table = self.query_one("#trades-table", DataTable)
        table.add_rows([
            ("EUR/USD", "BUY", "+$124", "1.08542", "1.08320", "1.08890", "5000"),
            ("GBP/JPY", "SELL", "-$41", "191.320", "191.650", "190.800", "3000"),
            ("USD/CAD", "BUY", "+$87", "1.37210", "1.36980", "1.37650", "4000"),
        ])

        # Seed sparkline with demo data
        spark = self.query_one("#nav-sparkline", Sparkline)
        spark.data = [100000 + random.randint(-500, 2000) for _ in range(30)]

        # Try to load real agent weights even in demo mode
        snap = self._provider.refresh()
        if snap.agents:
            self.query_one("#agent-panel", AgentPanel).update_from_snapshot(snap)
            self.query_one("#agent-panel", AgentPanel).refresh()

    def _apply_snapshot(self, snap: DashboardSnapshot) -> None:
        """Apply a DashboardSnapshot to all widgets (runs on main thread)."""
        header = self.query_one("#header-bar", HeaderBar)
        fallback_nav = self._demo_nav if not self._live else 0.0
        header.nav = snap.nav if snap.nav > 0 else fallback_nav
        header.pnl = snap.unrealized_pnl if snap.nav > 0 else (
            self._demo_nav - 100000 if not self._live else 0.0
        )
        header.open_count = len(snap.trades)
        header.oanda_ok = snap.oanda_connected
        header.scanner_ok = snap.scanner_ready

        # Update trades table
        table = self.query_one("#trades-table", DataTable)
        table.clear()
        if snap.trades:
            for tr in snap.trades:
                pnl_str = f"+${tr.pnl_pips:,.0f}" if tr.pnl_pips >= 0 else f"-${abs(tr.pnl_pips):,.0f}"
                table.add_row(
                    tr.pair, tr.direction, pnl_str,
                    f"{tr.entry_price:.5f}",
                    f"{tr.sl_price:.5f}" if tr.sl_price else "—",
                    f"{tr.tp_price:.5f}" if tr.tp_price else "—",
                    f"{tr.quantity:.0f}",
                )

        # Update agent panel
        agent_panel = self.query_one("#agent-panel", AgentPanel)
        agent_panel.update_from_snapshot(snap)
        agent_panel.refresh()

        # Update risk panel
        risk_panel = self.query_one("#risk-display", RiskPanel)
        risk_panel.update_from_snapshot(snap)
        risk_panel.refresh()

        # Update MTF panel
        mtf_panel = self.query_one("#mtf-display", MTFConfluencePanel)
        mtf_panel.update_from_snapshot(snap)
        mtf_panel.refresh()

        # Update NAV sparkline
        if snap.nav_history:
            spark = self.query_one("#nav-sparkline", Sparkline)
            spark.data = snap.nav_history

        # Update system health bar
        health = self.query_one("#system-health", SystemHealthBar)
        health.update_from_snapshot(snap)
        health.refresh()

        # Update Inbox screen (F2)
        try:
            inbox_screen = self.query_one("#inbox-screen", InboxScreen)
            inbox_screen.update_from_snapshot(snap)
        except Exception:
            pass

        # Update Trades screen (F3)
        try:
            trades_screen = self.query_one("#trades-screen", TradesScreen)
            trades_screen.update_from_snapshot(snap)
        except Exception:
            pass

        # Update Agents screen (F3)
        try:
            agents_screen = self.query_one("#agents-screen", AgentsScreen)
            agents_screen.update_from_snapshot(snap)
        except Exception:
            pass

        # Update Diagnostics screen (F7)
        try:
            diag_screen = self.query_one("#diag-screen", DiagnosticsScreen)
            diag_screen.update_from_snapshot(snap)
        except Exception:
            pass

        # Update Journal screen (F5) — mtime-guarded auto-refresh of
        # trade_journal_rl.json + learnings.md so closed trades land in
        # the table without requiring a TUI restart.
        try:
            journal_screen = self.query_one("#journal-screen", JournalScreen)
            journal_screen.update_from_snapshot(snap)
        except Exception:
            pass

        # Update state strip
        try:
            state_strip = self.query_one("#state-strip", StateStrip)
            state_strip.update_from_snapshot(snap)
        except Exception:
            pass

        # US-513: Update staleness banner (red alert when models > 7d old)
        try:
            banner = self.query_one("#staleness-banner", StalenessBanner)
            banner.update_from_snapshot(snap)
        except Exception:
            pass

    @work(thread=True)
    def _do_live_refresh(self) -> None:
        """Run data refresh in background thread."""
        snap = self._provider.refresh()
        scanner = getattr(self, "_scanner", None)
        if scanner is not None:
            snap.scanner_ready = bool(getattr(scanner, "is_ready", False))
            snap.scan_cycle_count = max(
                int(getattr(snap, "scan_cycle_count", 0) or 0),
                int(getattr(scanner, "_scan_count", 0) or 0),
            )
        self.call_from_thread(self._apply_snapshot, snap)

    def _refresh_all(self) -> None:
        """Periodic refresh — delegates to Worker for live, simulates for demo."""
        if self._live:
            self._do_live_refresh()
        else:
            self._refresh_demo()

    def _refresh_demo(self) -> None:
        """Simulate data movement in demo mode."""
        header = self.query_one("#header-bar", HeaderBar)
        self._demo_nav += random.uniform(-50, 80)
        header.nav = self._demo_nav
        header.pnl = self._demo_nav - 100000

        # Jitter agent scores in demo
        agents = self.query_one("#agent-panel", AgentPanel)
        jittered = []
        for name, score, signal in agents._agents:
            s = max(0.0, min(1.0, score + random.uniform(-0.03, 0.03)))
            jittered.append((name, s, signal))
        agents._agents = jittered
        agents.refresh()

        # Jitter risk
        risk = self.query_one("#risk-display", RiskPanel)
        risk._risk = max(0, risk._risk + random.uniform(-0.3, 0.3)) or 8.2
        risk._dd = max(0, risk._dd + random.uniform(-0.2, 0.2)) or 2.1
        risk._max_dd = max(risk._dd, risk._max_dd or 3.4)
        risk.refresh()

        # Refresh MTF sparklines
        self.query_one("#mtf-display", MTFConfluencePanel).refresh()
        self.query_one("#system-health", SystemHealthBar).refresh()

        # Update sparkline
        spark = self.query_one("#nav-sparkline", Sparkline)
        current = list(spark.data or [])
        current.append(self._demo_nav)
        if len(current) > 30:
            current = current[-30:]
        spark.data = current

    def _tick_brain(self) -> None:
        """Add a line to the brain stream.

        Plan C (2026-04-29): prefer real meta-pipeline events from the
        changes.jsonl ledger.  Each new JSONL line emits one brain-log line:
            🧠 HH:MM [{change_id[:8]}] {kind} → {stage}
        Falls back to _fake_brain_line() when the ledger has no new entries
        so the display never goes dead between orchestrator cycles.
        """
        try:
            log = self.query_one("#brain-log", RichLog)
        except Exception:
            return

        _LEDGER = (
            Path(__file__).resolve().parent.parent.parent
            / ".claude" / "meta" / "changes.jsonl"
        )
        emitted = False
        if _LEDGER.exists():
            try:
                with _LEDGER.open("r", encoding="utf-8") as fh:
                    fh.seek(self._brain_ledger_pos)
                    for raw in fh:
                        raw = raw.strip()
                        if not raw:
                            continue
                        try:
                            entry = json.loads(raw)
                        except json.JSONDecodeError:
                            continue
                        cid = str(entry.get("change_id", "?"))[:8]
                        kind = str(entry.get("kind", "?"))
                        stage = str(entry.get("stage", "?"))
                        ts = (
                            str(entry.get("ts") or entry.get("created_at") or "")
                            [:16].replace("T", " ")
                        )
                        markup = (
                            f"[bold cyan]🧠[/] [dim]{ts}[/] "
                            f"[[yellow]{cid}[/]] {kind} → [green]{stage}[/]"
                        )
                        log.write(Text.from_markup(markup))
                        emitted = True
                    self._brain_ledger_pos = fh.tell()
            except OSError:
                pass

        if not emitted:
            now = datetime.now(timezone.utc).strftime("%H:%M:%S")
            log.write(Text.from_markup(f"[dim]{now}[/] {_fake_brain_line()}"))

    def action_switch_tab(self, tab_id: str) -> None:
        if self._kill_in_progress:
            return
        # T1 (Tier 1): F9 "Jobs" pushes a modal Screen rather than switching a
        # tab pane. JobsScreen owns its own pause/resume/trigger actions and
        # pops back to whatever tab was active when the operator pressed F9.
        if tab_id == "jobs":
            self._open_jobs_screen()
            return
        tabs = self.query_one("#main-tabs", TabbedContent)
        tabs.active = tab_id

    def _open_jobs_screen(self) -> None:
        """Lazily build a ScheduledJobsRegistry and push the Jobs screen.

        We don't keep a persistent reference on the app — the TUI runtime
        path uses EmbeddedScanner, not Orchestrator, so the registry isn't
        cached anywhere else. Building it here on demand keeps the screen
        cheap (load() is one safe_json_read of jobs.json + state.json) and
        avoids coupling the EmbeddedScanner lifecycle to the screen.
        """
        try:
            from src.tui.screens.jobs_screen import JobsScreen
            from src.scanner.automation.scheduled_jobs import ScheduledJobsRegistry
        except ImportError as e:
            logger.error("F9 Jobs screen: import failed: %s", e)
            return
        try:
            registry = ScheduledJobsRegistry()
            registry.load()
        except (OSError, ValueError) as e:
            logger.error("F9 Jobs screen: registry load failed: %s", e)
            return
        self.push_screen(JobsScreen(registry=registry))

    def action_open_command_palette(self) -> None:
        """`:` opens the Vim-style palette over any screen.

        Modal returns the chosen command id; we route it here so the modal
        stays decoupled from the rest of the app surface.
        """
        if self._kill_in_progress:
            return

        def _route(cmd_id: str | None) -> None:
            if cmd_id is None:
                return
            # Navigation maps to switch_tab(<tab_id>).
            if cmd_id.startswith("nav."):
                self.action_switch_tab(cmd_id.split(".", 1)[1])
                return
            # Everything else maps 1:1 to an action_* method.
            mapping = {
                "supervisor.pause": self.action_supervisor_pause,
                "supervisor.kill": self.action_supervisor_kill,
                "supervisor.mode": self.action_supervisor_mode,
                "supervisor.abort": self.action_supervisor_abort,
                "broker.cycle": self.action_cycle_asset_class,
                "app.quit": self.action_quit,
                "log.view": self.action_open_log_viewer,
            }
            handler = mapping.get(cmd_id)
            if handler is not None:
                handler()
            else:
                logger.warning("CommandPalette: unknown command id %s", cmd_id)

        self.push_screen(CommandPaletteModal(), _route)

    def action_open_log_viewer(self) -> None:
        """Ctrl+L (or command palette → 'Live logs'): pop the log-tail modal."""
        if self._kill_in_progress:
            return
        repo_root = Path(__file__).resolve().parents[2]
        self.push_screen(LogViewerModal(repo_root))

    def action_edit_briefing(self) -> None:
        """Ctrl+E: pop the brain section editor for ``.claude/brain/briefing.md``.

        Tier 2 T7. Lazy-imports the screen module to keep app startup lean —
        the editor only loads when the operator actually invokes it.
        """
        if self._kill_in_progress:
            return
        from src.tui.screens.brain_editor_screen import BrainEditorScreen
        self.push_screen(BrainEditorScreen())

    def action_cycle_asset_class(self) -> None:
        """F7: cycle through FX → Futures → Hybrid → FX.  Blocked during kill.

        Updates the scanner config + restarts the scan loop with the new
        instrument set. The broker switches automatically:
          fx     → OANDA (EUR_USD, GBP_USD, ...)
          futures → IBKR  (ES, NQ, CL, GC, ZB, 6E)
          hybrid  → both  (all instruments, primary broker from config)
        """
        if self._kill_in_progress:
            return
        idx = self._ASSET_CLASSES.index(self._asset_class)
        self._asset_class = self._ASSET_CLASSES[(idx + 1) % len(self._ASSET_CLASSES)]

        mode_label = {
            "fx": "FX (OANDA) — 15 pairs",
            "futures": "FUTURES (IBKR) — ES NQ CL GC ZB 6E",
            "hybrid": "HYBRID — FX + Futures",
        }[self._asset_class]

        # Update scanner config if scanner is running
        if self._scanner is not None:
            try:
                config = self._scanner.get_config()
                if config is not None:
                    config.asset_class = self._asset_class
                    config.broker_type = config.active_broker_type
                    # Log the switch
                    import logging
                    logging.getLogger(__name__).warning(
                        "Asset class switched to %s — instruments: %s",
                        self._asset_class, config.active_instruments[:5],
                    )
            except Exception:
                pass

        # Notify the user via the brain log
        try:
            brain_log = self.query_one("#brain-log")
            brain_log.write(f"[bold cyan]⟨ MODE ⟩[/] {mode_label}")
        except Exception:
            pass

        self.notify(f"Asset class: {mode_label}", title="Mode Switch")

    def action_supervisor_pause(self) -> None:
        """Space: toggle scanner_paused via StateEngine (US-506)."""
        if self._kill_in_progress:
            return

        from src.scanner.automation.state_engine import StateEngine
        from src.scanner.automation.event_bus import get_event_bus

        se = StateEngine()

        # Guard: halted takes precedence — no pause toggle when system is halted
        if se.get_halted():
            self.notify("System is HALTED — resume not available.", title="Paused Blocked", severity="warning")
            return

        current = se.get_paused()
        new_paused = not current
        se.set_paused(new_paused)

        action = "pause" if new_paused else "resume"
        event_type = "control.pause" if new_paused else "control.resume"
        try:
            get_event_bus().publish(event_type, {"source": "TUI", "actor": "operator"})
        except Exception as _eb_err:
            logger.warning("action_supervisor_pause: event_bus publish failed: %s", _eb_err)

        # Brain log notification
        if new_paused:
            markup = "[bold yellow]◈ SCANNER PAUSED by operator[/bold yellow] — new executions suppressed, monitoring active"
        else:
            markup = "[bold green]◈ SCANNER RESUMED by operator[/bold green] — new executions enabled"
        try:
            from rich.text import Text as _Text
            self.query_one("#brain-log", RichLog).write(_Text.from_markup(markup))
        except Exception:
            pass

        # strategic_log.md entry
        try:
            from datetime import datetime as _dt, timezone as _tz
            from pathlib import Path as _Path
            _ts = _dt.now(_tz.utc).isoformat()
            _log_path = _Path(".claude/brain/strategic_log.md")
            if _log_path.exists():
                _entry = (
                    f"\n### {_ts} — Operator {action.upper()}\n"
                    f"**actor:** TUI  **action:** {action}  "
                    f"**before:** {'paused' if current else 'running'}  "
                    f"**after:** {'paused' if new_paused else 'running'}\n"
                )
                with open(_log_path, "a") as _f:
                    _f.write(_entry)
        except Exception as _sl_err:
            logger.debug("strategic_log write failed: %s", _sl_err)

        label = "PAUSED" if new_paused else "RUNNING"
        self.notify(f"Scanner {label}", title="Supervisor", severity="warning" if new_paused else "information")
        logger.info("action_supervisor_pause: scanner_paused=%s", new_paused)

    def action_supervisor_kill(self) -> None:
        """K: open kill-switch confirmation modal (US-505)."""
        if self._kill_in_progress:
            return

        snap = self._provider.snapshot

        if not snap.trades:
            self.notify("Nothing to flatten.", title="Kill Switch", severity="information")
            return

        def _on_modal_result(confirmed: bool) -> None:
            if confirmed:
                self._kill_in_progress = True
                self._do_flatten_worker()

        self.push_screen(KillModal(snap), _on_modal_result)

    @work
    async def _do_flatten_worker(self) -> None:
        """Async worker: invoke ExecutionManager.flatten_all and stream bus events to brain log."""
        import asyncio as _asyncio

        from rich.text import Text as _Text

        from src.scanner.automation.event_bus import get_event_bus
        from src.scanner.execution import (
            ExecutionManager,
            FlattenResult,
            KillSwitchPartialFailure,
        )

        def _brain(markup: str) -> None:
            try:
                self.query_one("#brain-log", RichLog).write(_Text.from_markup(markup))
            except Exception:
                pass

        _brain("[bold red]◈ KILL SWITCH — flattening all positions...[/]")

        bus = get_event_bus()
        em = ExecutionManager()
        _done = _asyncio.Event()

        async def _stream_bus() -> None:
            try:
                async for event in bus.subscribe():
                    if _done.is_set():
                        break
                    etype = event.get("type", "")
                    if etype == "signal.rejected" and event.get("reason") == "flattened":
                        tid = event.get("trade_id", "?")
                        _brain(f"[red]  ✗ trade {tid} flattened[/]")
                    elif etype == "control.kill":
                        _brain("[bold red]  ◈ control.kill — scanner halted[/]")
                        _done.set()
                        break
            except _asyncio.CancelledError:
                pass
            except Exception as _e:
                logger.debug("kill bus stream error: %s", _e)

        listener = _asyncio.create_task(_stream_bus())

        try:
            result: FlattenResult = await em.flatten_all("operator_kill")
            _done.set()
            listener.cancel()
            _brain(
                f"[green]✓ Flatten complete — {result.positions_closed} positions closed, "
                f"{result.orders_cancelled} orders cancelled ({result.duration_ms:.0f}ms)[/]"
            )
            self.notify("✓ All positions flat. Scanner halted.", title="Kill Switch")
        except KillSwitchPartialFailure as exc:
            _done.set()
            listener.cancel()
            remaining = ", ".join(exc.remaining_trades[:5])
            _brain(f"[bold red]  ✗ KillSwitchPartialFailure: {exc}[/]")
            self.notify(
                f"Partial failure — still open: {remaining}",
                title="Kill Switch Failed",
                severity="error",
            )
        except Exception as exc:
            _done.set()
            listener.cancel()
            _brain(f"[red]  ✗ flatten_all error: {exc}[/]")
            self.notify(f"Kill switch error: {exc}", title="Kill Switch Error", severity="error")
        finally:
            self._kill_in_progress = False
            if self._live:
                self._do_live_refresh()

    def action_supervisor_mode(self) -> None:
        """M: toggle dry-run ↔ live mode (US-507).

        live → dry_run: immediate (no confirmation needed).
        dry_run → live: credential precheck, then typed 'LIVE' confirmation modal.
        """
        if self._kill_in_progress:
            return

        from src.scanner.automation.state_engine import StateEngine

        se = StateEngine()
        current = se.get_mode()
        snap = self._provider.snapshot

        if current == "live":
            self._apply_mode_change(from_mode="live", to_mode="dry_run")
            return

        # dry_run → live: credential precheck before showing modal
        cred_ok, cred_msg = check_oanda_credentials()
        if not cred_ok:
            self.notify(
                f"Live mode blocked: {cred_msg}",
                title="Credentials Missing",
                severity="error",
            )
            logger.warning("action_supervisor_mode: credential check failed — %s", cred_msg)
            return

        def _on_modal_result(confirmed: bool) -> None:
            if confirmed:
                self._apply_mode_change(from_mode="dry_run", to_mode="live")

        self.push_screen(
            ModeConfirmModal(
                nav=snap.nav if snap.nav > 0 else self._demo_nav,
                open_trades=len(snap.trades),
            ),
            _on_modal_result,
        )

    def _apply_mode_change(self, from_mode: str, to_mode: str) -> None:
        """Persist mode, update state strip, emit event, append strategic_log.md."""
        from datetime import datetime as _dt, timezone as _tz
        from pathlib import Path as _Path

        from src.scanner.automation.event_bus import get_event_bus
        from src.scanner.automation.state_engine import StateEngine

        se = StateEngine()
        try:
            se.set_mode(to_mode)
        except Exception as _e:
            logger.error("_apply_mode_change: StateEngine.set_mode failed: %s", _e)
            self.notify(f"Mode change failed: {_e}", title="Error", severity="error")
            return

        # Immediate reactive update — don't wait for the next 3s refresh cycle
        try:
            state_strip = self.query_one("#state-strip", StateStrip)
            state_strip.mode = "LIVE" if to_mode == "live" else "DRY-RUN"
            state_strip.refresh()
        except Exception:
            pass

        snap = self._provider.snapshot
        nav = snap.nav if snap.nav > 0 else self._demo_nav
        open_trade_count = len(snap.trades)

        # Event bus: control.mode_change
        try:
            get_event_bus().publish("control.mode_change", {
                "from": from_mode,
                "to": to_mode,
                "actor": "operator",
                "nav": nav,
                "open_trade_count": open_trade_count,
            })
        except Exception as _eb_err:
            logger.warning("_apply_mode_change: event_bus publish failed: %s", _eb_err)

        # strategic_log.md entry
        try:
            _ts = _dt.now(_tz.utc).isoformat()
            _log_path = _Path(".claude/brain/strategic_log.md")
            if _log_path.exists():
                _entry = (
                    f"\n### {_ts} — Mode Change {from_mode.upper()} → {to_mode.upper()}\n"
                    f"**actor:** TUI operator  **from:** {from_mode}  **to:** {to_mode}  "
                    f"**nav:** ${nav:,.2f}  **open_trades:** {open_trade_count}\n"
                )
                with open(_log_path, "a") as _f:
                    _f.write(_entry)
        except Exception as _sl_err:
            logger.debug("_apply_mode_change: strategic_log write failed: %s", _sl_err)

        # Brain log
        if to_mode == "live":
            markup = "[bold magenta]◈ MODE → LIVE[/bold magenta] — real execution enabled, capital at risk"
        else:
            markup = "[bold cyan]◈ MODE → DRY-RUN[/bold cyan] — execution disabled, monitoring only"
        try:
            self.query_one("#brain-log", RichLog).write(Text.from_markup(markup))
        except Exception:
            pass

        label = "LIVE" if to_mode == "live" else "DRY-RUN"
        sev = "warning" if to_mode == "live" else "information"
        self.notify(f"Mode: {label}", title="Mode Switch", severity=sev)
        logger.info("_apply_mode_change: %s → %s (nav=%.2f, open=%d)", from_mode, to_mode, nav, open_trade_count)

    def _active_tab_id(self) -> str:
        """US-005: Return the id of the currently-active TabPane (or '').

        Used by the global a/c/r hotkey handlers to early-return when the
        focused tab owns a screen-local binding for the same key. Falls
        back to '' if the TabbedContent has not yet been mounted (e.g.
        during app boot before compose finishes).
        """
        try:
            tabs = self.query_one("#main-tabs", TabbedContent)
        except Exception:  # noqa: BLE001 — pre-mount lookup, treat as no tab
            return ""
        return str(getattr(tabs, "active", "") or "")

    def _emit_hotkey_routed(self, key: str, tab: str) -> None:
        """US-005: Single brain-feed line per AC-4.

        Format ``hotkey {key} routed to {tab} screen handler`` lands in
        both ``#brain-log`` (operator surface) and ``.claude/brain/feed.jsonl``
        via the existing ``_write_brain`` tee. Best-effort — observability
        is never load-bearing on the routing decision.
        """
        try:
            self._write_brain(f"hotkey {key} routed to {tab} screen handler")
        except Exception:  # noqa: BLE001 — observability is never load-bearing
            pass

    def action_supervisor_abort(self) -> None:
        """A: US-511 veto pending signal via AbortConfirmModal (US-003).

        Reads the most recent signal.pending from the replay buffer, shows the
        operator a confirmation modal with pair/direction/age, and only on
        confirm emits control.signal_veto + appends to strategic_log.md.

        US-005: defers to F2 Inbox's screen-local ``a`` (Approve) binding
        when the inbox tab is active — early-returns BEFORE event_bus
        lookup so no ``control.signal_veto`` is ever published while the
        operator is reviewing adjustments.
        """
        if self._active_tab_id() == "inbox":
            self._emit_hotkey_routed("a", "inbox")
            return
        if self._kill_in_progress:
            return
        logger.info("hotkey supervisor_abort pressed")

        try:
            from src.scanner.automation.event_bus import get_event_bus
            _bus = get_event_bus()
        except Exception as _e:
            logger.warning("action_supervisor_abort: event_bus unavailable: %s", _e)
            self.notify("Abort unavailable (event bus error)", severity="warning")
            return

        _pending_event: Optional[dict] = None
        for _evt in reversed(_bus.replay_buffer()):
            if _evt.get("event_type") == "signal.pending":
                _pending_event = _evt
                break

        def _on_modal_result(confirmed: Optional[bool]) -> None:
            # Operator dismissed / cancelled / empty-state → no-op.
            if not confirmed or _pending_event is None:
                logger.info(
                    "action_supervisor_abort: dismissed without veto (confirmed=%s, has_pending=%s)",
                    confirmed, _pending_event is not None,
                )
                return
            _emit_signal_veto(self, _bus, _pending_event)

        self.push_screen(AbortConfirmModal(_pending_event), _on_modal_result)

    @work
    async def _veto_subscription_worker(self) -> None:
        """US-004: Async worker subscribing to event_bus signal.pending events.

        For each ``signal.pending``, invokes ``_start_veto_countdown`` on the
        TUI thread with duration sourced from ``ScannerConfig.pre_trade_veto_seconds``
        (default 5). Other event types are ignored. A second ``signal.pending``
        before the prior countdown ends cancels the prior cleanly — see the
        zombie-timer rule in ``_start_veto_countdown``.

        The worker runs inside Textual's asyncio loop (see ``@work``); the
        bus's ``subscribe()`` async generator naturally yields events as the
        publisher thread schedules them via ``call_soon_threadsafe``.
        """
        import asyncio as _asyncio

        try:
            from src.scanner.automation.event_bus import get_event_bus
            bus = get_event_bus()
        except Exception as e:
            logger.warning("_veto_subscription_worker: bus unavailable: %s", e)
            return

        def _resolve_duration() -> int:
            try:
                from src.scanner.config import ScannerConfig as _SC
                cfg = _SC()
            except Exception:
                cfg = None
            return int(getattr(cfg, "pre_trade_veto_seconds", 5))

        try:
            async for event in bus.subscribe():
                if event.get("event_type") != "signal.pending":
                    continue
                payload = event.get("payload", {}) or {}
                pair = str(payload.get("pair", "?"))
                direction = str(payload.get("direction", "?"))
                signal_id = str(payload.get("signal_id", ""))
                duration_s = _resolve_duration()
                try:
                    self._start_veto_countdown(pair, direction, duration_s, signal_id)
                except Exception as e:
                    logger.debug("_start_veto_countdown dispatch failed: %s", e)
        except _asyncio.CancelledError:
            pass
        except Exception as e:
            logger.debug("_veto_subscription_worker stream error: %s", e)

    def _start_veto_countdown(
        self,
        pair: str,
        direction: str,
        duration_s: int,
        signal_id: str = "",
    ) -> None:
        """US-004/US-511: Brain-log countdown 'EXECUTING {pair} {dir} in {n}s… A to abort'.

        Uses the existing #brain-log RichLog rather than a new pinned banner
        widget (design choice — keeps composition/TCSS untouched, mirrors
        every other transient TUI surface that streams through the brain
        log).

        Zombie-timer rule: if a prior countdown is still ticking when this
        is called (a second signal.pending preempted the first), the prior
        ``set_interval`` handle is stopped before a new one is created, so
        we never have two banners decrementing concurrently.
        """
        prior_cancel = getattr(self, "_active_veto_countdown_cancel", None)
        if prior_cancel is not None:
            try:
                prior_cancel()
            except Exception as _e:
                logger.debug("_start_veto_countdown: prior cancel raised: %s", _e)
        self._active_veto_countdown_cancel = None
        self._active_veto_signal_id = signal_id

        _remaining_s = [max(1, int(duration_s))]
        _cancel_holder: list = [None]

        def _stop_self() -> None:
            handle = _cancel_holder[0]
            if handle is not None:
                try:
                    handle()
                except Exception:
                    pass
            if getattr(self, "_active_veto_countdown_cancel", None) is _stop_self:
                self._active_veto_countdown_cancel = None
                self._active_veto_signal_id = ""

        def _tick() -> None:
            n = _remaining_s[0]
            if n <= 0:
                _stop_self()
                return
            try:
                arrow = "↑" if direction.upper() in ("LONG", "BUY") else (
                    "↓" if direction.upper() in ("SHORT", "SELL") else "•"
                )
                self.query_one("#brain-log", RichLog).write(
                    f"[bold yellow]⏳ EXECUTING {pair} {arrow}{direction} in {n}s… "
                    f"[dim]A to abort[/][/]"
                )
            except Exception:
                pass
            _remaining_s[0] -= 1

        _handle = self.set_interval(1.0, _tick, pause=False)
        _cancel_holder[0] = _handle.stop
        self._active_veto_countdown_cancel = _stop_self

    def action_copy_snapshot(self) -> None:
        """Copy state to file + clipboard. Focused box → just that box.

        Mythos audit 2026-05-04 — operator request:

        * No widget focused → full snapshot (every canonical surface).
        * A focusable box is in focus (click-to-focus, magenta border) →
          dump JUST that box's content. Operator can target a single
          panel without dragging the full 14KB into chat.

        Output lands in .claude/snapshots/tui_<ts>.md AND is piped to
        the system clipboard (pbcopy on macOS / xclip / xsel / wl-copy
        on Linux). Operator pastes the markdown directly into chat.

        US-005: defers to F3 Trades's screen-local ``c`` (Close Trade)
        binding when the trades tab is active — early-returns BEFORE
        any snapshot work so ``trades_screen.action_close_selected_trade``
        handles ``c`` unambiguously.
        """
        if self._active_tab_id() == "trades":
            self._emit_hotkey_routed("c", "trades")
            return
        try:
            from src.tui.snapshot import (
                build_snapshot,
                copy_to_clipboard,
                write_and_copy,
            )
        except Exception as e:
            try:
                self._write_brain(
                    f"[red]✗ snapshot import failed: {e}[/]"
                )
            except Exception:
                pass
            return

        focused = self.focused
        focused_id = getattr(focused, "id", None) if focused is not None else None

        try:
            if focused_id:
                content = self._dump_focused_widget(focused, focused_id)
                from datetime import datetime, timezone as _tz
                from pathlib import Path as _Path
                snap_dir = _Path(".claude/snapshots")
                snap_dir.mkdir(parents=True, exist_ok=True)
                ts = datetime.now(_tz.utc).strftime("%Y%m%dT%H%M%SZ")
                path = snap_dir / f"box_{focused_id}_{ts}.md"
                path.write_text(content, encoding="utf-8")
                tool = copy_to_clipboard(content)
                source_label = f"box [{focused_id}]"
            else:
                path, tool = write_and_copy()
                source_label = "full snapshot"
        except Exception as e:
            try:
                self._write_brain(f"[red]✗ snapshot failed: {e}[/]")
            except Exception:
                pass
            return

        try:
            rel = path.relative_to(Path.cwd()) if path.is_absolute() else path
        except (ValueError, OSError):
            rel = path

        msg = f"[bold cyan]📋 {source_label}: [/][green]{rel}[/]"
        if tool:
            msg += f" [dim](copied via {tool})[/]"
        else:
            msg += " [dim](no clipboard tool found — paste from file)[/]"
        try:
            self._write_brain(msg)
        except Exception:
            pass

    def _dump_focused_widget(self, widget: Any, widget_id: str) -> str:
        """Render a focused widget's content as markdown.

        For known widget IDs we map to dedicated snapshot sections (the
        operator gets the same structured info they'd get from the full
        snapshot, just for that one box). For unknown widgets we fall
        back to whatever the widget's `render()` produces — converted
        from Rich Text to plain string when possible.
        """
        from datetime import datetime, timezone as _tz
        ts = datetime.now(_tz.utc).isoformat(timespec="seconds")
        header = f"# Buddy TUI Box — `{widget_id}` — {ts}\n\n"

        # Map widget IDs to the snapshot module's section helpers so the
        # operator gets the canonical structured view of just that box.
        try:
            from src.tui import snapshot as _snap
        except Exception:
            _snap = None

        if _snap is not None:
            id_to_section = {
                "header-bar": _snap._heartbeat_section,
                "agent-panel": _snap._agents_section,
                "brain-stream": _snap._brain_feed_section,
                "reflection-log": _snap._reflection_log_section,
                "trades-table": _snap._trades_section,
                "diag-error-log": _snap._debug_log_section,
            }
            section_fn = id_to_section.get(widget_id)
            if section_fn is not None:
                return header + section_fn()

        # Fallback: render whatever the widget produces as plain text.
        try:
            renderable = widget.render()
        except Exception as e:
            return header + f"```\n(render failed: {e!r})\n```\n"

        try:
            from rich.text import Text as _RichText
            if isinstance(renderable, _RichText):
                body = renderable.plain
            else:
                body = str(renderable)
        except Exception:
            body = repr(renderable)

        return header + f"```\n{body}\n```\n"

    def action_refresh_rules(self) -> None:
        """US-005: Safety guard for any global ``r`` binding routed to this app.

        Defers to F2 Inbox's screen-local ``r`` (Reject) and F7 Rules's
        screen-local ``r`` (Refresh) when their tabs are active — early-returns
        BEFORE doing anything else, so the screen-local handler wins
        unambiguously even if the operator's focus has drifted off the screen
        widget (e.g. onto a Footer or the tab header) and the global handler
        would otherwise fire.

        Currently no global ``r`` Binding exists in ``BuddyApp.BINDINGS`` —
        this method exists so the early-return contract is in place if a
        future change wires one. Outside the two owning tabs the method is
        a deliberate no-op (logs nothing, performs no side effects).
        """
        tab_id = self._active_tab_id()
        if tab_id == "inbox":
            self._emit_hotkey_routed("r", "inbox")
            return
        if tab_id == "rules":
            self._emit_hotkey_routed("r", "rules")
            return

    def action_unhalt(self) -> None:
        """Health-gated resume from the halted state.

        Mythos audit 2026-05-01. The auto-halt protocol (eacb617) sets
        state.halted=True when consecutive_losses ≥ threshold and
        EmbeddedScanner.run_one_cycle (a561f2d) early-returns whenever
        StateEngine().get_halted() is true. Without an unhalt hotkey,
        the operator had to hand-edit `.claude/state.json` to resume —
        the brain message even pointed at 'k' which is supervisor_kill
        (flatten-all), not unhalt.

        This action now runs the same live-source checks the operator would
        perform manually before resuming. It never clears halted when OANDA,
        scanner readiness, heartbeat, model freshness, or open-position state
        are unsafe.
        """
        try:
            from src.scanner.automation.state_engine import StateEngine
        except Exception as e:
            try:
                self._write_brain(
                    f"[red]✗ unhalt failed: state_engine import — {e}[/]"
                )
            except Exception:
                pass
            return
        try:
            engine = StateEngine()
            was_halted = bool(engine.get_halted())
            if not was_halted:
                self._write_brain(
                    "[dim]◈ unhalt no-op — scanner was not halted.[/]"
                )
                return

            ok, reasons = self._evaluate_unhalt_health()
            if not ok:
                reason_text = "; ".join(reasons)
                self._write_brain(
                    f"[bold red]◈ UNHALT BLOCKED[/] — {reason_text}"
                )
                self._append_control_log(
                    title="Unhalt Blocked",
                    fields={
                        "actor": "TUI operator",
                        "action": "unhalt_blocked",
                        "reasons": reason_text,
                    },
                )
                self.notify(
                    reason_text,
                    title="Unhalt Blocked",
                    severity="error",
                )
                return

            engine.set_halted(False)
            try:
                engine.set_last_actor("TUI_UNHALT")
            except Exception:
                pass
            try:
                from src.scanner.automation.event_bus import get_event_bus
                get_event_bus().publish("control.resume", {
                    "source": "TUI",
                    "actor": "operator",
                    "reason": "health_gated_unhalt",
                })
            except Exception as _eb_err:
                logger.warning("action_unhalt: event_bus publish failed: %s", _eb_err)
            self._append_control_log(
                title="Unhalt Approved",
                fields={
                    "actor": "TUI operator",
                    "action": "unhalt",
                    "nav": f"${self._provider.snapshot.nav:,.2f}",
                    "open_trades": str(len(self._provider.snapshot.trades)),
                },
            )
            if was_halted:
                self._write_brain(
                    "[bold green]◈ UNHALT — scanner will resume on next cycle.[/]"
                )
                self.notify("Scanner resume armed", title="Unhalt", severity="information")
        except Exception as e:
            try:
                self._write_brain(
                    f"[red]✗ unhalt failed: {e}[/]"
                )
            except Exception:
                pass

    def _evaluate_unhalt_health(self) -> tuple[bool, list[str]]:
        """Return whether it is safe to clear the halted flag."""
        reasons: list[str] = []
        snap = self._provider.snapshot
        scanner = getattr(self, "_scanner", None)

        if not self._live:
            reasons.append("TUI is not in live runtime")
        if not bool(getattr(snap, "oanda_connected", False)):
            reasons.append("OANDA is not connected")
        if scanner is None or not bool(getattr(scanner, "is_ready", False)):
            reasons.append("embedded scanner is not ready")
        if bool(getattr(snap, "scanner_paused", False)):
            reasons.append("scanner is paused")
        if getattr(snap, "trades", []):
            reasons.append("open trades must be flat before unhalt")

        heartbeat_path = PROJECT_ROOT / ".claude" / "heartbeat.json"
        try:
            payload = json.loads(heartbeat_path.read_text())
            # Heartbeat schema (src/tui/heartbeat.py:HeartbeatPayload) uses
            # `ts_iso`. Reading `ts` returned "" → fromisoformat("") raised
            # `Invalid isoformat string: ''` and silently blocked every
            # unhalt attempt regardless of actual heartbeat freshness.
            ts_raw = str(payload.get("ts_iso") or payload.get("ts") or "")
            hb_ts = datetime.fromisoformat(ts_raw.replace("Z", "+00:00"))
            age = (datetime.now(timezone.utc) - hb_ts).total_seconds()
            if age > 30:
                reasons.append(f"heartbeat stale ({age:.0f}s old)")
            if not bool(payload.get("scanner_alive", False)):
                reasons.append("heartbeat says scanner_alive=false")
        except Exception as exc:
            reasons.append(f"heartbeat unavailable ({exc})")

        try:
            health = scanner.get_model_health() if scanner is not None else {}
            loaded = int(health.get("count", 0) or 0)
            total = int(health.get("total", 0) or 0)
            if total > 0 and loaded < total:
                reasons.append(f"models incomplete ({loaded}/{total} loaded)")
            freshness = dict(health.get("freshness", {}) or {})
            status = str(freshness.get("status", "")).upper()
            oldest = freshness.get("oldest_age_days")
            if status in {"STALE", "CRITICAL"}:
                if oldest is None:
                    reasons.append(f"models {status.lower()}")
                else:
                    reasons.append(f"models {status.lower()} (oldest {float(oldest):.0f}d)")
        except Exception as exc:
            reasons.append(f"model health unavailable ({exc})")

        return (len(reasons) == 0), reasons

    def _append_control_log(self, title: str, fields: dict[str, str]) -> None:
        """Append a compact control action entry to strategic_log.md."""
        try:
            ts = datetime.now(timezone.utc).isoformat()
            log_path = PROJECT_ROOT / ".claude" / "brain" / "strategic_log.md"
            if not log_path.exists():
                return
            field_text = "  ".join(f"**{k}:** {v}" for k, v in fields.items())
            with log_path.open("a", encoding="utf-8") as fh:
                fh.write(f"\n### {ts} — {title}\n{field_text}\n")
        except Exception as exc:
            logger.debug("control strategic_log write failed: %s", exc)

    def action_safe_restart(self) -> None:
        """Mythos audit 2026-04-29 Tier-7 step D — state-preserving restart.

        Replaces the manual ``pkill python -m src.tui && ./buddy``
        dance every commit. Sequence:

          1. Persist scanner runtime state (trade journal, replay
             buffer, EWMA tables, agent weights). All these have their
             own atomic-write paths via ``Scanner.save_scan_state`` /
             ``ScannerAgentTeam.save_learned_weights`` /
             ``StateEngine.flush`` — we just call them.
          2. Drop a soft-handoff beacon at ``.claude/state.json`` so the
             next boot knows it's a restart (vs cold start) and can
             resume mid-flight rather than re-walking init defaults.
          3. Shut down the scanner cleanly so OANDA streaming sockets
             close politely (broker logs cleaner reconnect).
          4. ``os.execv`` to replace the process image. Operator's
             terminal session is preserved (no shell hop), open OANDA
             positions live in the broker (not in process memory) so
             they survive the restart natively.

        Failure-tolerant — every step is wrapped; if any phase raises
        we still attempt the execv. Better an ungraceful restart than
        a half-state hang.
        """
        import os
        import sys
        from datetime import datetime, timezone

        # Mythos audit 2026-05-04 — log to buddy_debug.log at INFO so
        # `grep safe_restart logs/buddy_debug.log` proves the action
        # actually fired. Pre-fix the only signal was the brain feed
        # (which scrolls); operators couldn't tell whether Ctrl+R was
        # being intercepted by the shell vs. firing-but-failing.
        logger.info(
            "safe_restart: triggered pid=%s argv=%s", os.getpid(), sys.argv
        )

        try:
            self._write_brain(
                "[bold magenta]◈ SAFE RESTART — preserving state...[/]"
            )
        except Exception:
            pass

        # 1. Persist scanner runtime state.
        if self._scanner is not None:
            for method_name in (
                "save_scan_state",
                "save_state",
                "flush_state",
            ):
                try:
                    fn = getattr(self._scanner, method_name, None)
                    if callable(fn):
                        fn()
                except Exception as e:
                    logger.debug(
                        "safe_restart: scanner.%s failed: %s",
                        method_name, e,
                    )

        # 2. Soft-handoff beacon — distinguishes warm restart from cold boot.
        try:
            from pathlib import Path as _Path
            import json as _json
            beacon_path = _Path(".claude/state.json")
            beacon_path.parent.mkdir(parents=True, exist_ok=True)
            existing = {}
            if beacon_path.exists():
                try:
                    existing = _json.loads(beacon_path.read_text(encoding="utf-8"))
                except (OSError, _json.JSONDecodeError):
                    existing = {}
            if not isinstance(existing, dict):
                existing = {}
            existing["safe_restart"] = {
                "at": datetime.now(timezone.utc).isoformat(),
                "session_id": os.environ.get("BUDDY_SESSION_ID", ""),
                "reason": "operator_ctrl_r",
            }
            tmp_path = beacon_path.with_suffix(".json.tmp")
            tmp_path.write_text(
                _json.dumps(existing, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            os.replace(str(tmp_path), str(beacon_path))
        except Exception as e:
            logger.debug("safe_restart: beacon write failed: %s", e)

        # 3. Clean scanner shutdown — close OANDA streams politely.
        if self._scanner is not None:
            try:
                self._scanner.shutdown()
            except Exception as e:
                logger.debug("safe_restart: scanner.shutdown failed: %s", e)

        # 4. Replace process image. os.execv preserves terminal session
        # — operator's tmux pane stays attached to the new TUI.
        try:
            python = sys.executable or "python"
            argv = [python, "-m", "src.tui"] + sys.argv[1:]
            os.execv(python, argv)
        except Exception as e:
            # If execv itself fails, fall back to a clean quit so the
            # operator can manually relaunch.
            logger.error("safe_restart: execv failed err=%s — quitting", e)
            try:
                self._write_brain(
                    f"[red]✗ SAFE RESTART failed (execv): {e!r} — quitting[/]"
                )
            except Exception:
                pass
            super().action_quit()

    def action_quit(self) -> None:
        """Clean shutdown — stop scanner before exiting."""
        if self._scanner is not None:
            try:
                self._scanner.shutdown()
            except Exception:
                pass
        super().action_quit()


# ── Entry Point ─────────────────────────────────────────────────────


def run():
    """Launch the Buddy Command Bridge TUI."""
    parser = argparse.ArgumentParser(description="BUDDY — Cyberpunk Command Bridge")
    parser.add_argument("--demo", action="store_true",
                        help="Run with simulated data instead of live OANDA feed")
    args = parser.parse_args()

    # Pick #6: surface brain-file cap overages before Textual takes over
    # stderr. Non-blocking — never raises, never delays startup more than
    # ~5ms for the file reads.
    try:
        from src.tui.boot_trace import emit_brain_cap_warnings
        emit_brain_cap_warnings()
    except Exception:
        pass

    # Default: LIVE always. Only demo when explicitly requested.
    live = not args.demo

    app = BuddyApp(live=live)
    app.run()


if __name__ == "__main__":
    run()

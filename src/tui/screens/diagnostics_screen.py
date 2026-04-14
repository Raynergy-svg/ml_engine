"""
DIAGNOSTICS SCREEN -- "The Engine Room"

Five-panel layout in a 2x3 grid:
  1. SYSTEM VITALS -- CPU/MEM/DISK ProgressBars with labels
  2. OANDA API HEALTH -- connection status, latency sparkline, rate limit bars
  3. MODEL HEALTH -- DataTable showing TCN/Ridge/RF model status and accuracy
  4. ERROR LOG -- RichLog with severity-colored entries
  5. MAINTENANCE TASKS -- DataTable with task schedule and status (full width)

Drop-in replacement for PlaceholderContent in the Diagnostics TabPane.
"""
from __future__ import annotations

import json
import logging
import random
import time
from collections import deque
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from rich.text import Text

from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.widgets import Button, DataTable, Label, RichLog, Select, Static

from src.tui.data_provider import DashboardSnapshot

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Color palette (matches theme.tcss cyberpunk palette)
# ---------------------------------------------------------------------------
_BG = "#131320"
_BORDER = "#2a2a4a"
_PRIMARY = "#00ffcc"
_SECONDARY = "#ff00ff"
_POSITIVE = "#00ff41"
_NEGATIVE = "#ff1744"
_WARNING = "#ffab00"
_TEXT = "#e0e0ff"
_DIM = "#6666aa"
_DATA = "#7c4dff"

# Severity colors for error log
_SEV_COLORS = {
    "ERR": _NEGATIVE,
    "WARN": _WARNING,
    "INFO": _PRIMARY,
    "DEBUG": _DIM,
}


# ---------------------------------------------------------------------------
# Helper: sparkline from floats
# ---------------------------------------------------------------------------

def _sparkline_str(values: list[float], width: int = 30) -> str:
    """Convert a list of values into a unicode sparkline string."""
    if not values or len(values) < 2:
        return "░" * width
    mn, mx = min(values), max(values)
    rng = mx - mn if mx != mn else 0.001
    blocks = "▁▂▃▄▅▆▇█"
    subset = list(values)[-width:]
    return "".join(blocks[min(7, int((v - mn) / rng * 7))] for v in subset)


def _gauge_color(pct: float) -> str:
    """Return color based on usage percentage: green < 60, amber 60-80, red > 80."""
    if pct < 60:
        return _POSITIVE
    elif pct < 80:
        return _WARNING
    return _NEGATIVE


def _format_bytes(b: float) -> str:
    """Format bytes into human-readable string."""
    if b < 1024:
        return f"{b:.0f}B"
    elif b < 1024 ** 2:
        return f"{b / 1024:.1f}KB"
    elif b < 1024 ** 3:
        return f"{b / (1024 ** 2):.1f}MB"
    return f"{b / (1024 ** 3):.2f}GB"


def _format_age(dt: datetime) -> str:
    """Format a datetime as relative age string."""
    now = datetime.now(timezone.utc)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    delta = now - dt
    if delta.days > 0:
        return f"{delta.days}d ago"
    hours = delta.seconds // 3600
    if hours > 0:
        return f"{hours}h ago"
    minutes = delta.seconds // 60
    return f"{minutes}m ago"


# ---------------------------------------------------------------------------
# SystemVitalsPanel -- CPU / MEM / DISK gauges
# ---------------------------------------------------------------------------

class SystemVitalsPanel(Static):
    """System resource gauges: CPU, Memory, Disk, Process RSS."""

    DEFAULT_CSS = """
    SystemVitalsPanel {
        height: auto;
        min-height: 8;
        padding: 0 1;
    }
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._cpu_pct: float = 0.0
        self._mem_pct: float = 0.0
        self._mem_used: float = 0.0
        self._mem_total: float = 0.0
        self._disk_pct: float = 0.0
        self._disk_used: float = 0.0
        self._disk_total: float = 0.0
        self._proc_rss: float = 0.0

    def update_vitals(
        self,
        cpu_pct: float = 0.0,
        mem_pct: float = 0.0,
        mem_used: float = 0.0,
        mem_total: float = 0.0,
        disk_pct: float = 0.0,
        disk_used: float = 0.0,
        disk_total: float = 0.0,
        proc_rss: float = 0.0,
    ) -> None:
        self._cpu_pct = cpu_pct
        self._mem_pct = mem_pct
        self._mem_used = mem_used
        self._mem_total = mem_total
        self._disk_pct = disk_pct
        self._disk_used = disk_used
        self._disk_total = disk_total
        self._proc_rss = proc_rss
        self.refresh()

    def _render_gauge(self, t: Text, label: str, pct: float, detail: str, width: int = 20) -> None:
        """Render a single ASCII gauge bar with label."""
        color = _gauge_color(pct)
        filled = int(max(0, min(1, pct / 100)) * width)
        empty = width - filled

        t.append(f"  {label:<6}", style=f"bold {_DIM}")
        t.append(f" {pct:5.1f}%  ", style=f"bold {color}")
        t.append("[", style=_BORDER)
        t.append("█" * filled, style=color)
        t.append("░" * empty, style=_BORDER)
        t.append("]", style=_BORDER)
        t.append(f"  {detail}", style=_DIM)
        t.append("\n")

    def render(self) -> Text:
        t = Text()

        self._render_gauge(
            t, "CPU", self._cpu_pct,
            f"{self._cpu_pct:.0f}% utilization",
        )
        self._render_gauge(
            t, "MEM", self._mem_pct,
            f"{_format_bytes(self._mem_used)} / {_format_bytes(self._mem_total)}",
        )
        self._render_gauge(
            t, "DISK", self._disk_pct,
            f"{_format_bytes(self._disk_used)} / {_format_bytes(self._disk_total)}",
        )

        # Process RSS line
        t.append("\n")
        t.append("  PY RSS ", style=f"bold {_DIM}")
        rss_color = _POSITIVE if self._proc_rss < 500 * 1024 * 1024 else _WARNING
        t.append(f" {_format_bytes(self._proc_rss)}", style=f"bold {rss_color}")
        t.append("  (Python process)", style=_DIM)

        return t


# ---------------------------------------------------------------------------
# OandaHealthPanel -- connection status + latency sparkline
# ---------------------------------------------------------------------------

class OandaHealthPanel(Static):
    """OANDA API connection health, latency sparkline, rate limits."""

    DEFAULT_CSS = """
    OandaHealthPanel {
        height: auto;
        min-height: 8;
        padding: 0 1;
    }
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._connected: bool = False
        self._latency_ms: float = 0.0
        self._latency_history: deque[float] = deque(maxlen=30)
        self._last_ping_ms: float | None = None
        self._rate_used: float = 0.0
        self._rate_limit: float = 120.0

    def update_health(
        self,
        connected: bool = False,
        latency_ms: float = 0.0,
    ) -> None:
        self._connected = connected
        self._latency_ms = latency_ms
        if latency_ms > 0:
            self._latency_history.append(latency_ms)
        self.refresh()

    def set_ping_result(self, latency_ms: float | None) -> None:
        """Set the result of a manual ping test."""
        self._last_ping_ms = latency_ms
        if latency_ms is not None and latency_ms > 0:
            self._latency_history.append(latency_ms)
        self.refresh()

    def render(self) -> Text:
        t = Text()

        # Connection status
        if self._connected:
            t.append("  Status: ", style=_DIM)
            t.append("● CONNECTED", style=f"bold {_POSITIVE}")
        else:
            t.append("  Status: ", style=_DIM)
            t.append("● DISCONNECTED", style=f"bold {_NEGATIVE}")
        t.append("\n")

        # Current latency
        t.append("  Latency: ", style=_DIM)
        if self._latency_ms > 0:
            lat_color = _POSITIVE if self._latency_ms < 100 else _WARNING if self._latency_ms < 300 else _NEGATIVE
            t.append(f"{self._latency_ms:.0f}ms", style=f"bold {lat_color}")
        else:
            t.append("--", style=_DIM)
        t.append("\n")

        # Latency sparkline
        t.append("  History: ", style=_DIM)
        hist = list(self._latency_history)
        if hist:
            spark = _sparkline_str(hist, 20)
            avg = sum(hist) / len(hist)
            spark_color = _POSITIVE if avg < 100 else _WARNING if avg < 300 else _NEGATIVE
            t.append(spark, style=spark_color)
            t.append(f"  avg {avg:.0f}ms", style=_DIM)
        else:
            t.append("░" * 20, style=_BORDER)
            t.append("  no data", style=_DIM)
        t.append("\n")

        # Rate limit gauge
        rate_pct = (self._rate_used / self._rate_limit * 100) if self._rate_limit > 0 else 0
        t.append("  Rate:    ", style=_DIM)
        rate_color = _POSITIVE if rate_pct < 60 else _WARNING if rate_pct < 80 else _NEGATIVE
        filled = int(max(0, min(1, rate_pct / 100)) * 15)
        empty = 15 - filled
        t.append("[", style=_BORDER)
        t.append("█" * filled, style=rate_color)
        t.append("░" * empty, style=_BORDER)
        t.append("]", style=_BORDER)
        t.append(f" {self._rate_used:.0f}/{self._rate_limit:.0f} req/s", style=_DIM)
        t.append("\n")

        # Last ping result
        if self._last_ping_ms is not None:
            t.append("\n")
            t.append("  Last ping: ", style=_DIM)
            ping_color = _POSITIVE if self._last_ping_ms < 100 else _WARNING
            t.append(f"{self._last_ping_ms:.0f}ms", style=f"bold {ping_color}")

        return t


# ---------------------------------------------------------------------------
# DiagnosticsScreen -- Main Container
# ---------------------------------------------------------------------------

class DiagnosticsScreen(Container):
    """The Engine Room -- F6 screen.

    Five-panel layout in 2x3 grid:
      Row 0: System Vitals (left) + OANDA API Health (right)
      Row 1: Model Health DataTable (left) + Error Log RichLog (right)
      Row 2: Maintenance Tasks DataTable (full width)
    """

    DEFAULT_CSS = """
    DiagnosticsScreen {
        height: 1fr;
        layout: vertical;
    }
    """

    def __init__(self, project_root: str | None = None, live: bool = False, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if project_root:
            self._project_root = Path(project_root)
        else:
            self._project_root = Path(__file__).resolve().parents[3]
        self._live = live
        self._latency_history: deque[float] = deque(maxlen=30)
        self._error_log_entries: list[dict[str, Any]] = []
        self._severity_filter: str = "All"
        self._refresh_timer: Any = None
        self._demo_log_counter: int = 0

    def compose(self) -> ComposeResult:
        # Row 0: System Vitals + OANDA Health
        with Horizontal(id="diag-row-0"):
            with Vertical(id="diag-vitals-panel", classes="panel"):
                yield Label("  SYSTEM VITALS", classes="panel-title")
                yield SystemVitalsPanel(id="diag-vitals")

            with Vertical(id="diag-oanda-panel", classes="panel"):
                yield Label("  OANDA API HEALTH", classes="panel-title")
                yield OandaHealthPanel(id="diag-oanda")
                yield Button("TEST CONNECTION", id="diag-test-btn", variant="default")

        # Row 1: Model Health + Error Log
        with Horizontal(id="diag-row-1"):
            with Vertical(id="diag-models-panel", classes="panel"):
                yield Label("  MODEL HEALTH", classes="panel-title")
                yield DataTable(id="diag-models-table", cursor_type="row")

            with Vertical(id="diag-errors-panel", classes="panel"):
                with Horizontal(id="diag-error-header"):
                    yield Label("  ERROR LOG", classes="panel-title")
                    yield Select(
                        [
                            ("All", "All"),
                            ("ERR", "ERR"),
                            ("WARN", "WARN"),
                            ("INFO", "INFO"),
                        ],
                        value="All",
                        id="diag-severity-filter",
                        allow_blank=False,
                    )
                yield RichLog(
                    id="diag-error-log",
                    highlight=True,
                    markup=True,
                    max_lines=200,
                    auto_scroll=True,
                )

        # Row 2: Maintenance Tasks (full width)
        with Vertical(id="diag-maint-panel", classes="panel"):
            yield Label("  MAINTENANCE TASKS", classes="panel-title")
            yield DataTable(id="diag-maint-table", cursor_type="row")

    def on_mount(self) -> None:
        # Set up model health table
        model_table = self.query_one("#diag-models-table", DataTable)
        model_table.add_columns("Model", "Status", "Accuracy", "Last Trained", "Files")
        model_table.zebra_stripes = True

        # Set up maintenance tasks table
        maint_table = self.query_one("#diag-maint-table", DataTable)
        maint_table.add_columns("Task", "Schedule", "Last Run", "Next Due", "Status")
        maint_table.zebra_stripes = True

        # Initial data load
        self._refresh_vitals()
        self._refresh_models()
        self._refresh_maintenance()
        if not self._live:
            self._seed_error_log()

        # Start auto-refresh every 5 seconds
        self._refresh_timer = self.set_interval(5.0, self._on_refresh_tick)

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle TEST CONNECTION button press."""
        if event.button.id == "diag-test-btn":
            self._test_oanda_connection()

    def on_select_changed(self, event: Select.Changed) -> None:
        """Handle severity filter change."""
        if event.select.id == "diag-severity-filter":
            self._severity_filter = str(event.value) if event.value is not None else "All"
            self._refilter_error_log()

    # ------------------------------------------------------------------
    # Public: update from snapshot
    # ------------------------------------------------------------------

    def update_from_snapshot(self, snap: DashboardSnapshot) -> None:
        """Refresh panels from a new DashboardSnapshot."""
        # Update OANDA health from snapshot
        try:
            oanda_panel = self.query_one("#diag-oanda", OandaHealthPanel)
            oanda_panel.update_health(
                connected=snap.oanda_connected,
                latency_ms=snap.oanda_latency_ms,
            )
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Private: periodic refresh
    # ------------------------------------------------------------------

    def _on_refresh_tick(self) -> None:
        """Called every 5 seconds to refresh vitals (+ demo log entries if not live)."""
        self._refresh_vitals()
        if not self._live:
            self._add_demo_log_entry()

    # ------------------------------------------------------------------
    # Private: system vitals
    # ------------------------------------------------------------------

    def _refresh_vitals(self) -> None:
        """Refresh system vitals using psutil (with graceful fallback)."""
        cpu_pct = 0.0
        mem_pct = 0.0
        mem_used = 0.0
        mem_total = 0.0
        disk_pct = 0.0
        disk_used = 0.0
        disk_total = 0.0
        proc_rss = 0.0

        try:
            import psutil

            cpu_pct = psutil.cpu_percent(interval=None)

            mem = psutil.virtual_memory()
            mem_pct = mem.percent
            mem_used = float(mem.used)
            mem_total = float(mem.total)

            disk = psutil.disk_usage("/")
            disk_pct = disk.percent
            disk_used = float(disk.used)
            disk_total = float(disk.total)

            proc = psutil.Process()
            proc_rss = float(proc.memory_info().rss)

        except ImportError:
            # psutil not installed -- use demo fallbacks
            cpu_pct = random.uniform(8, 35)
            mem_pct = random.uniform(45, 72)
            mem_used = mem_pct / 100 * 16 * 1024 ** 3
            mem_total = 16 * 1024 ** 3
            disk_pct = random.uniform(40, 65)
            disk_used = disk_pct / 100 * 500 * 1024 ** 3
            disk_total = 500 * 1024 ** 3
            proc_rss = random.uniform(200, 450) * 1024 * 1024
        except Exception as exc:
            logger.debug("psutil error: %s", exc)
            # Return zeros -- vitals panel handles gracefully
            pass

        try:
            vitals_panel = self.query_one("#diag-vitals", SystemVitalsPanel)
            vitals_panel.update_vitals(
                cpu_pct=cpu_pct,
                mem_pct=mem_pct,
                mem_used=mem_used,
                mem_total=mem_total,
                disk_pct=disk_pct,
                disk_used=disk_used,
                disk_total=disk_total,
                proc_rss=proc_rss,
            )
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Private: model health
    # ------------------------------------------------------------------

    def _refresh_models(self) -> None:
        """Scan trained_data/models/ for model files and populate the table."""
        try:
            table = self.query_one("#diag-models-table", DataTable)
        except Exception:
            return

        table.clear()

        models_dir = self._project_root / "trained_data" / "models"

        # Define the three core models and their file patterns
        model_defs = [
            ("TCN", ["buddy_tf.keras", "buddy_tf_candidate.keras", "tcn_volatility_regime.keras"]),
            ("Ridge", ["buddy_meta.pkl", "buddy_meta.model.json"]),
            ("RF", ["rf_feature_selector.pkl", "rf_feature_selector_features.pkl"]),
        ]

        # Load agent weights for RL sync status
        rl_synced = False
        rl_sync_time: datetime | None = None
        weights_path = models_dir / "agent_weights.json"
        if weights_path.exists():
            try:
                stat = weights_path.stat()
                rl_sync_time = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc)
                rl_synced = True
            except Exception:
                pass

        for model_name, file_patterns in model_defs:
            found_files: list[str] = []
            latest_mtime: float = 0.0
            accuracy: float | None = None

            for pattern in file_patterns:
                fpath = models_dir / pattern
                if fpath.exists():
                    try:
                        found_files.append(pattern)
                        mtime = fpath.stat().st_mtime
                        if mtime > latest_mtime:
                            latest_mtime = mtime
                    except Exception:
                        pass

            # Determine status
            if found_files:
                status_text = "LOADED"
                status_color = _POSITIVE
            else:
                status_text = "MISSING"
                status_color = _NEGATIVE

            # Try to extract accuracy from meta files
            for pattern in file_patterns:
                if pattern.endswith(".json"):
                    meta_path = models_dir / pattern
                    if meta_path.exists():
                        try:
                            meta = json.loads(meta_path.read_text(encoding="utf-8"))
                            if isinstance(meta, dict):
                                acc = meta.get("accuracy", meta.get("test_accuracy", meta.get("val_accuracy")))
                                if acc is not None:
                                    accuracy = float(acc)
                                    break
                        except (json.JSONDecodeError, OSError, ValueError):
                            pass

            # Build styled cells
            name_text = Text(f" {model_name}", style=f"bold {_TEXT}")
            status_styled = Text(f" {status_text}", style=f"bold {status_color}")

            if accuracy is not None:
                acc_pct = accuracy * 100 if accuracy <= 1.0 else accuracy
                acc_color = _POSITIVE if acc_pct > 65 else _WARNING if acc_pct > 55 else _NEGATIVE
                acc_text = Text(f" {acc_pct:.1f}%", style=f"bold {acc_color}")
            else:
                acc_text = Text(" --", style=_DIM)

            if latest_mtime > 0:
                trained_dt = datetime.fromtimestamp(latest_mtime, tz=timezone.utc)
                age_str = _format_age(trained_dt)
                age_text = Text(f" {age_str}", style=_DIM)
            else:
                age_text = Text(" never", style=_NEGATIVE)

            files_text = Text(f" {len(found_files)}/{len(file_patterns)}", style=_DIM)

            table.add_row(name_text, status_styled, acc_text, age_text, files_text)

        # Add RL Sync row
        rl_name = Text(" RL Weights", style=f"bold {_TEXT}")
        if rl_synced and rl_sync_time:
            rl_status = Text(" SYNCED", style=f"bold {_POSITIVE}")
            rl_age = Text(f" {_format_age(rl_sync_time)}", style=_DIM)
        else:
            rl_status = Text(" NOT SYNCED", style=f"bold {_WARNING}")
            rl_age = Text(" --", style=_DIM)
        rl_acc = Text(" --", style=_DIM)
        rl_files = Text(f" {'1/1' if rl_synced else '0/1'}", style=_DIM)
        table.add_row(rl_name, rl_status, rl_acc, rl_age, rl_files)

    # ------------------------------------------------------------------
    # Private: error log
    # ------------------------------------------------------------------

    def _seed_error_log(self) -> None:
        """Seed the error log with initial demo entries."""
        now = datetime.now(timezone.utc)
        seed_entries = [
            {"ts": now - timedelta(minutes=12), "sev": "INFO", "msg": "Scanner initialized — 7 pairs loaded"},
            {"ts": now - timedelta(minutes=10), "sev": "INFO", "msg": "OANDA connection established"},
            {"ts": now - timedelta(minutes=8), "sev": "INFO", "msg": "Models loaded: TCN + Ridge + RF ensemble ready"},
            {"ts": now - timedelta(minutes=5), "sev": "WARN", "msg": "EUR/GBP spread elevated: 3.2 pips (threshold: 2.5)"},
            {"ts": now - timedelta(minutes=3), "sev": "INFO", "msg": "Scan cycle #427 completed — 2 tradeable setups"},
            {"ts": now - timedelta(minutes=2), "sev": "WARN", "msg": "GBP/JPY model disagreement: 0.31 > 0.30 threshold"},
            {"ts": now - timedelta(minutes=1), "sev": "ERR", "msg": "Rate limit warning: 95/120 requests in last 60s"},
            {"ts": now, "sev": "INFO", "msg": "Diagnostics screen initialized"},
        ]
        self._error_log_entries = seed_entries
        self._refilter_error_log()

    def _add_demo_log_entry(self) -> None:
        """Add a simulated log entry periodically."""
        self._demo_log_counter += 1

        # Generate a realistic log entry every few ticks
        if self._demo_log_counter % 3 != 0:
            return

        now = datetime.now(timezone.utc)
        templates = [
            ("INFO", "Scan cycle #{n} completed — {t:.1f}s"),
            ("INFO", "Agent weights refreshed from RL feedback"),
            ("INFO", "Correlation filter: all pairs within limits"),
            ("WARN", "Latency spike detected: {lat:.0f}ms"),
            ("WARN", "Memory usage elevated: {mem:.0f}MB RSS"),
            ("DEBUG", "Heartbeat: OANDA API responding normally"),
            ("INFO", "Drawdown guardian check: {dd:.1f}% — OK"),
            ("ERR", "OANDA timeout on pricing stream (retry #{r})"),
            ("WARN", "Model staleness: Ridge model trained {d}d ago"),
            ("INFO", "Journal compaction: {j} entries archived"),
        ]

        sev, tmpl = random.choice(templates)
        msg = tmpl.format(
            n=self._demo_log_counter + 400,
            t=random.uniform(0.3, 2.5),
            lat=random.uniform(200, 800),
            mem=random.uniform(300, 600),
            dd=random.uniform(1.0, 6.0),
            r=random.randint(1, 3),
            d=random.randint(2, 14),
            j=random.randint(10, 50),
        )

        entry = {"ts": now, "sev": sev, "msg": msg}
        self._error_log_entries.append(entry)

        # Keep log bounded
        if len(self._error_log_entries) > 200:
            self._error_log_entries = self._error_log_entries[-200:]

        # Only write if passes current filter
        if self._severity_filter == "All" or self._severity_filter == sev:
            self._write_log_entry(entry)

    def _refilter_error_log(self) -> None:
        """Re-render the error log with the current severity filter."""
        try:
            log_widget = self.query_one("#diag-error-log", RichLog)
        except Exception:
            return

        log_widget.clear()
        for entry in self._error_log_entries:
            sev = entry.get("sev", "INFO")
            if self._severity_filter == "All" or self._severity_filter == sev:
                self._write_log_entry(entry)

    def _write_log_entry(self, entry: dict[str, Any]) -> None:
        """Write a single log entry to the RichLog widget."""
        try:
            log_widget = self.query_one("#diag-error-log", RichLog)
        except Exception:
            return

        ts: datetime = entry.get("ts", datetime.now(timezone.utc))
        sev: str = entry.get("sev", "INFO")
        msg: str = entry.get("msg", "")
        color = _SEV_COLORS.get(sev, _DIM)

        ts_str = ts.strftime("%H:%M:%S")
        markup_line = f"[dim]{ts_str}[/] [{color}][{sev:<5}][/] [{color}]{msg}[/]"

        try:
            log_widget.write(Text.from_markup(markup_line))
        except Exception:
            # Fallback: plain text
            log_widget.write(f"{ts_str} [{sev}] {msg}")

    # ------------------------------------------------------------------
    # Private: maintenance tasks
    # ------------------------------------------------------------------

    def _refresh_maintenance(self) -> None:
        """Populate the maintenance tasks table with system task schedule."""
        try:
            table = self.query_one("#diag-maint-table", DataTable)
        except Exception:
            return

        table.clear()

        now = datetime.now(timezone.utc)

        # Static list of system maintenance tasks with simulated dates
        tasks = [
            {
                "name": "Journal Compaction",
                "schedule": "Daily @ 00:00",
                "last_run": now - timedelta(hours=random.randint(1, 23)),
                "next_due": now + timedelta(hours=random.randint(1, 23)),
                "status": "OK",
            },
            {
                "name": "Learnings Consolidation",
                "schedule": "Every 10 cycles",
                "last_run": now - timedelta(hours=random.randint(2, 48)),
                "next_due": now + timedelta(hours=random.randint(1, 12)),
                "status": "OK",
            },
            {
                "name": "Config Archive",
                "schedule": "Weekly (Mon)",
                "last_run": now - timedelta(days=random.randint(1, 6)),
                "next_due": now + timedelta(days=random.randint(1, 6)),
                "status": "OK",
            },
            {
                "name": "Model Retraining",
                "schedule": "Every 500 trades",
                "last_run": now - timedelta(days=random.randint(3, 14)),
                "next_due": now + timedelta(days=random.randint(1, 30)),
                "status": "PENDING" if random.random() > 0.7 else "OK",
            },
            {
                "name": "State Validation",
                "schedule": "Hourly",
                "last_run": now - timedelta(minutes=random.randint(5, 55)),
                "next_due": now + timedelta(minutes=random.randint(5, 55)),
                "status": "OK",
            },
            {
                "name": "Weight Backup",
                "schedule": "Daily @ 06:00",
                "last_run": now - timedelta(hours=random.randint(1, 23)),
                "next_due": now + timedelta(hours=random.randint(1, 23)),
                "status": "OK",
            },
        ]

        # Check for real files to improve status accuracy
        journal_path = self._project_root / "trained_data" / "trade_journal_rl.json"
        if journal_path.exists():
            try:
                stat = journal_path.stat()
                tasks[0]["last_run"] = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc)
            except Exception:
                pass

        learnings_path = self._project_root / ".claude" / "learnings.md"
        if learnings_path.exists():
            try:
                stat = learnings_path.stat()
                tasks[1]["last_run"] = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc)
            except Exception:
                pass

        weights_snapshot_dir = self._project_root / "trained_data" / "models" / "weight_snapshots"
        if weights_snapshot_dir.exists():
            try:
                snapshots = list(weights_snapshot_dir.iterdir())
                if snapshots:
                    latest = max(snapshots, key=lambda p: p.stat().st_mtime)
                    tasks[5]["last_run"] = datetime.fromtimestamp(
                        latest.stat().st_mtime, tz=timezone.utc
                    )
            except Exception:
                pass

        for task in tasks:
            name_text = Text(f" {task['name']}", style=_TEXT)
            sched_text = Text(f" {task['schedule']}", style=_DIM)

            last_run: datetime = task["last_run"]
            last_str = _format_age(last_run)
            last_text = Text(f" {last_str}", style=_DIM)

            next_due: datetime = task["next_due"]
            # Check if overdue
            is_overdue = next_due < now
            next_str = _format_age(next_due) if is_overdue else next_due.strftime("%H:%M UTC")
            next_color = _NEGATIVE if is_overdue else _DIM
            next_text = Text(f" {next_str}", style=next_color)

            status = task["status"]
            if status == "OK":
                status_text = Text(" OK", style=f"bold {_POSITIVE}")
            elif status == "PENDING":
                status_text = Text(" PENDING", style=f"bold {_WARNING}")
            else:
                status_text = Text(f" {status}", style=f"bold {_NEGATIVE}")

            table.add_row(name_text, sched_text, last_text, next_text, status_text)

    # ------------------------------------------------------------------
    # Private: OANDA connection test
    # ------------------------------------------------------------------

    def _test_oanda_connection(self) -> None:
        """Test OANDA connection and report latency."""
        try:
            oanda_panel = self.query_one("#diag-oanda", OandaHealthPanel)
        except Exception:
            return

        # Try a real connection test via the broker
        try:
            from src.brokers.oanda import OandaBroker
            broker = OandaBroker.from_env()

            t0 = time.monotonic()
            broker.connect()
            nav = broker.get_nav()
            latency = (time.monotonic() - t0) * 1000

            oanda_panel.update_health(connected=True, latency_ms=latency)
            oanda_panel.set_ping_result(latency)

            # Log the result
            entry = {
                "ts": datetime.now(timezone.utc),
                "sev": "INFO",
                "msg": f"Connection test passed: {latency:.0f}ms — NAV ${nav:,.2f}",
            }
            self._error_log_entries.append(entry)
            if self._severity_filter in ("All", "INFO"):
                self._write_log_entry(entry)

        except Exception as exc:
            # Connection failed -- report it
            oanda_panel.update_health(connected=False, latency_ms=0.0)
            oanda_panel.set_ping_result(None)

            entry = {
                "ts": datetime.now(timezone.utc),
                "sev": "ERR",
                "msg": f"Connection test failed: {exc}",
            }
            self._error_log_entries.append(entry)
            if self._severity_filter in ("All", "ERR"):
                self._write_log_entry(entry)

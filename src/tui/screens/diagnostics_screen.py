"""
DIAGNOSTICS SCREEN -- "The Engine Room"

Five-panel layout in a 2x3 grid:
  1. SYSTEM VITALS -- CPU/MEM/DISK ProgressBars with labels
  2. OANDA API HEALTH -- connection status, latency sparkline, rate limit bars
  3. MODEL HEALTH -- DataTable showing Transformer/LightGBM/XGBoost model status and accuracy
  4. ERROR LOG -- RichLog with severity-colored entries
  5. MAINTENANCE TASKS -- DataTable with task schedule and status (full width)

Drop-in replacement for PlaceholderContent in the Diagnostics TabPane.
"""
from __future__ import annotations

import json
import logging
import re
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
from src.tui.widgets.model_inventory import ModelInventory

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

_LOG_LINE_RE = re.compile(
    r"^(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\s+-\s+"
    r"(?P<logger>.*?)\s+-\s+(?P<level>[A-Z]+)\s+-\s+(?P<msg>.*)$"
)
_LOG_SOURCES = (
    Path("logs/buddy_tui.log"),
    Path("logs/buddy_tui.stderr.log"),
    Path("logs/buddy_headless_run.log"),
)
_JSONL_SOURCES = (
    Path("logs/reflection_log.jsonl"),
    Path("trained_data/alerts.jsonl"),
)


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


def _parse_dt(value: Any) -> datetime | None:
    """Best-effort parse for ISO strings, epoch seconds, and log timestamps."""
    if value in (None, ""):
        return None
    if isinstance(value, (int, float)):
        try:
            return datetime.fromtimestamp(float(value), tz=timezone.utc)
        except Exception:
            return None
    text = str(value).strip()
    if not text:
        return None
    for candidate in (text, text.replace("Z", "+00:00")):
        try:
            dt = datetime.fromisoformat(candidate)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)
        except ValueError:
            pass
    try:
        return datetime.strptime(text[:19], "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def _read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _tail_lines(path: Path, max_lines: int = 120) -> list[str]:
    try:
        with path.open("r", encoding="utf-8", errors="replace") as fh:
            return fh.readlines()[-max_lines:]
    except Exception:
        return []


def _severity_from_level(level: str) -> str:
    level = level.upper()
    if level in {"ERROR", "CRITICAL", "ERR", "FATAL"}:
        return "ERR"
    if level in {"WARNING", "WARN"}:
        return "WARN"
    if level == "DEBUG":
        return "DEBUG"
    return "INFO"


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
        self._psutil_error_shown: bool = False

    def show_psutil_error(self) -> None:
        """Mount a visible error banner when psutil is not installed."""
        if self._psutil_error_shown:
            return
        self._psutil_error_shown = True
        self.mount(
            Static(
                "psutil missing — install with `pip install psutil` to see vitals",
                classes="error-banner",
            )
        )

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
        if self._psutil_error_shown:
            return Text()
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
        self._latest_snapshot: DashboardSnapshot | None = None

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

        # Row 3: Per-Pair Model Inventory (T10 — full width)
        with Vertical(id="diag-model-inv-panel", classes="panel"):
            yield Label("  PER-PAIR MODEL INVENTORY", classes="panel-title")
            yield DataTable(id="diag-model-inv-table", cursor_type="row")

    def on_mount(self) -> None:
        # Set up model health table
        model_table = self.query_one("#diag-models-table", DataTable)
        model_table.add_columns("Model", "Status", "Accuracy", "Last Trained", "Files")
        model_table.zebra_stripes = True

        # Set up maintenance tasks table
        maint_table = self.query_one("#diag-maint-table", DataTable)
        maint_table.add_columns("Task", "Schedule", "Last Run", "Next Due", "Status")
        maint_table.zebra_stripes = True

        # Set up per-pair model inventory table (T10)
        inv_table = self.query_one("#diag-model-inv-table", DataTable)
        inv_table.add_columns(
            "Pair", "Status", "Granularity", "Holdout %", "Age (days)", "Pipeline Ver"
        )
        inv_table.zebra_stripes = True
        self._model_inventory = ModelInventory(
            models_root=self._project_root / "trained_data" / "models"
        )

        # Initial data load
        self._refresh_vitals()
        self._refresh_models()
        self._refresh_maintenance()
        self._refresh_inventory()
        # P0-1 (2026-06-21 audit): render REAL diagnostics from repo artifacts
        # regardless of live/demo. Previously demo mode seeded hardcoded FX
        # lines ("OANDA connection established", "EUR/GBP spread elevated")
        # and a 5s demo-entry tick kept appending fake FX log lines — pure
        # fiction now that the equity harvester is the live stack. We read the
        # same real log sources in both modes and show an honest empty state
        # when nothing is on disk.
        self._refresh_error_log()

        # Start auto-refresh every 5 seconds
        self._refresh_timer = self.set_interval(5.0, self._on_refresh_tick)
        # Per-pair inventory refreshes on its own slower cadence (30s).
        self._inventory_timer = self.set_interval(30.0, self._refresh_inventory)

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
        self._latest_snapshot = snap

        # Update OANDA health from snapshot
        try:
            oanda_panel = self.query_one("#diag-oanda", OandaHealthPanel)
            oanda_panel.update_health(
                connected=snap.oanda_connected,
                latency_ms=snap.oanda_latency_ms,
            )
        except Exception:
            pass

        self._refresh_models(snap)
        self._refresh_maintenance(snap)
        if self._live:
            self._refresh_error_log()
        if snap.last_error:
            self._append_error_entry("WARN", snap.last_error)

    # ------------------------------------------------------------------
    # Private: periodic refresh
    # ------------------------------------------------------------------

    def _on_refresh_tick(self) -> None:
        """Called every 5 seconds to refresh vitals + real diagnostics.

        P0-1 (2026-06-21 audit): no longer appends fake FX demo entries in
        demo mode. Real log sources are read in both live and demo mode; an
        honest empty state is shown when nothing is on disk.
        """
        self._refresh_vitals()
        self._refresh_models(self._latest_snapshot)
        self._refresh_maintenance(self._latest_snapshot)
        self._refresh_error_log()

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
            try:
                vitals_panel = self.query_one("#diag-vitals", SystemVitalsPanel)
                vitals_panel.show_psutil_error()
            except Exception:
                pass
            return
        except Exception as exc:
            logger.debug("psutil error: %s", exc)
            return

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

    def _refresh_models(self, snap: DashboardSnapshot | None = None) -> None:
        """Populate model health from live scanner snapshot, with filesystem fallback."""
        try:
            table = self.query_one("#diag-models-table", DataTable)
        except Exception:
            return

        table.clear()

        if snap is not None and snap.models_detail:
            self._render_snapshot_model_health(table, snap)
            return

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

    def _render_snapshot_model_health(
        self,
        table: DataTable,
        snap: DashboardSnapshot,
    ) -> None:
        """Render Scanner.get_model_health() details pushed through DashboardSnapshot."""
        models_dir = self._project_root / "trained_data" / "models"
        detail = dict(snap.models_detail or {})
        freshness = self._read_model_freshness()

        for name, loaded in sorted(detail.items()):
            status_color = _POSITIVE if loaded else _NEGATIVE
            status = "LOADED" if loaded else "MISSING"
            row_name = Text(f" {name}", style=f"bold {_TEXT}")
            row_status = Text(f" {status}", style=f"bold {status_color}")
            accuracy = Text(" --", style=_DIM)
            trained = Text(f" {self._model_age_for(name, freshness, models_dir)}", style=_DIM)
            files = Text(" live", style=_DIM)
            table.add_row(row_name, row_status, accuracy, trained, files)

        if not detail:
            table.add_row(
                Text(" scanner models", style=f"bold {_TEXT}"),
                Text(" WAITING", style=f"bold {_WARNING}"),
                Text(" --", style=_DIM),
                Text(" no scan snapshot", style=_DIM),
                Text(" --", style=_DIM),
            )

    def _read_model_freshness(self) -> dict[str, Any]:
        try:
            from src.scanner.automation.model_freshness import get_model_freshness
            data = get_model_freshness()
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    def _model_age_for(
        self,
        model_name: str,
        freshness: dict[str, Any],
        models_dir: Path,
    ) -> str:
        for group in freshness.get("groups", []) or []:
            if not isinstance(group, dict):
                continue
            group_name = str(group.get("name", ""))
            if group_name and (group_name in model_name or model_name in group_name):
                trained_at = _parse_dt(group.get("trained_at"))
                if trained_at is not None:
                    return _format_age(trained_at)
                age = group.get("age_days")
                if age is not None:
                    try:
                        return f"{float(age):.0f}d ago"
                    except Exception:
                        pass

        candidates = self._model_file_candidates(model_name, models_dir)
        existing = [p for p in candidates if p.exists()]
        if existing:
            latest = max(existing, key=lambda p: p.stat().st_mtime)
            return _format_age(datetime.fromtimestamp(latest.stat().st_mtime, tz=timezone.utc))
        return "--"

    def _model_file_candidates(self, model_name: str, models_dir: Path) -> list[Path]:
        mapping = {
            "agent_team_weights": [models_dir / "agent_weights.json"],
            "calibration_params": [models_dir / "calibration_params.json"],
            "pair_affinity_matrix": [models_dir / "pair_affinity_matrix.json"],
            "modular_ensemble_meta": [models_dir / "modular_ensemble.meta.json"],
            "rl_position_sizer": [models_dir / "rl_position_sizer.zip"],
            "rl_gate_thresholds": [models_dir / "sac_gate_thresholds.zip"],
            "rl_optimal_exit": [models_dir / "ppo_optimal_exit.zip"],
            "confidence_ridge": [models_dir / "joint" / "ridge_confidence.pkl"],
            "risk_rf": [models_dir / "joint" / "lgbm_risk.pkl"],
            "momentum_lightgbm": [models_dir / "joint" / "lgbm_momentum.pkl"],
            "transformer_direction": [models_dir / "joint" / "transformer_direction.keras"],
            "transformer_gate": [models_dir / "joint" / "transformer_direction.keras"],
            "tcn_volatility": [models_dir / "joint" / "tcn_volatility_regime.keras"],
            "tcn_volatility_gate": [models_dir / "joint" / "tcn_volatility_regime.keras"],
        }
        return mapping.get(model_name, [models_dir / model_name])

    # ------------------------------------------------------------------
    # Private: error log
    # ------------------------------------------------------------------

    def _refresh_error_log(self) -> None:
        """Load recent live diagnostics and runtime errors from repo artifacts."""
        entries = self._load_live_log_entries()
        if not entries:
            entries = [{
                "ts": datetime.now(timezone.utc),
                "sev": "INFO",
                "msg": "No live diagnostic log entries found yet",
            }]
        self._error_log_entries = entries[-200:]
        self._refilter_error_log()

    def _load_live_log_entries(self) -> list[dict[str, Any]]:
        entries: list[dict[str, Any]] = []

        for rel_path in _LOG_SOURCES:
            path = self._project_root / rel_path
            for line in _tail_lines(path):
                parsed = self._parse_standard_log_line(line, rel_path.name)
                if parsed is not None:
                    entries.append(parsed)

        for rel_path in _JSONL_SOURCES:
            path = self._project_root / rel_path
            for line in _tail_lines(path):
                parsed = self._parse_jsonl_log_line(line, rel_path.name)
                if parsed is not None:
                    entries.append(parsed)

        scan_report = _read_json(
            self._project_root / "trained_data" / "scan_diagnostics_report.json",
            default={},
        )
        if isinstance(scan_report, dict) and scan_report:
            ts = _parse_dt(scan_report.get("timestamp")) or datetime.now(timezone.utc)
            summary = scan_report.get("diagnostic_summary", "Scan diagnostics updated")
            sev = "WARN" if scan_report.get("stalled") else "INFO"
            entries.append({"ts": ts, "sev": sev, "msg": str(summary)[:240]})

        health = _read_json(
            self._project_root / "trained_data" / "health_registry_log.json",
            default={},
        )
        if isinstance(health, dict) and health:
            try:
                rejected = int(health.get("cycles_rejected", 0) or 0)
            except (TypeError, ValueError):
                rejected = 0
            states = health.get("module_states", {}) or {}
            failing = [
                name for name, state in states.items()
                if isinstance(state, dict) and (
                    state.get("last_check_passed") is False or state.get("disabled")
                )
            ]
            sev = "ERR" if rejected or failing else "INFO"
            msg = (
                f"Health registry: {health.get('total_preflights', 0)} preflights, "
                f"{rejected} rejected cycles"
            )
            if failing:
                msg += f", failing: {', '.join(failing[:4])}"
            # health_registry_log.json can be rotated/removed by the live
            # registry between the _read_json above and this stat(); fall back
            # to now() so one missing file can't crash the 5s refresh tick.
            try:
                _health_ts = datetime.fromtimestamp(
                    (
                        self._project_root / "trained_data"
                        / "health_registry_log.json"
                    ).stat().st_mtime,
                    tz=timezone.utc,
                )
            except OSError:
                _health_ts = datetime.now(timezone.utc)
            entries.append({"ts": _health_ts, "sev": sev, "msg": msg})

        entries.sort(key=lambda e: e.get("ts", datetime.min.replace(tzinfo=timezone.utc)))
        return entries[-200:]

    def _parse_standard_log_line(
        self,
        line: str,
        source_name: str,
    ) -> dict[str, Any] | None:
        match = _LOG_LINE_RE.match(line.strip())
        if not match:
            return None
        ts = _parse_dt(match.group("ts")) or datetime.now(timezone.utc)
        sev = _severity_from_level(match.group("level"))
        logger_name = match.group("logger").rsplit(".", 1)[-1]
        msg = match.group("msg").strip()
        return {"ts": ts, "sev": sev, "msg": f"{logger_name}: {msg}"[:300]}

    def _parse_jsonl_log_line(
        self,
        line: str,
        source_name: str,
    ) -> dict[str, Any] | None:
        try:
            data = json.loads(line)
        except Exception:
            return None
        if not isinstance(data, dict):
            return None
        ts = (
            _parse_dt(data.get("ts"))
            or _parse_dt(data.get("timestamp"))
            or _parse_dt(data.get("created_at"))
            or datetime.now(timezone.utc)
        )
        error = data.get("error")
        status = str(data.get("status", "")).lower()
        success = data.get("success")
        sev = "ERR" if error or success is False or status in {"error", "failed"} else "INFO"
        msg = (
            data.get("hypothesis")
            or data.get("summary")
            or data.get("message")
            or error
            or data.get("event_type")
            or source_name
        )
        trade_id = data.get("trade_id")
        if trade_id:
            msg = f"{trade_id}: {msg}"
        return {"ts": ts, "sev": sev, "msg": str(msg)[:300]}

    def _append_error_entry(self, sev: str, msg: str) -> None:
        entry = {
            "ts": datetime.now(timezone.utc),
            "sev": sev,
            "msg": str(msg)[:300],
        }
        if self._error_log_entries and self._error_log_entries[-1].get("msg") == entry["msg"]:
            return
        self._error_log_entries.append(entry)
        self._error_log_entries = self._error_log_entries[-200:]
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

    def _refresh_maintenance(self, snap: DashboardSnapshot | None = None) -> None:
        """Populate the maintenance table from live state and persisted artifacts."""
        try:
            table = self.query_one("#diag-maint-table", DataTable)
        except Exception:
            return

        table.clear()

        now = datetime.now(timezone.utc)

        tasks = self._build_maintenance_tasks(now, snap)

        for task in tasks:
            name_text = Text(f" {task['name']}", style=_TEXT)
            sched_text = Text(f" {task['schedule']}", style=_DIM)

            last_run: datetime = task["last_run"]
            last_str = "never" if last_run.timestamp() == 0 else _format_age(last_run)
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

    def _build_maintenance_tasks(
        self,
        now: datetime,
        snap: DashboardSnapshot | None = None,
    ) -> list[dict[str, Any]]:
        tasks: list[dict[str, Any]] = []

        def file_time(rel_path: str) -> datetime | None:
            path = self._project_root / rel_path
            if not path.exists():
                return None
            return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)

        journal = file_time("trained_data/trade_journal_rl.json")
        tasks.append(self._make_task(
            "Journal Compaction",
            "Daily / on write",
            journal,
            timedelta(days=1),
            missing_status="PENDING",
        ))

        learnings = file_time(".claude/learnings.md")
        tasks.append(self._make_task(
            "Learnings Consolidation",
            "Every 10 cycles",
            learnings,
            timedelta(hours=12),
            missing_status="PENDING",
        ))

        config_archive = (
            file_time(".claude/config_adjustments.json.bak")
            or file_time(".claude/config_adjustments.json")
        )
        tasks.append(self._make_task(
            "Config Archive",
            "Weekly",
            config_archive,
            timedelta(days=7),
            missing_status="PENDING",
        ))

        retrain_state = _read_json(
            self._project_root / "trained_data" / ".autonomous_retrain_state.json",
            default={},
        )
        retrain_last = None
        retrain_status = "PENDING"
        if isinstance(retrain_state, dict) and retrain_state:
            retrain_last = _parse_dt(
                retrain_state.get("last_completed_at")
                or retrain_state.get("last_started_at")
            )
            raw_status = str(retrain_state.get("last_status", "unknown")).lower()
            if raw_status == "success":
                retrain_status = "OK"
            elif raw_status == "running":
                retrain_status = "PENDING"
            else:
                retrain_status = "ERROR"
        tasks.append(self._make_task(
            "Model Retraining",
            "Stale/weekly trigger",
            retrain_last,
            timedelta(days=5),
            status=retrain_status,
            missing_status="PENDING",
        ))

        health = _read_json(
            self._project_root / "trained_data" / "health_registry_log.json",
            default={},
        )
        health_status = "PENDING"
        if isinstance(health, dict) and health:
            modules = health.get("module_states", {}) or {}
            failing = [
                name for name, state in modules.items()
                if isinstance(state, dict) and (
                    state.get("last_check_passed") is False or state.get("disabled")
                )
            ]
            health_status = "ERROR" if failing or health.get("cycles_rejected", 0) else "OK"
        tasks.append(self._make_task(
            "State Validation",
            "Every cycle",
            file_time("trained_data/health_registry_log.json"),
            timedelta(hours=1),
            status=health_status,
            missing_status="PENDING",
        ))

        latest_snapshot = None
        snapshots_dir = self._project_root / "trained_data" / "models" / "weight_snapshots"
        if snapshots_dir.exists():
            snapshots = [p for p in snapshots_dir.iterdir() if p.is_file()]
            if snapshots:
                latest = max(snapshots, key=lambda p: p.stat().st_mtime)
                latest_snapshot = datetime.fromtimestamp(latest.stat().st_mtime, tz=timezone.utc)
        tasks.append(self._make_task(
            "Weight Backup",
            "Daily @ 06:00",
            latest_snapshot,
            timedelta(days=1),
            missing_status="PENDING",
        ))

        scan_diag = (
            file_time("trained_data/scan_diagnostics_state.json")
            or file_time("trained_data/scan_diagnostics_report.json")
        )
        scan_report = _read_json(
            self._project_root / "trained_data" / "scan_diagnostics_report.json",
            default={},
        )
        scan_status = "OK"
        if isinstance(scan_report, dict) and scan_report.get("stalled"):
            scan_status = "ERROR"
        elif snap is not None and snap.scanner_ready is False:
            scan_status = "PENDING"
        tasks.append(self._make_task(
            "Scan Diagnostics",
            "Every 10 scans",
            scan_diag,
            timedelta(hours=1),
            status=scan_status,
            missing_status="PENDING",
        ))

        return tasks

    def _make_task(
        self,
        name: str,
        schedule: str,
        last_run: datetime | None,
        max_age: timedelta,
        status: str | None = None,
        missing_status: str = "PENDING",
    ) -> dict[str, Any]:
        now = datetime.now(timezone.utc)
        if last_run is None:
            last_run = datetime.fromtimestamp(0, tz=timezone.utc)
            task_status = missing_status
            next_due = now
        else:
            next_due = last_run + max_age
            task_status = status or ("OK" if next_due >= now else "PENDING")
        return {
            "name": name,
            "schedule": schedule,
            "last_run": last_run,
            "next_due": next_due,
            "status": task_status,
        }

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

    # ------------------------------------------------------------------
    # T10: Per-pair model inventory
    # ------------------------------------------------------------------

    def _refresh_inventory(self) -> None:
        """Rescan trained_data/models/ and rebuild the per-pair inventory table.

        Called once on mount and every 30s thereafter. Safe to call when the
        models dir is missing (ModelInventory.scan() handles that).
        """
        try:
            self._model_inventory.scan()
            table = self.query_one("#diag-model-inv-table", DataTable)
            table.clear()
            for c in self._model_inventory.cards():
                table.add_row(
                    c.pair,
                    c.status,
                    c.granularity or "—",
                    f"{c.holdout_accuracy:.1%}" if c.holdout_accuracy is not None else "—",
                    f"{c.age_days:.1f}" if c.age_days is not None else "—",
                    c.pipeline_version or "—",
                )
        except Exception as exc:  # noqa: BLE001 — refresh tick must never crash
            logger.warning("Per-pair model inventory refresh failed: %s", exc)

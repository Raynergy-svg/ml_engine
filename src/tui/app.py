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
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path

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
from src.tui.screens.trades_screen import TradesScreen
from src.tui.screens.agents_screen import AgentsScreen
from src.tui.screens.journal_screen import JournalScreen

# Lazy imports for screens still being built — fallback to None
try:
    from src.tui.screens.config_screen import ConfigScreen
except ImportError:
    ConfigScreen = None  # type: ignore[misc,assignment]

try:
    from src.tui.screens.diagnostics_screen import DiagnosticsScreen
except ImportError:
    DiagnosticsScreen = None  # type: ignore[misc,assignment]

logger = logging.getLogger(__name__)

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


class HeaderBar(Static):
    """Persistent header: NAV, P/L, connection status."""

    nav = reactive(0.0)
    pnl = reactive(0.0)
    open_count = reactive(0)
    oanda_ok = reactive(False)
    scanner_ok = reactive(False)
    mode_label = reactive("DEMO")

    def render(self) -> Text:
        now = datetime.now(timezone.utc).strftime("%H:%M UTC")
        pnl_sign = "+" if self.pnl >= 0 else ""
        pnl_color = "green" if self.pnl >= 0 else "red"
        oanda_style = "bold #00ff41" if self.oanda_ok else "bold #ff1744"
        oanda_text = "● LIVE" if self.oanda_ok else "● OFF"
        scanner_style = "bold #00ff41" if self.scanner_ok else "bold #6666aa"
        scanner_text = "● ACTIVE" if self.scanner_ok else "● IDLE"

        t = Text()
        t.append("  ◈ ", style="bold #ff00ff")
        t.append("BUDDY", style="bold #00ffcc")
        t.append(f" [{self.mode_label}]", style="bold #7c4dff" if self.mode_label == "DEMO" else "bold #00ff41")
        t.append("  │  ", style="#2a2a4a")
        t.append("NAV ", style="#6666aa")
        t.append(f"${self.nav:,.0f}", style="bold #00ff41")
        t.append("  │  ", style="#2a2a4a")
        t.append("P/L ", style="#6666aa")
        t.append(f"{pnl_sign}${self.pnl:,.0f}", style=f"bold {pnl_color}")
        t.append("  │  ", style="#2a2a4a")
        t.append(f"Open: {self.open_count}", style="#6666aa")
        t.append("  │  ", style="#2a2a4a")
        t.append("OANDA ", style="#6666aa")
        t.append(oanda_text, style=oanda_style)
        t.append("  │  ", style="#2a2a4a")
        t.append("Scanner ", style="#6666aa")
        t.append(scanner_text, style=scanner_style)
        t.append("  │  ", style="#2a2a4a")
        t.append(f"{now}", style="#6666aa")
        return t


class AgentPanel(Static):
    """12 agents with colored bar indicators. Reads from snapshot or demo."""

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
            color = "#00ff41" if s >= 0.7 else "#ffab00" if s >= 0.5 else "#ff1744"
            t.append(f"  {name:<6}", style="#6666aa")
            t.append(" █" * filled, style=color)
            t.append("░" * empty, style="#2a2a4a")
            t.append(f" {s:.2f}", style=f"bold {color}")
            t.append(f"  {signal}", style="#6666aa")
            t.append("\n")

        if self._agents:
            avg = sum(s for _, s, _ in self._agents) / len(self._agents)
            bar_filled = int(avg * 20)
            bar_empty = 20 - bar_filled
            t.append("  ─────────────────────────────\n", style="#2a2a4a")
            t.append("  WEIGHTED ", style="#6666aa")
            t.append(f"{avg:.2f}", style="bold #00ffcc")
            t.append("  [", style="#2a2a4a")
            t.append("█" * bar_filled, style="#00ffcc")
            t.append("░" * bar_empty, style="#2a2a4a")
            t.append("]", style="#2a2a4a")
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

    def _format(self, entry: dict) -> str:
        # Trade ID — keep first 12 chars (human-recognizable prefix)
        tid = str(entry.get("trade_id", "?"))[:12]
        mode = str(entry.get("mode", "?"))
        dur = float(entry.get("duration_seconds", 0) or 0)
        cost = float(entry.get("cost_usd", 0) or 0)
        success = bool(entry.get("success"))
        hyp = str(entry.get("hypothesis") or "").strip()[:70]
        err = str(entry.get("error") or "").strip()[:60]
        # HH:MM — parse the ISO timestamp properly (handles microseconds + TZ)
        ts_raw = str(entry.get("ts", ""))
        try:
            from datetime import datetime as _dt
            ts_obj = _dt.fromisoformat(ts_raw.replace("Z", "+00:00"))
            ts = ts_obj.strftime("%H:%M")
        except (ValueError, TypeError):
            ts = "--:--"

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
                    self._stop.wait(self._poll_interval)
                    continue

                size = self._log_path.stat().st_size
                if size < self._offset:
                    # File was truncated/rotated — reset
                    self._offset = 0
                if size == self._offset:
                    self._stop.wait(self._poll_interval)
                    continue

                # Once we have a real file, clear the waiting flag so a later
                # deletion + recreation re-announces.
                self._announced_waiting = False
                with open(self._log_path, "r") as f:
                    f.seek(self._offset)
                    for raw in f:
                        if not raw.strip():
                            continue
                        try:
                            entry = json.loads(raw)
                        except json.JSONDecodeError:
                            # Skip malformed line, keep consuming
                            continue
                        try:
                            self._callback(self._format(entry))
                        except Exception:
                            pass  # callback errors must not poison reader
                    self._offset = f.tell()
            except Exception as e:
                logger.debug("ReflectionLogReader error: %s", e)
            self._stop.wait(self._poll_interval)


class RiskPanel(Static):
    """Risk metrics panel. Reads from snapshot or uses defaults.

    Defense-in-depth: coerces any NaN/inf/None upstream value to a stable 0.0
    and renders "—" (em-dash) when no real data has arrived yet so the UI
    never shows literal "nan".
    """

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
        rc = "#00ff41" if pr < 10 else "#ffab00" if pr < 13 else "#ff1744"
        dc = "#00ff41" if dd < 3 else "#ffab00" if dd < 5 else "#ff1744"
        corr_text = "OK ✓" if self._corr_ok else "WARN ▲"
        corr_style = "bold #00ff41" if self._corr_ok else "bold #ffab00"

        # Render em-dash placeholder while we're still waiting for first scan.
        def _fmt(val: float) -> str:
            if not self._has_data and val == 0.0:
                return "    —  "
            return f"{val:.1f}%"

        t = Text()
        t.append("  Portfolio Risk: ", style="#6666aa")
        t.append(f"{_fmt(pr)}\n", style=f"bold {rc}")
        t.append("  Drawdown:       ", style="#6666aa")
        t.append(f"{_fmt(dd)}\n", style=f"bold {dc}")
        t.append("  Max DD Today:   ", style="#6666aa")
        t.append(f"{_fmt(mdd)}\n", style="bold #ffab00")
        t.append("  Correlation:    ", style="#6666aa")
        t.append(f"{corr_text}\n", style=corr_style)
        return t


class MTFConfluencePanel(Static):
    """Multi-Timeframe Confluence visualization."""

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
            color = "#00ff41" if score >= 0.7 else "#ffab00" if score >= 0.5 else "#ff1744"
            signal = "BULLISH" if score >= 0.7 else "CAUTION" if score >= 0.5 else "WEAK"
            spark = self._sparkline_str(closes)

            t.append(f"  {label:<10}", style="bold #7c4dff")
            t.append(f"  {spark}", style=color)
            t.append(f"  {score:.2f}", style=f"bold {color}")
            t.append(f"  {signal:<8}", style=color)
            t.append(f"  {indicator}", style="#6666aa")
            t.append("\n")

        conf = self._confluence or sum(s * w for (_, s, _, _), w in zip(screens, weights))
        conf_color = "#00ff41" if conf >= 0.65 else "#ffab00" if conf >= 0.45 else "#ff1744"
        conf_signal = "PROCEED ✓" if conf >= 0.65 else "CAUTION ▲" if conf >= 0.45 else "REJECT ✗"
        bar_filled = int(conf * 25)
        bar_empty = 25 - bar_filled

        t.append("  ──────────────────────────────────────────────────────\n", style="#2a2a4a")
        t.append("  CONFLUENCE ", style="#6666aa")
        t.append(f"{conf:.2f}", style=f"bold {conf_color}")
        t.append("  [", style="#2a2a4a")
        t.append("█" * bar_filled, style=conf_color)
        t.append("░" * bar_empty, style="#2a2a4a")
        t.append("]  ", style="#2a2a4a")
        t.append(conf_signal, style=f"bold {conf_color}")
        return t


class SystemHealthBar(Static):
    """Bottom status bar with system health metrics."""

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
        oanda_style = "bold #00ff41" if oanda_ok else "bold #ff1744"

        t = Text()
        t.append("  Scanner: ", style="#6666aa")
        t.append(f"● {scan_ms:.0f}ms", style="bold #00ff41")
        t.append("  │  ", style="#2a2a4a")
        t.append("OANDA: ", style="#6666aa")
        t.append(f"● {api_ms:.0f}ms", style=oanda_style)
        t.append("  │  ", style="#2a2a4a")
        t.append("Models: ", style="#6666aa")
        t.append(f"● {models}", style="bold #00ff41")
        t.append("  │  ", style="#2a2a4a")
        t.append("RL Sync: ", style="#6666aa")
        t.append("● CURRENT", style="bold #00ff41")
        t.append("  │  ", style="#2a2a4a")
        t.append(f"CPU: {cpu:.0f}%", style="#6666aa")
        t.append("  │  ", style="#2a2a4a")
        t.append(f"MEM: {mem:.0f}MB", style="#6666aa")
        return t


class PlaceholderContent(Static):
    DEFAULT_CSS = "PlaceholderContent { align: center middle; height: 1fr; }"

    def __init__(self, screen_name: str) -> None:
        super().__init__()
        self.screen_name = screen_name

    def render(self) -> Text:
        t = Text(justify="center")
        t.append("\n\n")
        t.append("╔══════════════════════════════════╗\n", style="#2a2a4a")
        t.append(f"║  {self.screen_name:^30}  ║\n", style="bold #7c4dff")
        t.append("║     ◈ COMING SOON ◈              ║\n", style="#ff00ff")
        t.append("║  Phase 2-5 of Command Bridge     ║\n", style="#6666aa")
        t.append("╚══════════════════════════════════╝\n", style="#2a2a4a")
        return t


# ── Main App ────────────────────────────────────────────────────────


class BuddyApp(App):
    """BUDDY — Cyberpunk Command Bridge."""

    CSS_PATH = "theme.tcss"
    TITLE = "BUDDY"

    BINDINGS = [
        Binding("f1", "switch_tab('overview')", "Overview", show=True),
        Binding("f2", "switch_tab('trades')", "Trades", show=True),
        Binding("f3", "switch_tab('agents')", "Agents", show=True),
        Binding("f4", "switch_tab('journal')", "Journal", show=True),
        Binding("f5", "switch_tab('config')", "Config", show=True),
        Binding("f6", "switch_tab('diag')", "Diagnostics", show=True),
        Binding("q", "quit", "Quit", show=True),
    ]

    def __init__(self, live: bool = False, **kwargs) -> None:
        super().__init__(**kwargs)
        self._live = live
        self._provider = DataProvider(project_root=str(Path(__file__).resolve().parent.parent.parent))
        self._demo_nav = 101420.0
        self._scanner = None  # EmbeddedScanner (live mode only)
        self._reflection_stop = threading.Event()  # Signals ReflectionLogReader to exit

    def compose(self) -> ComposeResult:
        yield HeaderBar(id="header-bar")

        with TabbedContent(id="main-tabs"):
            with TabPane("◈ Overview", id="overview"):
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
                            yield RichLog(id="brain-log", highlight=True, markup=True,
                                         max_lines=500, auto_scroll=True)

            with TabPane("◈ Trades", id="trades"):
                yield TradesScreen(id="trades-screen")
            with TabPane("◈ Agents", id="agents"):
                yield AgentsScreen(id="agents-screen")
            with TabPane("◈ Journal", id="journal"):
                yield JournalScreen(
                    project_root=str(Path(__file__).resolve().parent.parent.parent),
                    id="journal-screen",
                )
            with TabPane("◈ Config", id="config"):
                if ConfigScreen is not None:
                    yield ConfigScreen(
                        project_root=str(Path(__file__).resolve().parent.parent.parent),
                        id="config-screen",
                    )
                else:
                    yield PlaceholderContent("CONFIG — The Tuning Bench")
            with TabPane("◈ Diagnostics", id="diag"):
                if DiagnosticsScreen is not None:
                    yield DiagnosticsScreen(
                        project_root=str(Path(__file__).resolve().parent.parent.parent),
                        live=self._live,
                        id="diag-screen",
                    )
                else:
                    yield PlaceholderContent("DIAGNOSTICS — The Engine Room")

        yield SystemHealthBar(id="system-health")
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

    def on_unmount(self) -> None:
        """Signal background readers to exit cleanly."""
        try:
            self._reflection_stop.set()
        except Exception:
            pass

    def _write_boot_sequence(self) -> None:
        log = self.query_one("#brain-log", RichLog)
        header = self.query_one("#header-bar", HeaderBar)
        now = datetime.now(timezone.utc)
        mode = "LIVE" if self._live else "DEMO"
        header.mode_label = mode

        msgs = [
            f"[bold #ff00ff]◈ BUDDY COMMAND BRIDGE — {mode} MODE[/]",
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
        """Safely write to brain log (must be called on main thread)."""
        self.query_one("#brain-log", RichLog).write(Text.from_markup(markup))

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

        project_root = str(Path(__file__).resolve().parent.parent.parent)
        self._scanner = EmbeddedScanner(
            project_root=project_root,
            brain_callback=self._scanner_brain_bridge,
            auto_execute=True,
            interval_minutes=5,
        )
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
        project_root = Path(__file__).resolve().parent.parent.parent
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
        header.nav = snap.nav if snap.nav > 0 else self._demo_nav
        header.pnl = snap.unrealized_pnl if snap.nav > 0 else (self._demo_nav - 100000)
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

        # Update Trades screen (F2)
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

        # Update Diagnostics screen (F6)
        try:
            diag_screen = self.query_one("#diag-screen", DiagnosticsScreen)
            diag_screen.update_from_snapshot(snap)
        except Exception:
            pass

    @work(thread=True)
    def _do_live_refresh(self) -> None:
        """Run data refresh in background thread."""
        snap = self._provider.refresh()
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
        """Add a line to the brain stream."""
        log = self.query_one("#brain-log", RichLog)
        now = datetime.now(timezone.utc).strftime("%H:%M:%S")
        line = _fake_brain_line()
        log.write(Text.from_markup(f"[dim]{now}[/] {line}"))

    def action_switch_tab(self, tab_id: str) -> None:
        tabs = self.query_one("#main-tabs", TabbedContent)
        tabs.active = tab_id

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

    # Default: LIVE always. Only demo when explicitly requested.
    live = not args.demo

    app = BuddyApp(live=live)
    app.run()


if __name__ == "__main__":
    run()

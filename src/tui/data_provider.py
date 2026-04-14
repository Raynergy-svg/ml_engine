"""
Data Provider — Bridge between Buddy's trading systems and the TUI.

Wraps all data sources (OANDA broker, Scanner, Orchestrator) and provides
a thread-safe DashboardSnapshot that the Textual app polls periodically.
"""
from __future__ import annotations

import collections
import json
import logging
import math
import os
import threading
import time
from dataclasses import dataclass, field


def _safe_float(v, default: float = 0.0) -> float:
    """Coerce v to a finite float; fall back to default for None/NaN/inf/bad input.

    Prevents UI from ever rendering literal "nan" or "inf" in numeric fields.
    """
    if v is None:
        return default
    try:
        fv = float(v)
    except (TypeError, ValueError):
        return default
    if math.isnan(fv) or math.isinf(fv):
        return default
    return fv


def _is_valid_risk(v) -> bool:
    """True if v is a usable risk metric (finite, >= 0).

    The embedded scanner uses -1.0 as a sentinel meaning "couldn't compute,
    don't overwrite last-good value".
    """
    try:
        fv = float(v)
    except (TypeError, ValueError):
        return False
    return not (math.isnan(fv) or math.isinf(fv)) and fv >= 0.0
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class TradeRow:
    """Simplified trade for display in the TUI DataTable."""
    pair: str
    direction: str
    pnl_pips: float
    entry_price: float
    sl_price: float
    tp_price: float
    quantity: float
    trade_id: str = ""


@dataclass
class AgentScore:
    """Single agent's score and signal for display."""
    name: str
    score: float
    signal: str
    weight: float = 1.0


@dataclass
class DashboardSnapshot:
    """Immutable snapshot of all dashboard data at a point in time."""

    # Account
    nav: float = 0.0
    balance: float = 0.0
    unrealized_pnl: float = 0.0
    margin_used: float = 0.0

    # Open Trades
    trades: list[TradeRow] = field(default_factory=list)

    # Agent Scores (from last scan)
    agents: list[AgentScore] = field(default_factory=list)
    weighted_vote_score: float = 0.0

    # Risk Metrics
    portfolio_risk_pct: float = 0.0
    drawdown_pct: float = 0.0
    max_drawdown_pct: float = 0.0
    correlation_ok: bool = True

    # MTF Candle Close Prices (for sparklines)
    candles_h4: list[float] = field(default_factory=list)
    candles_h1: list[float] = field(default_factory=list)
    candles_m15: list[float] = field(default_factory=list)
    mtf_confluence_score: float = 0.0
    mtf_h4_score: float = 0.0
    mtf_h1_score: float = 0.0
    mtf_m15_score: float = 0.0
    mtf_h4_signal: str = ""
    mtf_h1_signal: str = ""
    mtf_m15_signal: str = ""

    # NAV history for sparkline
    nav_history: list[float] = field(default_factory=list)

    # System Health
    oanda_connected: bool = False
    oanda_latency_ms: float = 0.0
    scanner_ready: bool = False
    scan_cycle_count: int = 0
    scan_duration_ms: float = 0.0
    models_loaded: int = 0
    models_total: int = 0  # set by scanner, no longer hardcoded
    models_detail: dict = field(default_factory=dict)
    momentum_model_type: str = "none"
    system_health_score: float = 0.0
    cpu_pct: float = 0.0
    mem_mb: float = 0.0

    # Scan results (last scan)
    last_scan_time: datetime | None = None
    tradeable_count: int = 0
    scanned_count: int = 0

    # Errors
    last_error: str = ""

    # Timestamp
    last_refresh: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class DataProvider:
    """Thread-safe data provider that bridges sync broker/scanner calls
    to the async Textual event loop.

    Usage:
        provider = DataProvider()
        provider.connect()  # Call once at startup
        provider.refresh()  # Call periodically from a Worker
        snap = provider.snapshot  # Read from main thread (immutable)
    """

    def __init__(self, project_root: str | None = None) -> None:
        self._project_root = Path(project_root or os.getcwd())
        self._broker = None
        self._lock = threading.Lock()
        self._snapshot = DashboardSnapshot()
        self._nav_history: collections.deque[float] = collections.deque(maxlen=60)
        self._connected = False
        self._scan_count = 0
        # Scan enrichment overlay (set by EmbeddedScanner after each cycle)
        self._scan_enrichment = None  # ScanEnrichment | None

    @property
    def snapshot(self) -> DashboardSnapshot:
        """Return the latest snapshot (thread-safe read)."""
        with self._lock:
            return self._snapshot

    def apply_scan_enrichment(self, enrichment) -> None:
        """Store scan-derived enrichment data (thread-safe).

        Called by EmbeddedScanner after each scan cycle completes.
        The enrichment is overlaid onto the next refresh() snapshot,
        replacing stale file-based defaults for MTF scores, agents, etc.
        """
        with self._lock:
            self._scan_enrichment = enrichment

    def connect(self) -> bool:
        """Initialize the OANDA broker connection.

        Returns True if connection succeeded, False otherwise.
        Does NOT raise — all errors are caught and logged.
        """
        try:
            from src.brokers.oanda import OandaBroker
            self._broker = OandaBroker.from_env()
            self._broker.connect()

            # Test connection by fetching NAV
            nav = self._broker.get_nav()
            logger.info(f"OANDA connected — NAV: ${nav:,.2f}")
            self._connected = True
            return True
        except Exception as e:
            logger.warning(f"OANDA connection failed: {e}")
            self._connected = False
            return False

    def refresh(self) -> DashboardSnapshot:
        """Refresh all data from live sources.

        This runs in a background Worker thread. All broker/scanner calls
        happen here. The result is stored as an immutable snapshot.

        Returns the new snapshot.
        """
        snap = DashboardSnapshot()
        snap.oanda_connected = self._connected

        # ── Account Data ────────────────────────────────────────
        if self._broker and self._connected:
            snap = self._refresh_account(snap)
            snap = self._refresh_trades(snap)
            snap = self._refresh_candles(snap)

        # ── System Metrics ──────────────────────────────────────
        snap = self._refresh_system(snap)

        # ── Last Scan Results ───────────────────────────────────
        snap = self._refresh_last_scan(snap)

        # ── Scan Enrichment Overlay (from EmbeddedScanner) ─────
        # Overrides stale file-based data with real scan results
        enrichment = self._scan_enrichment  # atomic read under GIL
        if enrichment is not None:
            snap.scanner_ready = enrichment.scanner_ready
            snap.scan_duration_ms = enrichment.scan_duration_ms
            snap.scan_cycle_count = enrichment.scan_cycle_count
            snap.tradeable_count = enrichment.tradeable_count
            snap.scanned_count = enrichment.scanned_count
            snap.last_scan_time = enrichment.last_scan_time
            snap.mtf_h4_score = enrichment.mtf_h4_score
            snap.mtf_h1_score = enrichment.mtf_h1_score
            snap.mtf_m15_score = enrichment.mtf_m15_score
            snap.mtf_h4_signal = enrichment.mtf_h4_signal
            snap.mtf_h1_signal = enrichment.mtf_h1_signal
            snap.mtf_m15_signal = enrichment.mtf_m15_signal
            snap.mtf_confluence_score = enrichment.mtf_confluence_score
            # Sentinel: -1.0 means the scanner couldn't compute (account
            # fetch failed or exception). Keep the last-good snapshot value
            # instead of overwriting with a misleading 0.0 or NaN.
            if _is_valid_risk(enrichment.portfolio_risk_pct):
                snap.portfolio_risk_pct = _safe_float(enrichment.portfolio_risk_pct)
            if _is_valid_risk(enrichment.drawdown_pct):
                snap.drawdown_pct = _safe_float(enrichment.drawdown_pct)
            if _is_valid_risk(enrichment.max_drawdown_pct):
                snap.max_drawdown_pct = _safe_float(enrichment.max_drawdown_pct)
            snap.correlation_ok = enrichment.correlation_ok
            if enrichment.agents:
                snap.agents = enrichment.agents
                snap.weighted_vote_score = enrichment.weighted_vote_score
            # Real model health count from Scanner.get_model_health()
            if getattr(enrichment, "models_total", 0) > 0:
                snap.models_loaded = int(enrichment.models_loaded_count)
                snap.models_total = int(enrichment.models_total)
                snap.models_detail = dict(enrichment.models_detail or {})
                snap.momentum_model_type = str(enrichment.momentum_model_type or "none")

        # ── NAV History (deque is thread-safe + auto-bounded) ──
        if snap.nav > 0:
            self._nav_history.append(snap.nav)
        snap.nav_history = list(self._nav_history)

        snap.last_refresh = datetime.now(timezone.utc)

        with self._lock:
            self._snapshot = snap

        return snap

    def _refresh_account(self, snap: DashboardSnapshot) -> DashboardSnapshot:
        """Fetch account summary from OANDA."""
        try:
            t0 = time.monotonic()
            summary = self._broker.get_account_summary()
            latency = (time.monotonic() - t0) * 1000

            snap.nav = summary.nav
            snap.balance = summary.balance
            snap.unrealized_pnl = summary.unrealized_pnl
            snap.margin_used = summary.margin_used
            snap.oanda_latency_ms = latency
            snap.oanda_connected = True
        except Exception as e:
            logger.warning(f"Account refresh failed: {e}")
            snap.last_error = f"Account: {e}"
            snap.oanda_connected = False
        return snap

    def _refresh_trades(self, snap: DashboardSnapshot) -> DashboardSnapshot:
        """Fetch open trades from OANDA (only OPEN, not closed history)."""
        try:
            # Get open positions for unrealized P/L lookup
            positions = self._broker.get_open_positions()
            pnl_by_instrument = {p.instrument: p.unrealized_pnl for p in positions}

            # Fetch only OPEN trades — guard private _client access
            client = getattr(self._broker, '_client', None)
            if client is None:
                logger.warning("Broker client not initialized, skipping trade refresh")
                return snap
            raw_trades = client.get_trades(state="OPEN", count=50)
            trades = []
            for trade_dict in raw_trades.get("trades", []):
                instrument = trade_dict.get("instrument", "")
                units = float(trade_dict.get("currentUnits", trade_dict.get("initialUnits", 0)))
                direction = "BUY" if units > 0 else "SELL"
                entry = float(trade_dict.get("price", 0))
                sl_data = trade_dict.get("stopLossOrder", {})
                tp_data = trade_dict.get("takeProfitOrder", {})
                sl = float(sl_data.get("price", 0)) if sl_data else 0.0
                tp = float(tp_data.get("price", 0)) if tp_data else 0.0
                pnl = float(trade_dict.get("unrealizedPL", 0))

                trades.append(TradeRow(
                    pair=instrument.replace("_", "/"),
                    direction=direction,
                    pnl_pips=pnl,
                    entry_price=entry,
                    sl_price=sl,
                    tp_price=tp,
                    quantity=abs(units),
                    trade_id=str(trade_dict.get("id", "")),
                ))
            snap.trades = trades
        except Exception as e:
            logger.warning(f"Trades refresh failed: {e}")
            snap.last_error = f"Trades: {e}"
        return snap

    def _refresh_candles(self, snap: DashboardSnapshot) -> DashboardSnapshot:
        """Fetch MTF candle data for sparklines."""
        try:
            from src.brokers.instrument import Instrument

            # Use first traded pair or default to EUR_USD
            pair_symbol = "EUR_USD"
            if snap.trades:
                pair_symbol = snap.trades[0].pair.replace("/", "_")

            instr = Instrument.fx(pair_symbol, pair_symbol, pip_value=10.0)

            for gran, attr in [("H4", "candles_h4"), ("H1", "candles_h1"), ("M15", "candles_m15")]:
                try:
                    candles = self._broker.fetch_candles(instr, gran, 30)
                    closes = [c.close for c in candles]
                    setattr(snap, attr, closes)
                except Exception as e:
                    logger.debug(f"Candle fetch {gran} failed: {e}")
        except Exception as e:
            logger.warning(f"Candle refresh failed: {e}")
        return snap

    def _refresh_system(self, snap: DashboardSnapshot) -> DashboardSnapshot:
        """Gather system health metrics."""
        try:
            import psutil
            proc = psutil.Process()
            snap.cpu_pct = proc.cpu_percent(interval=None)  # Non-blocking (since last call)
            snap.mem_mb = proc.memory_info().rss / (1024 * 1024)
        except ImportError:
            snap.cpu_pct = 0.0
            snap.mem_mb = 0.0
        except Exception:
            pass

        # Model count is now populated by ScanEnrichment from the real
        # Scanner.get_model_health() call (see apply_scan_enrichment below).
        # We only set a file-system fallback here so the header shows a
        # reasonable number before the first scan completes.
        models_dir = self._project_root / "trained_data" / "models"
        if models_dir.exists() and snap.models_total == 0:
            # Rough pre-scan estimate: count loadable model files (not hardcoded)
            model_files = (
                list(models_dir.glob("*.pkl"))
                + list(models_dir.glob("*.joblib"))
                + list(models_dir.glob("*.keras"))
            )
            snap.models_loaded = len(model_files)
            # Real total gets set from scanner.get_model_health() on first cycle
            snap.models_total = len(model_files)

        # Check scanner state
        state_path = self._project_root / ".claude" / "state.json"
        if state_path.exists():
            try:
                state = json.loads(state_path.read_text())
                snap.scan_cycle_count = state.get("scan_cycles", 0)
                snap.scanner_ready = True
            except Exception:
                pass

        return snap

    def _refresh_last_scan(self, snap: DashboardSnapshot) -> DashboardSnapshot:
        """Load agent scores and scan results from last scan state."""
        # Try to load agent weights
        weights_path = self._project_root / "trained_data" / "models" / "agent_weights.json"
        base_weights = {
            "trend": 1.15, "mean_reversion": 0.90, "volatility": 1.00,
            "risk_sentinel": 1.25, "uncertainty": 1.10, "execution_quality": 1.05,
            "momentum": 1.05, "news_risk": 0.95, "multi_timeframe": 1.10,
            "pair_performance": 0.85, "session_timing": 0.80,
            "support_resistance": 1.00,
        }

        learned = {}
        if weights_path.exists():
            try:
                data = json.loads(weights_path.read_text())
                learned = data.get("_global", data.get("NORMAL", {}))
            except Exception:
                pass

        # Build agent display data from weights
        agent_display_names = {
            "trend": ("trend", "TREND"),
            "mean_reversion": ("m_rev", "M.REV"),
            "volatility": ("volat", "VOL"),
            "risk_sentinel": ("risk", "RISK"),
            "uncertainty": ("uncrt", "UNC"),
            "execution_quality": ("exec", "EXEC"),
            "momentum": ("momt", "MOM"),
            "news_risk": ("news", "NEWS"),
            "multi_timeframe": ("mtf", "MTF"),
            "pair_performance": ("pair", "PAIR"),
            "session_timing": ("sess", "SESS"),
            "support_resistance": ("sr", "S/R"),
        }

        agents = []
        for agent_key, (short_name, label) in agent_display_names.items():
            weight = learned.get(agent_key, base_weights.get(agent_key, 1.0))
            # Normalize weight to 0-1 score for display (weights range ~0.5-1.5)
            score = min(1.0, max(0.0, weight / 1.5))
            signal = "HIGH" if score >= 0.7 else "MED" if score >= 0.4 else "LOW"
            agents.append(AgentScore(
                name=short_name,
                score=score,
                signal=signal,
                weight=weight,
            ))
        snap.agents = agents

        if agents:
            snap.weighted_vote_score = sum(a.score for a in agents) / len(agents)

        # Load trade journal for risk/drawdown approximation
        journal_path = self._project_root / "trained_data" / "trade_journal_rl.json"
        if journal_path.exists():
            try:
                entries = json.loads(journal_path.read_text())
                if isinstance(entries, list) and entries:
                    # Calculate basic stats from recent trades
                    recent = entries[-20:]
                    wins = sum(1 for e in recent
                               if e.get("outcome", {}).get("trade_won", False))
                    snap.tradeable_count = len(recent)

                    # Rough drawdown from recent P/L
                    pnls = [e.get("outcome", {}).get("realized_pl", 0) for e in recent]
                    cumulative = []
                    running = 0
                    for p in pnls:
                        running += p
                        cumulative.append(running)
                    if cumulative:
                        peak = max(cumulative)
                        trough = min(cumulative)
                        if snap.nav > 0:
                            snap.drawdown_pct = abs(trough) / snap.nav * 100 if trough < 0 else 0
                            snap.max_drawdown_pct = abs(peak - trough) / snap.nav * 100
            except Exception:
                pass

        return snap

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
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


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


logger = logging.getLogger(__name__)


# US-011: backoff schedule for OANDA auto-reconnect on transient disconnect.
# Schedule: 5s after 1st failed attempt, 10s after 2nd, 30s after 3rd, then
# 60s capped indefinitely. Reset to 0 on successful reconnect.
_RECONNECT_BACKOFF_SCHEDULE: tuple[int, ...] = (5, 10, 30, 60)
_RECONNECT_BACKOFF_CAP_SECONDS: int = 60
_BRAIN_FEED_MAX_BYTES: int = 5 * 1024 * 1024


def _next_reconnect_delay_seconds(attempts_so_far: int) -> int:
    """Delay (in seconds) before the next reconnect attempt.

    ``attempts_so_far`` is the count of FAILED attempts INCLUDING the one
    that just failed. Schedule produces 5s/10s/30s/60s for attempts 1-4
    and clamps at 60s for attempt 5+.
    """
    if attempts_so_far <= 0:
        return _RECONNECT_BACKOFF_SCHEDULE[0]
    idx = min(attempts_so_far - 1, len(_RECONNECT_BACKOFF_SCHEDULE) - 1)
    return _RECONNECT_BACKOFF_SCHEDULE[idx]


# ── Equity-harvester state-file paths (relative to project root) ───────
# Exact contracts (verified from src/equity/*.py, 2026-06-22):
#   SHIP_GATE.json     — src.equity.ship_gate / control_loop SHIP_GATE_PATH_DEFAULT
#   live_gate_state    — src.equity.live_gate.STATE_PATH_DEFAULT (canonical)
#   loop/portfolio     — control_loop state_path/portfolio_state_path are
#                        configuration-time decisions with no entry point yet;
#                        we import the canonical defaults from control_loop so
#                        the loop launcher and this reader share one source of
#                        truth, and degrade gracefully when the file is absent.
from src.equity.control_loop import (  # noqa: E402
    LOOP_STATE_PATH_DEFAULT as _EQUITY_LOOP_STATE_PATH,
    PORTFOLIO_STATE_PATH_DEFAULT as _EQUITY_PORTFOLIO_STATE_PATH,
    SHIP_GATE_PATH_DEFAULT as _EQUITY_SHIP_GATE_PATH,
)
from src.equity.live_gate import (  # noqa: E402
    STATE_PATH_DEFAULT as _EQUITY_LIVE_GATE_STATE_PATH,
)
from src.equity.rebalance import (  # noqa: E402
    STATE_PATH_DEFAULT as _EQUITY_REBALANCE_STATE_PATH,
)
from src.equity.alerts import (  # noqa: E402
    AUDIT_DIR_DEFAULT as _EQUITY_ALERTS_AUDIT_DIR,
    STATE_PATH_DEFAULT as _EQUITY_ALERTS_STATE_PATH,
)


def _read_equity_json(path: Path) -> dict | None:
    """Read + parse an equity state file, returning None on any problem.

    Honours the project JSON-safety rule: read in try/except, validate the
    parsed payload is a dict, never raise. A missing file (loop not running
    yet) and a corrupt file both return None — callers render a graceful
    "pending / not running" state in both cases.
    """
    try:
        if not path.exists():
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        logger.warning("equity state read failed for %s: %s", path, exc)
        return None
    if not isinstance(payload, dict):
        logger.warning("equity state at %s is not a JSON object", path)
        return None
    return payload


def _emit_brain_feed_line(project_root: Path, msg: str) -> None:
    """Append ``msg`` to ``<project_root>/.claude/brain/feed.jsonl`` as JSONL.

    Mirrors :func:`src.tui.app._tee_brain_to_disk` for background threads.
    DataProvider runs in a Worker thread, so we can't reuse the TUI helper
    (which assumes Textual app context). Best-effort: any OSError is
    swallowed so a missing/full disk never breaks the refresh loop.
    """
    feed_path = project_root / ".claude" / "brain" / "feed.jsonl"
    try:
        feed_path.parent.mkdir(parents=True, exist_ok=True)
    except OSError:
        return

    try:
        if feed_path.exists() and feed_path.stat().st_size > _BRAIN_FEED_MAX_BYTES:
            rotated = feed_path.with_suffix(".jsonl.1")
            try:
                rotated.unlink()
            except FileNotFoundError:
                pass
            feed_path.rename(rotated)
    except OSError:
        pass

    line = json.dumps(
        {
            "ts": datetime.now(timezone.utc).isoformat(),
            "msg": msg,
            "raw": msg,
        },
        ensure_ascii=False,
    )
    try:
        with feed_path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
    except OSError:
        return


def _apply_atr_to_trades(
    trades: "list[TradeRow]",
    atr_by_instrument: "dict[str, float] | None",
) -> None:
    """Pipe per-pair ATR (pips) onto live trades.

    US-002: TradeRow.pair is OANDA-display style ("EUR/USD"); ScanResult
    keys are OANDA-instrument style ("EUR_USD"). Normalise on lookup so
    both forms hit the same dict entry. Non-positive / missing values
    leave trade.live_atr_pips untouched so the drill-down keeps showing
    "—" rather than a misleading 0.0.
    """
    if not atr_by_instrument or not trades:
        return
    lookup = dict(atr_by_instrument)
    for tr in trades:
        key = (tr.pair or "").replace("/", "_")
        atr = lookup.get(key)
        if atr is None:
            continue
        try:
            atr_f = float(atr)
        except (TypeError, ValueError):
            continue
        if atr_f > 0.0:
            tr.live_atr_pips = atr_f


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
    # Live OANDA spread in pips for this instrument. None when the pricing
    # call hasn't succeeded yet (cold start, network failure, OANDA error).
    # The drill-down renders "—" instead of a synthetic placeholder when None
    # — never substitute a fake value, the operator's flatten decision may
    # depend on this.
    live_spread_pips: float | None = None
    # Latest scanner-computed ATR (pips) for this instrument. Populated
    # in refresh() from ScanEnrichment.atr_value (per-pair dict, last
    # successful scan). None when the scanner hasn't reported ATR for the
    # instrument yet — drill-down renders "—" (US-002: NEVER back-derive
    # ATR from SL distance; SL is derived from ATR upstream, so the
    # circular fallback hid model behaviour).
    live_atr_pips: float | None = None


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

    # Supervisor state (US-504)
    scanner_paused: bool = False
    halted: bool = False
    mode: str = "live"
    max_component_age_days: float = 0.0
    config_profile: str = "smart"
    config_values: dict = field(default_factory=dict)

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
        # ``OandaBroker`` import is deferred to keep test imports cheap and to
        # avoid a hard dependency on the broker module at TUI startup. ``Any``
        # is the honest annotation here — the field carries either ``None`` or
        # a real :class:`OandaBroker` instance after :meth:`connect`.
        self._broker: Any = None
        self._lock = threading.Lock()
        self._snapshot = DashboardSnapshot()
        self._nav_history: collections.deque[float] = collections.deque(maxlen=60)
        self._connected = False
        self._scan_count = 0
        # Scan enrichment overlay (set by EmbeddedScanner after each cycle)
        self._scan_enrichment = None  # ScanEnrichment | None
        # US-011: auto-reconnect state. ``_reconnect_attempts`` counts failed
        # tries since the last successful connect; ``_next_reconnect_at`` gates
        # the retry cadence (refresh() refuses to retry before this deadline).
        self._reconnect_attempts: int = 0
        self._next_reconnect_at: datetime = datetime.now(timezone.utc)

    @property
    def snapshot(self) -> DashboardSnapshot:
        """Return the latest snapshot (thread-safe read)."""
        with self._lock:
            return self._snapshot

    @property
    def oanda_connected(self) -> bool:
        """Whether the OANDA broker is currently flagged connected.

        Exposed as the public contract surface for US-011 — flips False when
        a refresh-path API call raises (so the next refresh enters the
        reconnect-with-backoff branch) and True on a verified reconnect.
        """
        return self._connected

    def get_snapshot(self) -> DashboardSnapshot:
        """Compatibility accessor for call sites that predate the property."""
        return self.snapshot

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
            # US-011: clean slate for the backoff state on successful connect.
            self._reconnect_attempts = 0
            self._next_reconnect_at = datetime.now(timezone.utc)
            return True
        except Exception as e:
            logger.warning(f"OANDA connection failed: {e}")
            self._connected = False
            return False

    def _maybe_attempt_reconnect(self) -> None:
        """US-011: try to revive a dead OANDA connection with backoff.

        Called at the top of ``refresh()``. When the broker is flagged
        disconnected AND ``now() >= self._next_reconnect_at``, attempts a
        verified reconnect (``connect()`` + ``get_nav()``). On success,
        clears attempt counter and emits an ``OANDA reconnected`` INFO line.
        On failure, increments the counter, schedules the next attempt per
        the backoff schedule (5s, 10s, 30s, 60s, 60s ...), and emits an INFO
        line carrying the attempt number and next-retry delay.
        """
        if self._connected:
            return
        now = datetime.now(timezone.utc)
        if now < self._next_reconnect_at:
            return

        attempt = self._reconnect_attempts + 1
        try:
            if self._broker is None:
                from src.brokers.oanda import OandaBroker
                self._broker = OandaBroker.from_env()
            self._broker.connect()
            # connect() is a no-op on the REST broker — verify with a real call.
            nav = self._broker.get_nav()
        except Exception as e:
            self._reconnect_attempts = attempt
            delay = _next_reconnect_delay_seconds(attempt)
            self._next_reconnect_at = now + timedelta(seconds=delay)
            msg = (
                f"OANDA reconnect attempt {attempt} failed: {e}; "
                f"retry in {delay}s"
            )
            logger.info(msg)
            _emit_brain_feed_line(self._project_root, msg)
            return

        # Success.
        self._connected = True
        self._reconnect_attempts = 0
        self._next_reconnect_at = now
        attempt_msg = (
            f"OANDA reconnect attempt {attempt} succeeded (next delay 0s)"
        )
        logger.info(attempt_msg)
        _emit_brain_feed_line(self._project_root, attempt_msg)
        success_msg = f"OANDA reconnected — NAV: ${nav:,.2f}"
        logger.info(success_msg)
        _emit_brain_feed_line(self._project_root, success_msg)

    def refresh(self) -> DashboardSnapshot:
        """Refresh all data from live sources.

        This runs in a background Worker thread. All broker/scanner calls
        happen here. The result is stored as an immutable snapshot.

        Returns the new snapshot.
        """
        # US-011: try to revive a dead OANDA connection BEFORE we use it.
        # No-op when self._connected is True or the backoff gate hasn't elapsed.
        self._maybe_attempt_reconnect()

        with self._lock:
            previous = self._snapshot

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
            else:
                snap.portfolio_risk_pct = previous.portfolio_risk_pct
            if _is_valid_risk(enrichment.drawdown_pct):
                snap.drawdown_pct = _safe_float(enrichment.drawdown_pct)
            else:
                snap.drawdown_pct = previous.drawdown_pct
            if _is_valid_risk(enrichment.max_drawdown_pct):
                snap.max_drawdown_pct = _safe_float(enrichment.max_drawdown_pct)
            else:
                snap.max_drawdown_pct = previous.max_drawdown_pct
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
            # Model staleness (days since oldest ensemble component was trained)
            age = getattr(enrichment, "max_component_age_days", None)
            if age is not None:
                snap.max_component_age_days = _safe_float(age)
            snap.config_profile = str(
                getattr(enrichment, "config_profile", snap.config_profile)
                or snap.config_profile
            )
            snap.config_values = dict(getattr(enrichment, "config_values", {}) or {})

            # US-002: pipe per-pair ATR onto the broker-built trades so the
            # drill-down can read trade.live_atr_pips.
            _apply_atr_to_trades(
                snap.trades,
                getattr(enrichment, "atr_value", None),
            )

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
            # US-011: a failed account call is the signal that the broker
            # client is dead. Flip the connection flag so the next refresh()
            # enters the reconnect-with-backoff branch instead of pointlessly
            # re-calling get_account_summary() against the same dead client.
            self._connected = False
        return snap

    def _refresh_trades(self, snap: DashboardSnapshot) -> DashboardSnapshot:
        """Fetch open trades from OANDA (only OPEN, not closed history)."""
        try:
            # Touch open positions (kept for connection liveness; the per-trade
            # unrealized P/L below comes from the raw trade dicts, not this call)
            self._broker.get_open_positions()

            # Fetch only OPEN trades — guard private _client access
            client = getattr(self._broker, '_client', None)
            if client is None:
                logger.warning("Broker client not initialized, skipping trade refresh")
                return snap
            raw_trades = client.get_trades(state="OPEN", count=50)
            open_dicts = raw_trades.get("trades", [])

            # Fetch live spreads for every open instrument in ONE pricing call
            # so the drill-down panel shows real bid/ask spread (NOT a random
            # placeholder). pip_size is 0.01 for JPY pairs, 0.0001 otherwise.
            # Underlying _request already enforces (5s connect, 30s read)
            # timeout + exponential backoff on 5xx/timeout via _should_retry.
            instruments_set: set[str] = {
                str(d.get("instrument", ""))
                for d in open_dicts
                if d.get("instrument")
            }
            spreads_by_instrument: dict[str, float] = (
                self._fetch_live_spreads_pips(client, instruments_set)
            )

            trades = []
            for trade_dict in open_dicts:
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
                    live_spread_pips=spreads_by_instrument.get(instrument),
                ))
            snap.trades = trades
        except Exception as e:
            logger.warning(f"Trades refresh failed: {e}")
            snap.last_error = f"Trades: {e}"
        return snap

    @staticmethod
    def _fetch_live_spreads_pips(
        client: Any, instruments: set[str]
    ) -> dict[str, float]:
        """Fetch live bid/ask spreads (in pips) for the given OANDA instruments.

        Uses a single /v3/accounts/{id}/pricing call (comma-separated
        instruments). The underlying _request enforces (5s connect, 30s read)
        timeout and 3-attempt exponential backoff on 5xx + timeouts already.

        Returns a dict keyed by OANDA instrument symbol (e.g. ``"EUR_USD"``).
        Instruments that fail to parse are simply omitted — callers must
        treat a missing key as "no live spread available" and render the
        TUI placeholder (NEVER substitute a fake number).
        """
        if not instruments:
            return {}

        try:
            payload = client.get_pricing(
                instruments=",".join(sorted(instruments))
            )
        except Exception as e:
            logger.warning(f"OANDA pricing fetch failed: {e}")
            return {}

        result: dict[str, float] = {}
        prices = (payload or {}).get("prices") or []
        if not isinstance(prices, list):
            return {}

        for price in prices:
            if not isinstance(price, dict):
                continue
            instrument = str(price.get("instrument", ""))
            if not instrument:
                continue
            try:
                bids = price.get("bids") or []
                asks = price.get("asks") or []
                if not bids or not asks:
                    continue
                bid = float(bids[0].get("price", 0.0))
                ask = float(asks[0].get("price", 0.0))
            except (TypeError, ValueError, AttributeError, IndexError):
                continue

            if bid <= 0 or ask <= 0 or ask < bid:
                continue

            # JPY pairs use pip_size = 0.01; all others use 0.0001.
            pip_size = 0.01 if instrument.endswith("_JPY") else 0.0001
            spread_pips = (ask - bid) / pip_size
            if not math.isfinite(spread_pips) or spread_pips < 0:
                continue
            result[instrument] = float(spread_pips)

        return result

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

        # Check supervisor flags via StateEngine — the canonical source of
        # truth (handles schema migration + atomic read). Reading state.json
        # directly bypassed migration and risked stale/partial reads during
        # in-flight writes. Scanner readiness stays in-process (state.json
        # can outlive the embedded scanner).
        state_path = self._project_root / ".claude" / "state.json"
        try:
            from src.scanner.automation.state_engine import StateEngine
            _se = StateEngine(state_path=state_path)
            _state = _se.load_state()
            snap.scan_cycle_count = int(
                _state.get("scan_cycle_count", _state.get("scan_cycles", 0)) or 0
            )
            snap.scanner_paused = _se.get_paused()
            snap.halted = _se.get_halted()
            snap.mode = _se.get_mode()
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
                    # Calculate basic drawdown stats from recent closed trades.
                    # Do not overload tradeable_count here; that field is a
                    # live scan result and must stay 0 until a scan completes.
                    recent = entries[-20:]

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

    # ──────────────────────────────────────────────────────────────────
    # Equity-harvester state readers (read-only, graceful degradation)
    #
    # These read the equity control-loop / ship-gate / live-gate state
    # files written by src/equity/*.py. They are NOT part of the FX
    # DashboardSnapshot — the TUI screens call them directly. Every reader
    # returns a small dict with an ``available: bool`` key so callers can
    # render a "pending / not running" state when the loop hasn't run live.
    # ──────────────────────────────────────────────────────────────────

    def get_ship_gate_status(self) -> dict[str, Any]:
        """Read ``trained_data/backtests/SHIP_GATE.json`` (US-005 contract).

        Returns a dict with ``available`` False when the file is absent or
        corrupt. When available it surfaces ``gate_pass`` (bool),
        ``net_sharpe``, ``max_dd``, ``positive_years``, the ``thresholds``
        dict, ``universe_hash``, and ``recommendation``.
        """
        path = self._project_root / _EQUITY_SHIP_GATE_PATH
        payload = _read_equity_json(path)
        if payload is None:
            return {"available": False}
        thresholds = payload.get("thresholds")
        return {
            "available": True,
            "gate_pass": bool(payload.get("gate_pass", False)),
            "net_sharpe": _safe_float(payload.get("net_sharpe")),
            "max_dd": _safe_float(payload.get("max_dd")),
            "positive_years": _safe_float(payload.get("positive_years")),
            "total_years": _safe_float(payload.get("total_years")),
            "thresholds": thresholds if isinstance(thresholds, dict) else {},
            "universe_hash": str(payload.get("universe_hash") or ""),
            "recommendation": str(payload.get("recommendation") or ""),
            "asof": str(payload.get("asof") or ""),
        }

    def get_live_gate_status(self) -> dict[str, Any]:
        """Read ``trained_data/equity/live_gate_state.json`` (US-022).

        The live gate has no explicit ``mode`` field — ``armed`` is the
        single truth source. ``armed`` True means LIVE execution is enabled;
        absent/False means SHADOW. We surface a derived ``mode`` string
        ("LIVE" / "SHADOW") plus the raw ``armed`` flag and nav fractions so
        the header can default to SHADOW when the file is absent.
        """
        path = self._project_root / _EQUITY_LIVE_GATE_STATE_PATH
        payload = _read_equity_json(path)
        if payload is None:
            # Default SHADOW when absent — the safe assumption.
            return {"available": False, "armed": False, "mode": "SHADOW"}
        armed = bool(payload.get("armed", False))
        nav_frac = payload.get("initial_nav_fraction")
        return {
            "available": True,
            "armed": armed,
            "mode": "LIVE" if armed else "SHADOW",
            "nav_fraction": (
                _safe_float(nav_frac) if nav_frac is not None else None
            ),
            "universe_hash": str(payload.get("universe_hash") or ""),
            "last_event": str(payload.get("last_event") or ""),
            "last_event_reason": str(payload.get("last_event_reason") or ""),
        }

    def get_harvester_status(self) -> dict[str, Any]:
        """Read the equity control-loop + portfolio state files.

        Combines ``loop_state.json`` (cycle_count, halted, transport, risk,
        reconcile) with ``portfolio_state.json`` (nav, peak_nav, drawdown).
        Returns ``available`` False only when BOTH files are absent/corrupt
        (the harvester has not run yet). When only one is present the other
        block's fields are omitted/None.
        """
        loop_path = self._project_root / _EQUITY_LOOP_STATE_PATH
        portfolio_path = self._project_root / _EQUITY_PORTFOLIO_STATE_PATH
        loop = _read_equity_json(loop_path)
        portfolio = _read_equity_json(portfolio_path)
        if loop is None and portfolio is None:
            return {"available": False}

        status: dict[str, Any] = {"available": True}

        if loop is not None:
            status["cycle_count"] = int(loop.get("cycle_count", 0) or 0)
            status["halted"] = bool(loop.get("halted", False))
            status["last_cycle_asof"] = str(loop.get("last_cycle_asof") or "")
            status["transport_state"] = str(
                loop.get("last_transport_state") or ""
            )
            status["consecutive_failures"] = int(
                loop.get("consecutive_failures", 0) or 0
            )
            reconcile = loop.get("last_reconcile")
            if isinstance(reconcile, dict):
                breaches = reconcile.get("drift_breaches")
                status["reconcile"] = {
                    "asof": str(reconcile.get("asof") or ""),
                    "nav": _safe_float(reconcile.get("nav")),
                    "cash_drift_frac": _safe_float(
                        reconcile.get("cash_drift_frac")
                    ),
                    "n_breaches": (
                        len(breaches) if isinstance(breaches, list) else 0
                    ),
                }
            risk = loop.get("last_risk_decision")
            if isinstance(risk, dict):
                reasons = risk.get("reasons")
                status["risk"] = {
                    "block_trade": bool(risk.get("block_trade", False)),
                    "halt": bool(risk.get("halt", False)),
                    "degross_factor": _safe_float(
                        risk.get("degross_factor"), 1.0
                    ),
                    "reasons": (
                        [str(r) for r in reasons]
                        if isinstance(reasons, list)
                        else []
                    ),
                }

        if portfolio is not None:
            nav = _safe_float(portfolio.get("nav"))
            peak = _safe_float(portfolio.get("peak_nav"))
            drawdown = (
                max(0.0, 1.0 - (nav / peak)) if peak > 0 and nav >= 0 else 0.0
            )
            status["nav"] = nav
            status["peak_nav"] = peak
            status["drawdown_pct"] = drawdown * 100.0
            # portfolio-level halt also counts toward the displayed halt flag
            if bool(portfolio.get("halted", False)):
                status["halted"] = True
            status["last_rebalance_asof"] = str(
                portfolio.get("last_rebalance_asof") or ""
            )

        return status

    def get_risk_gates_status(self) -> dict[str, Any]:
        """Read the equity risk-gate verdicts from ``loop_state.json``.

        The risk layer (:mod:`src.equity.risk_agents`) does NOT write its own
        state file — the control loop persists the full ``RiskDecision`` (with
        per-gate ``verdicts``) under ``last_risk_decision`` in the loop state.
        We surface every gate's pass/block + the composite degross factor +
        the aggregate block/halt flags.

        Returns ``available`` False when the loop state is absent/corrupt OR
        present but carries no risk decision yet (loop ran but never reached
        the risk step). Each gate dict carries ``name``, ``passed``,
        ``block_trade``, ``score``, ``weight``, ``reason`` and ``reason_code``.
        """
        loop_path = self._project_root / _EQUITY_LOOP_STATE_PATH
        loop = _read_equity_json(loop_path)
        if loop is None:
            return {"available": False}
        decision = loop.get("last_risk_decision")
        if not isinstance(decision, dict):
            return {"available": False}

        raw_verdicts = decision.get("verdicts")
        gates: list[dict[str, Any]] = []
        if isinstance(raw_verdicts, list):
            for v in raw_verdicts:
                if not isinstance(v, dict):
                    continue
                gates.append(
                    {
                        "name": str(v.get("name") or ""),
                        "passed": bool(v.get("passed", False)),
                        "block_trade": bool(v.get("block_trade", False)),
                        "score": _safe_float(v.get("score")),
                        "weight": _safe_float(v.get("weight"), 1.0),
                        "reason": str(v.get("reason") or ""),
                        "reason_code": str(v.get("reason_code") or ""),
                    }
                )

        reasons = decision.get("reasons")
        return {
            "available": True,
            "block_trade": bool(decision.get("block_trade", False)),
            "halt": bool(decision.get("halt", False)),
            "degross_factor": _safe_float(
                decision.get("degross_factor"), 1.0
            ),
            "reasons": (
                [str(r) for r in reasons]
                if isinstance(reasons, list)
                else []
            ),
            "gates": gates,
            "last_cycle_asof": str(loop.get("last_cycle_asof") or ""),
        }

    def get_rebalance_status(self) -> dict[str, Any]:
        """Read ``trained_data/equity/rebalance_state.json`` (US-007 contract).

        Surfaces ``last_rebalance_asof``, the target weights of the active
        plan, and per-order status (PENDING/SENT/FILLED/FAILED). Returns
        ``available`` False when the file is absent/corrupt (no rebalance has
        run yet). ``active_plan`` is None when the state exists but no plan is
        currently in flight (all orders filled / never scheduled).
        """
        path = self._project_root / _EQUITY_REBALANCE_STATE_PATH
        payload = _read_equity_json(path)
        if payload is None:
            return {"available": False}

        status: dict[str, Any] = {
            "available": True,
            "last_rebalance_asof": str(
                payload.get("last_rebalance_asof") or ""
            ),
        }

        actual = payload.get("current_actual_weights")
        status["current_weights"] = (
            {str(k): _safe_float(v) for k, v in actual.items()}
            if isinstance(actual, dict)
            else {}
        )

        plan = payload.get("active_plan")
        if not isinstance(plan, dict):
            status["active_plan"] = None
            return status

        raw_targets = plan.get("target_weights")
        targets = (
            {str(k): _safe_float(v) for k, v in raw_targets.items()}
            if isinstance(raw_targets, dict)
            else {}
        )
        raw_orders = plan.get("orders")
        orders: list[dict[str, Any]] = []
        if isinstance(raw_orders, list):
            for o in raw_orders:
                if not isinstance(o, dict):
                    continue
                orders.append(
                    {
                        "ticker": str(o.get("ticker") or ""),
                        "side": str(o.get("side") or ""),
                        "target_weight": _safe_float(o.get("target_weight")),
                        "status": str(o.get("status") or "PENDING"),
                    }
                )
        status["active_plan"] = {
            "rebalance_id": str(plan.get("rebalance_id") or ""),
            "asof": str(plan.get("asof") or ""),
            "target_weights": targets,
            "orders": orders,
        }
        return status

    def get_alerts_status(self) -> dict[str, Any]:
        """Read equity alert state + the most recent audit sidecars.

        Combines ``alerts_state.json`` (per-alert-type ``last_fired`` cooldown
        log) with the newest ``alerts_audit/*.json`` sidecars (one per fired
        notification: type/severity/message/timestamp). Returns ``available``
        False only when BOTH the state file and the audit dir are absent. The
        ``alerts`` list is newest-first, capped at ``limit`` entries.
        """
        state_path = self._project_root / _EQUITY_ALERTS_STATE_PATH
        audit_dir = self._project_root / _EQUITY_ALERTS_AUDIT_DIR
        state = _read_equity_json(state_path)

        last_fired: dict[str, float] = {}
        if state is not None:
            raw = state.get("last_fired")
            if isinstance(raw, dict):
                for k, v in raw.items():
                    fv = _safe_float(v, -1.0)
                    if fv >= 0.0:
                        last_fired[str(k)] = fv

        alerts = self._read_alert_audit(audit_dir)

        if state is None and not alerts:
            return {"available": False}

        return {
            "available": True,
            "last_fired": last_fired,
            "universe_hash": str((state or {}).get("universe_hash") or ""),
            "last_updated": str((state or {}).get("last_updated") or ""),
            "alerts": alerts,
        }

    @staticmethod
    def _read_alert_audit(
        audit_dir: Path, limit: int = 20
    ) -> list[dict[str, Any]]:
        """Read the newest alert-audit sidecars (newest-first, capped).

        Each sidecar wraps a ``notification`` dict (alert_type / severity /
        message / timestamp). Sort by filename — the manager prefixes every
        name with the notification timestamp, so lexical order ≈ chronological.
        Best-effort: a missing dir or a corrupt sidecar is skipped, never
        raised.
        """
        try:
            if not audit_dir.is_dir():
                return []
            files = sorted(
                audit_dir.glob("*.json"), reverse=True
            )[:limit]
        except OSError:
            return []

        out: list[dict[str, Any]] = []
        for f in files:
            payload = _read_equity_json(f)
            if payload is None:
                continue
            note = payload.get("notification")
            if not isinstance(note, dict):
                continue
            out.append(
                {
                    "alert_type": str(note.get("alert_type") or ""),
                    "severity": str(note.get("severity") or ""),
                    "message": str(note.get("message") or ""),
                    "timestamp": str(note.get("timestamp") or ""),
                }
            )
        return out

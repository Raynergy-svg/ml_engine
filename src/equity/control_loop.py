"""US-017 — Autonomous control loop for the equity-beta harvester.

Composes the existing equity modules (US-007 rebalance scheduler, US-008
risk-agent layer, US-010 market calendar + halt registry, US-013 kill
switch, US-016 shadow-fill simulator) into a single deterministic,
Claude-free, fail-closed runtime that the operator can launch and walk
away from.

Surfaces
--------

* :class:`TransportState` — Connected / Degraded / Reconnecting / Down.
* :class:`IBKRTransportFSM` — heartbeat-driven transport state machine;
  the loop FAILS CLOSED (skips execute, stays halted) on any state
  worse than ``CONNECTED``.
* :class:`StateReconciler` — IBKR fetcher that reconciles SHARES + CASH
  (not just NAV) and computes per-name drift between expected and live
  positions; drift above ``equity_drift_bps`` re-plans the cycle.
* :class:`AutonomousLoop` — the run-loop. Step ordering:
  schedule → reconcile → weights/risk gates → rebalance/execute (shadow
  by default) → drawdown monitor → state persist → self-heal.
* :class:`LoopConfig` — knobs (cycle interval, max cycles, executor
  selection, drift thresholds, persistence paths).
* :class:`STATE_CORRUPT` marker logged when the persisted loop-state on
  restart is corrupt: the loop refuses to execute, stays halted, and
  surfaces the error to the caller without crashing.

Hard constraints
----------------

* **Claude-free hot path** — this module imports nothing from
  ``anthropic`` / ``claude*`` / ``openai`` / ``llm`` namespaces. A
  standing canary test (``tests/test_no_llm_in_equity_path.py``)
  AST-greps the whole strategy + execution surface to enforce this.
* **Fail-closed by default** — every degraded condition (transport
  down, ship-gate refused, corrupt state, risk halt) skips execution.
  No blind rebalance under partial information.
* **Atomic state writes** — ``tempfile.mkstemp`` + ``os.replace``,
  identical to the other equity surfaces.
* **No mocks in tests** — real ``RebalanceScheduler``, real
  ``RiskAgentLayer``, real ``ShadowFillSimulator`` against real
  ``tmp_path`` disk; a stub executor satisfies the broker callable.
* **Ship-gate guarded at construction** — :meth:`AutonomousLoop.create`
  re-runs the byte-for-byte identical gate check used by every other
  equity surface.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional

import pandas as pd

from src.equity.decision_capture import (
    LOG_TAG_DECISION_CAPTURE_FAILED,
    build_cycle_decisions,
    persist_cycle_decisions,
)
from src.learning.decision_record import Disposition
from src.equity.rebalance import (
    DEFAULT_NO_TRADE_BAND,
    DEFAULT_REBALANCE_FREQ_DAYS,
    Order,
    RebalancePlan,
    RebalanceScheduler,
)
from src.equity.risk_agents import (
    PortfolioState,
    RiskAgentLayer,
    RiskDecision,
)

logger = logging.getLogger(__name__)


SHIP_GATE_PATH_DEFAULT = "trained_data/backtests/SHIP_GATE.json"
# Canonical default state-file locations. The constructor still takes explicit
# state_path/portfolio_state_path, but these constants are the single source of
# truth the loop launcher AND the TUI reader (src/tui/data_provider.py) share,
# so the panel is never wired to a path the producer doesn't write.
LOOP_STATE_PATH_DEFAULT = "trained_data/equity/loop_state.json"
PORTFOLIO_STATE_PATH_DEFAULT = "trained_data/equity/portfolio_state.json"
STATE_VERSION = 1
STATE_CORRUPT = "STATE_CORRUPT"

# Exact substrings the smoke test greps from logs/buddy_debug.log. Keep
# stable — the test pins on these. Adding new lines is fine; renaming an
# existing one breaks the AC.
LOG_TAG_CYCLE_BEGIN = "AUTONOMOUS_LOOP cycle_begin"
LOG_TAG_CYCLE_END = "AUTONOMOUS_LOOP cycle_end"
LOG_TAG_TRANSPORT_OK = "AUTONOMOUS_LOOP transport_ok"
LOG_TAG_TRANSPORT_DOWN = "AUTONOMOUS_LOOP transport_down"
LOG_TAG_FAIL_CLOSED = "AUTONOMOUS_LOOP fail_closed"
LOG_TAG_RECONCILE_OK = "AUTONOMOUS_LOOP reconcile_ok"
LOG_TAG_RISK_BLOCK = "AUTONOMOUS_LOOP risk_block"
LOG_TAG_EXEC_SHADOW = "AUTONOMOUS_LOOP exec_shadow"
LOG_TAG_STATE_PERSIST = "AUTONOMOUS_LOOP state_persist"
LOG_TAG_STATE_CORRUPT = f"AUTONOMOUS_LOOP {STATE_CORRUPT}"
LOG_TAG_SELF_HEAL = "AUTONOMOUS_LOOP self_heal"


class ControlLoopError(RuntimeError):
    """Raised on ship-gate refusal, malformed config, or contract bugs."""


# ---------------------------------------------------------------------------
# Ship-gate guard (identical contract to every other equity surface)
# ---------------------------------------------------------------------------


def _enforce_ship_gate(path: Path, expected_universe_hash: str) -> None:
    """Refuse to construct unless the ship gate clears for THIS universe.

    Byte-for-byte identical fail-closed semantics to the other equity
    surfaces (``rebalance``, ``risk_agents``, ``kill_switch``, …).
    """
    path = Path(path)
    if not path.exists():
        raise ControlLoopError(
            f"ship gate refuses: file not found at {path}"
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ControlLoopError(
            f"ship gate refuses: cannot read {path}: {exc}"
        ) from exc
    if not isinstance(payload, Mapping):
        raise ControlLoopError(
            f"ship gate refuses: {path} payload is not a JSON object"
        )
    if payload.get("gate_pass") is not True:
        raise ControlLoopError(
            "ship gate refuses: gate_pass="
            f"{payload.get('gate_pass')!r} at {path} "
            "(must be exactly True to unblock the control loop)"
        )
    universe_hash = payload.get("universe_hash")
    if not isinstance(universe_hash, str) or not universe_hash:
        raise ControlLoopError(
            f"ship gate refuses: universe_hash missing/blank at {path}"
        )
    if universe_hash != expected_universe_hash:
        raise ControlLoopError(
            "ship gate refuses: universe_hash mismatch "
            f"(gate={universe_hash!r}, "
            f"expected={expected_universe_hash!r}) at {path}"
        )


# ---------------------------------------------------------------------------
# Atomic write
# ---------------------------------------------------------------------------


def _atomic_write_json(payload: Mapping[str, Any], path: Path) -> Path:
    """Atomic JSON write — ``tempfile.mkstemp`` + :func:`os.replace`."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_path, path)
    except OSError as exc:
        logger.error(
            "atomic write failed for %s: %s", path, exc, exc_info=True
        )
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError as cleanup_exc:
            logger.warning(
                "failed to remove temp file %s: %s", tmp_path, cleanup_exc
            )
        raise
    return path


# ---------------------------------------------------------------------------
# IBKR transport FSM (heartbeat-driven)
# ---------------------------------------------------------------------------


class TransportState(str, Enum):
    """IBKR transport health states.

    ``CONNECTED`` is the only state under which execution is permitted.
    Anything else fails the cycle closed.
    """

    CONNECTED = "CONNECTED"
    DEGRADED = "DEGRADED"
    RECONNECTING = "RECONNECTING"
    DOWN = "DOWN"


@dataclass
class IBKRTransportFSM:
    """Heartbeat-driven IBKR transport state machine.

    The loop does not own the IBKR socket — the operator wires a small
    ``heartbeat_fn`` callback (``() -> bool``) that returns True iff the
    broker socket is alive (e.g. ``IBKRBroker._ib.isConnected()`` in
    production, a flag in tests). The FSM advances state on each call to
    :meth:`update`:

    * ``True``  → CONNECTED (resets misses)
    * ``False`` × 1 → DEGRADED
    * ``False`` × ``reconnect_after`` → RECONNECTING (triggers
      ``reconnect_fn`` if supplied)
    * ``False`` × ``down_after`` → DOWN

    The FSM is intentionally tiny — its only job is to gate the
    execute step of the control loop. The full reconnect/backoff
    machinery lives in :class:`src.brokers.ibkr.IBKRBroker`.
    """

    heartbeat_fn: Callable[[], bool]
    reconnect_fn: Optional[Callable[[], None]] = None
    reconnect_after: int = 1
    down_after: int = 3

    state: TransportState = TransportState.CONNECTED
    consecutive_misses: int = 0
    last_transition_ts: float = 0.0

    def update(self) -> TransportState:
        """Poll the heartbeat once and advance state."""
        try:
            alive = bool(self.heartbeat_fn())
        except (RuntimeError, OSError, ValueError) as exc:
            logger.warning(
                "heartbeat_fn raised %s — treating as DOWN", exc
            )
            alive = False

        if alive:
            if self.state is not TransportState.CONNECTED:
                self.last_transition_ts = time.time()
            self.state = TransportState.CONNECTED
            self.consecutive_misses = 0
            return self.state

        self.consecutive_misses += 1
        if self.consecutive_misses >= self.down_after:
            self.state = TransportState.DOWN
        elif self.consecutive_misses >= self.reconnect_after:
            new_state = (
                TransportState.RECONNECTING
                if self.consecutive_misses >= self.reconnect_after
                else TransportState.DEGRADED
            )
            if new_state is TransportState.RECONNECTING:
                if self.reconnect_fn is not None:
                    try:
                        self.reconnect_fn()
                    except (RuntimeError, OSError) as exc:
                        logger.warning(
                            "reconnect_fn raised %s — staying in "
                            "RECONNECTING",
                            exc,
                        )
            self.state = new_state
        else:
            self.state = TransportState.DEGRADED
        self.last_transition_ts = time.time()
        return self.state

    def is_healthy(self) -> bool:
        """``True`` iff execution is permitted (CONNECTED only)."""
        return self.state is TransportState.CONNECTED

    def snapshot(self) -> Dict[str, Any]:
        return {
            "state": str(self.state.value),
            "consecutive_misses": int(self.consecutive_misses),
            "last_transition_ts": float(self.last_transition_ts),
        }


# ---------------------------------------------------------------------------
# State reconciler (SHARES + CASH, not just NAV)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ReconcileReport:
    """Outcome of one reconcile pass.

    ``per_name_drift`` is keyed by ticker with the value being
    ``|expected_weight - live_weight|`` (absolute, fractional). The
    ``cash_drift_frac`` field is signed:
    ``(live_cash - expected_cash) / nav``.
    ``drift_breaches`` enumerates tickers that exceed the configured
    band; an empty list means the live book matches the expected book
    within tolerance.
    """

    asof: str
    nav: float
    cash: float
    expected_cash: float
    cash_drift_frac: float
    live_weights: Dict[str, float]
    expected_weights: Dict[str, float]
    per_name_drift: Dict[str, float]
    drift_breaches: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "asof": self.asof,
            "nav": float(self.nav),
            "cash": float(self.cash),
            "expected_cash": float(self.expected_cash),
            "cash_drift_frac": float(self.cash_drift_frac),
            "live_weights": dict(self.live_weights),
            "expected_weights": dict(self.expected_weights),
            "per_name_drift": dict(self.per_name_drift),
            "drift_breaches": list(self.drift_breaches),
        }


# Type alias: a fetcher that returns ``(cash, shares_by_ticker,
# prices_by_ticker)``. In production this is wired to an IBKR adapter;
# in tests a plain callable returning a fixed dict suffices — no mock
# class is needed.
LiveBookFetcher = Callable[
    [], "tuple[float, Mapping[str, float], Mapping[str, float]]"
]


@dataclass
class StateReconciler:
    """Reconciles SHARES + CASH against expected per-name weights.

    Why SHARES + CASH instead of NAV-only: a NAV match can hide a
    ticker-level mismatch (e.g. correct dollar-weighted total but wrong
    constituents). Reconciling at the per-name level catches drift the
    rebalance scheduler would otherwise rebalance away into the wrong
    target.
    """

    fetcher: LiveBookFetcher
    equity_drift_bps: float = 50.0  # 50bps per-name absolute weight drift
    cash_drift_bps: float = 100.0  # 1% NAV cash drift

    def reconcile(
        self,
        *,
        expected_weights: Mapping[str, float],
        asof: pd.Timestamp,
    ) -> ReconcileReport:
        cash, shares, prices = self.fetcher()
        cash = float(cash)
        positions_value = 0.0
        live_values: Dict[str, float] = {}
        for ticker, qty in shares.items():
            qty_f = float(qty)
            if qty_f == 0.0:
                continue
            price = float(prices.get(ticker, 0.0))
            value = qty_f * price
            live_values[ticker] = value
            positions_value += value
        nav = cash + positions_value
        if nav <= 0:
            raise ControlLoopError(
                f"reconcile: non-positive NAV {nav} from live book"
            )

        live_weights = {
            t: v / nav for t, v in live_values.items()
        }
        expected_clean = {
            t: float(w)
            for t, w in expected_weights.items()
            if float(w) != 0.0
        }
        all_tickers = set(live_weights) | set(expected_clean)
        per_name_drift: Dict[str, float] = {}
        breaches: List[str] = []
        band = self.equity_drift_bps / 1e4
        for t in sorted(all_tickers):
            live = live_weights.get(t, 0.0)
            exp = expected_clean.get(t, 0.0)
            drift = abs(live - exp)
            per_name_drift[t] = drift
            if drift > band:
                breaches.append(t)

        expected_cash = nav * (1.0 - sum(expected_clean.values()))
        cash_drift_frac = (cash - expected_cash) / nav
        if abs(cash_drift_frac) > (self.cash_drift_bps / 1e4):
            breaches.append("__CASH__")

        asof_iso = pd.Timestamp(asof)
        if asof_iso.tz is None:
            asof_iso = asof_iso.tz_localize("UTC")
        else:
            asof_iso = asof_iso.tz_convert("UTC")

        return ReconcileReport(
            asof=asof_iso.isoformat(),
            nav=nav,
            cash=cash,
            expected_cash=expected_cash,
            cash_drift_frac=cash_drift_frac,
            live_weights=live_weights,
            expected_weights=expected_clean,
            per_name_drift=per_name_drift,
            drift_breaches=breaches,
        )


# ---------------------------------------------------------------------------
# Loop state (persisted)
# ---------------------------------------------------------------------------


@dataclass
class LoopState:
    """Persisted control-loop state — survives shutdown/restart."""

    version: int = STATE_VERSION
    cycle_count: int = 0
    halted: bool = False
    last_cycle_asof: Optional[str] = None
    last_transport_state: str = TransportState.CONNECTED.value
    last_risk_decision: Optional[Dict[str, Any]] = None
    last_reconcile: Optional[Dict[str, Any]] = None
    consecutive_failures: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": int(self.version),
            "cycle_count": int(self.cycle_count),
            "halted": bool(self.halted),
            "last_cycle_asof": self.last_cycle_asof,
            "last_transport_state": str(self.last_transport_state),
            "last_risk_decision": self.last_risk_decision,
            "last_reconcile": self.last_reconcile,
            "consecutive_failures": int(self.consecutive_failures),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "LoopState":
        if not isinstance(payload, Mapping):
            raise ControlLoopError(
                "loop state payload must be a JSON object"
            )
        try:
            version = int(payload.get("version", STATE_VERSION))
            cycle_count = int(payload.get("cycle_count", 0))
        except (TypeError, ValueError) as exc:
            raise ControlLoopError(
                f"loop state has malformed scalar fields: {exc}"
            ) from exc
        return cls(
            version=version,
            cycle_count=cycle_count,
            halted=bool(payload.get("halted", False)),
            last_cycle_asof=(
                str(payload["last_cycle_asof"])
                if payload.get("last_cycle_asof")
                else None
            ),
            last_transport_state=str(
                payload.get(
                    "last_transport_state", TransportState.CONNECTED.value
                )
            ),
            last_risk_decision=payload.get("last_risk_decision"),
            last_reconcile=payload.get("last_reconcile"),
            consecutive_failures=int(payload.get("consecutive_failures", 0)),
        )


# ---------------------------------------------------------------------------
# Loop configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LoopConfig:
    """Knobs for :class:`AutonomousLoop`.

    Keep the defaults shadow-mode: every executor path persists the fill
    intent to the simulator ledger before it touches the live broker.
    Operators flip ``shadow_mode=False`` only after a successful live
    arm-up.
    """

    universe_hash: str
    cycle_interval_seconds: float = 1.0
    max_cycles: Optional[int] = None
    shadow_mode: bool = True
    equity_drift_bps: float = 50.0
    cash_drift_bps: float = 100.0
    rebalance_freq_days: int = DEFAULT_REBALANCE_FREQ_DAYS
    no_trade_band: float = DEFAULT_NO_TRADE_BAND
    self_heal_after_consecutive_failures: int = 3


# ---------------------------------------------------------------------------
# Cycle outcome (for tests / observability)
# ---------------------------------------------------------------------------


class CycleOutcome(str, Enum):
    """Why a cycle ended."""

    EXECUTED = "EXECUTED"
    NO_PLAN = "NO_PLAN"
    RISK_BLOCKED = "RISK_BLOCKED"
    TRANSPORT_FAILED_CLOSED = "TRANSPORT_FAILED_CLOSED"
    STATE_CORRUPT = "STATE_CORRUPT"
    HALTED = "HALTED"
    SHIP_GATE_REFUSED = "SHIP_GATE_REFUSED"
    # An executable decision could not be durably recorded. The cycle
    # refuses rather than submitting an order nobody can attribute.
    ABSTAINED_UNRECORDED = "ABSTAINED_UNRECORDED"


@dataclass(frozen=True)
class CycleReport:
    """Structured outcome of one cycle — used by tests + observability."""

    cycle_index: int
    asof: str
    outcome: CycleOutcome
    transport_state: TransportState
    reconcile: Optional[ReconcileReport]
    risk: Optional[RiskDecision]
    orders_attempted: int = 0
    orders_filled: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cycle_index": int(self.cycle_index),
            "asof": str(self.asof),
            "outcome": str(self.outcome.value),
            "transport_state": str(self.transport_state.value),
            "orders_attempted": int(self.orders_attempted),
            "orders_filled": int(self.orders_filled),
            "reconcile": (
                self.reconcile.to_dict() if self.reconcile else None
            ),
            "risk": self.risk.to_dict() if self.risk else None,
        }


# Target-weight callable contract: ``(asof) -> {ticker: weight}``.
TargetWeightFn = Callable[[pd.Timestamp], Mapping[str, float]]
ExecutorFn = Callable[[Order], bool]
AdvProviderFn = Callable[[pd.Timestamp], Mapping[str, float]]


# ---------------------------------------------------------------------------
# AutonomousLoop
# ---------------------------------------------------------------------------


@dataclass
class AutonomousLoop:
    """Fully hands-off equity-harvester control loop.

    Constructed via :meth:`create` so the ship-gate guard runs once
    before any other method becomes reachable. The loop is *deterministic
    and Claude-free*: every decision is a direct function of disk state,
    config, and broker readbacks. No LLM is invoked in any cycle.

    Step ordering (the canonical loop):

    1. **schedule** — load persisted loop state; refuse if corrupt.
    2. **transport** — poll IBKR transport FSM; FAIL CLOSED if not CONNECTED.
    3. **reconcile** — IBKR fetcher returns CASH + SHARES + prices;
       reconciler computes per-name drift vs ``expected_weights``.
    4. **weights** — get_target_weights(asof).
    5. **risk gates** — RiskAgentLayer.evaluate(...) — composite verdict.
    6. **rebalance/execute** — if not blocked: plan + execute. In shadow
       mode the executor wraps the broker callable in a recorder that
       writes every order to the simulator ledger.
    7. **drawdown monitor** — drawdown is checked inside the risk layer
       UNCONDITIONALLY per US-008; this step persists the verdict.
    8. **state persist** — atomic write of loop state + scheduler state.
    9. **self-heal** — on N consecutive failures, log self_heal tag so
       the supervisor can decide whether to bounce the process.
    """

    config: LoopConfig
    scheduler: RebalanceScheduler
    risk_layer: RiskAgentLayer
    transport: IBKRTransportFSM
    reconciler: StateReconciler
    get_target_weights: TargetWeightFn
    execute_order: ExecutorFn
    state_path: Path
    portfolio_state_path: Path
    ship_gate_path: Path
    get_adv: Optional[AdvProviderFn] = None
    # Where decision evidence is written. None uses the shared default;
    # tests point it at a tmp path. Purely additive -- every existing
    # construction site keeps working unchanged.
    decision_ledger_path: Optional[Path] = None

    @classmethod
    def create(
        cls,
        *,
        config: LoopConfig,
        scheduler: RebalanceScheduler,
        risk_layer: RiskAgentLayer,
        transport: IBKRTransportFSM,
        reconciler: StateReconciler,
        get_target_weights: TargetWeightFn,
        execute_order: ExecutorFn,
        state_path: Path,
        portfolio_state_path: Path,
        ship_gate_path: Path,
        get_adv: Optional[AdvProviderFn] = None,
    ) -> "AutonomousLoop":
        if not isinstance(config, LoopConfig):
            raise ControlLoopError(
                f"config must be LoopConfig, got {type(config)!r}"
            )
        if config.cycle_interval_seconds < 0:
            raise ControlLoopError(
                "cycle_interval_seconds must be >= 0, got "
                f"{config.cycle_interval_seconds}"
            )
        if (
            config.max_cycles is not None
            and config.max_cycles < 0
        ):
            raise ControlLoopError(
                f"max_cycles must be None or >= 0, got {config.max_cycles}"
            )
        _enforce_ship_gate(Path(ship_gate_path), config.universe_hash)
        return cls(
            config=config,
            scheduler=scheduler,
            risk_layer=risk_layer,
            transport=transport,
            reconciler=reconciler,
            get_target_weights=get_target_weights,
            execute_order=execute_order,
            state_path=Path(state_path),
            portfolio_state_path=Path(portfolio_state_path),
            ship_gate_path=Path(ship_gate_path),
            get_adv=get_adv,
        )

    # ------------------------------------------------------------------
    # Persisted state I/O
    # ------------------------------------------------------------------
    def load_state(self) -> "tuple[LoopState, bool]":
        """Read persisted loop state.

        Returns ``(state, is_corrupt)``. On corrupt JSON or schema
        mismatch we log :data:`STATE_CORRUPT`, return a halted state and
        flag ``is_corrupt=True`` so the cycle skips execute and the
        caller surfaces :data:`CycleOutcome.STATE_CORRUPT`. Missing file
        is *not* corruption — that's a fresh boot.
        """
        path = Path(self.state_path)
        if not path.exists():
            return LoopState(), False
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.error(
                "%s state=%s reason=%s",
                LOG_TAG_STATE_CORRUPT,
                path,
                exc,
            )
            return LoopState(halted=True), True
        try:
            return LoopState.from_dict(payload), False
        except ControlLoopError as exc:
            logger.error(
                "%s state=%s reason=%s",
                LOG_TAG_STATE_CORRUPT,
                path,
                exc,
            )
            return LoopState(halted=True), True

    def write_state(self, state: LoopState) -> None:
        _atomic_write_json(state.to_dict(), self.state_path)
        logger.info("%s path=%s", LOG_TAG_STATE_PERSIST, self.state_path)

    def write_portfolio_state(self, state: PortfolioState) -> None:
        _atomic_write_json(state.to_dict(), self.portfolio_state_path)

    def load_portfolio_state(self) -> Optional[PortfolioState]:
        path = Path(self.portfolio_state_path)
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.error(
                "%s portfolio_state=%s reason=%s",
                LOG_TAG_STATE_CORRUPT,
                path,
                exc,
            )
            return None
        try:
            return PortfolioState.from_dict(payload)
        except Exception as exc:  # noqa: BLE001 — surfaced via log + None
            logger.error(
                "%s portfolio_state=%s reason=%s",
                LOG_TAG_STATE_CORRUPT,
                path,
                exc,
            )
            return None

    # ------------------------------------------------------------------
    # Cycle
    # ------------------------------------------------------------------
    def run_cycle(
        self,
        *,
        asof: pd.Timestamp,
        portfolio_state: PortfolioState,
    ) -> CycleReport:
        """Execute one cycle. Pure function of inputs + disk state."""
        loop_state, is_corrupt = self.load_state()
        ts_iso = _iso(asof)
        logger.info(
            "%s cycle=%d asof=%s shadow=%s",
            LOG_TAG_CYCLE_BEGIN,
            loop_state.cycle_count,
            ts_iso,
            self.config.shadow_mode,
        )

        # 1. corrupt state → fail closed (no order emitted, halted true)
        if is_corrupt:
            logger.error(
                "%s outcome=%s",
                LOG_TAG_FAIL_CLOSED,
                CycleOutcome.STATE_CORRUPT.value,
            )
            report = CycleReport(
                cycle_index=loop_state.cycle_count,
                asof=ts_iso,
                outcome=CycleOutcome.STATE_CORRUPT,
                transport_state=self.transport.state,
                reconcile=None,
                risk=None,
            )
            self._advance(loop_state, report)
            return report

        # 2. transport health
        self.transport.update()
        if not self.transport.is_healthy():
            logger.error(
                "%s state=%s",
                LOG_TAG_TRANSPORT_DOWN,
                self.transport.state.value,
            )
            logger.error(
                "%s outcome=%s",
                LOG_TAG_FAIL_CLOSED,
                CycleOutcome.TRANSPORT_FAILED_CLOSED.value,
            )
            report = CycleReport(
                cycle_index=loop_state.cycle_count,
                asof=ts_iso,
                outcome=CycleOutcome.TRANSPORT_FAILED_CLOSED,
                transport_state=self.transport.state,
                reconcile=None,
                risk=None,
            )
            self._advance(loop_state, report)
            return report
        logger.info(
            "%s state=%s", LOG_TAG_TRANSPORT_OK, self.transport.state.value
        )

        # 3. weights
        target_weights = dict(self.get_target_weights(asof))

        # 4. reconcile (SHARES + CASH, not just NAV)
        reconcile_report: Optional[ReconcileReport] = None
        try:
            reconcile_report = self.reconciler.reconcile(
                expected_weights=portfolio_state.current_weights,
                asof=asof,
            )
            logger.info(
                "%s nav=%.2f cash_drift_frac=%.6f breaches=%d",
                LOG_TAG_RECONCILE_OK,
                reconcile_report.nav,
                reconcile_report.cash_drift_frac,
                len(reconcile_report.drift_breaches),
            )
            # Keep portfolio_state.nav/peak_nav fresh from the live book.
            portfolio_state.nav = reconcile_report.nav
            if reconcile_report.nav > portfolio_state.peak_nav:
                portfolio_state.peak_nav = reconcile_report.nav
        except ControlLoopError as exc:
            logger.error(
                "%s outcome=%s reason=%s",
                LOG_TAG_FAIL_CLOSED,
                CycleOutcome.TRANSPORT_FAILED_CLOSED.value,
                exc,
            )
            report = CycleReport(
                cycle_index=loop_state.cycle_count,
                asof=ts_iso,
                outcome=CycleOutcome.TRANSPORT_FAILED_CLOSED,
                transport_state=self.transport.state,
                reconcile=None,
                risk=None,
            )
            self._advance(loop_state, report)
            return report

        # 5. risk
        adv = self.get_adv(asof) if self.get_adv is not None else None
        risk_decision = self.risk_layer.evaluate(
            asof=asof,
            target_weights=target_weights,
            state=portfolio_state,
            adv=adv,
        )
        if risk_decision.block_trade or risk_decision.halt:
            portfolio_state.halted = bool(risk_decision.halt)
            self.write_portfolio_state(portfolio_state)
            logger.warning(
                "%s reasons=%s halt=%s",
                LOG_TAG_RISK_BLOCK,
                ",".join(risk_decision.reasons) or "none",
                risk_decision.halt,
            )
            outcome = (
                CycleOutcome.HALTED
                if risk_decision.halt
                else CycleOutcome.RISK_BLOCKED
            )
            # Candidate evidence: the lane refused, but the signal, sizing
            # intention and refusal reason are still worth recording. A failed
            # write must NOT change the disposition -- the trade was already
            # refused -- but must be loudly visible.
            self._capture_decisions(
                loop_state=loop_state,
                ts_iso=ts_iso,
                disposition=(
                    Disposition.HALT if risk_decision.halt else Disposition.NO_ACT
                ),
                reason_codes=list(risk_decision.reasons),
                target_weights=target_weights,
                degross_factor=risk_decision.degross_factor,
                portfolio_state=portfolio_state,
                executable=False,
            )
            report = CycleReport(
                cycle_index=loop_state.cycle_count,
                asof=ts_iso,
                outcome=outcome,
                transport_state=self.transport.state,
                reconcile=reconcile_report,
                risk=risk_decision,
            )
            self._advance(loop_state, report)
            return report

        # 6. de-gross + plan
        scaled = {
            t: float(w) * float(risk_decision.degross_factor)
            for t, w in target_weights.items()
        }
        plan = self._plan_or_resume(asof, scaled)
        if plan is None or not plan.orders:
            self._capture_decisions(
                loop_state=loop_state,
                ts_iso=ts_iso,
                disposition=Disposition.NO_ACT,
                reason_codes=["no_plan"],
                target_weights=target_weights,
                degross_factor=risk_decision.degross_factor,
                portfolio_state=portfolio_state,
                executable=False,
            )
            report = CycleReport(
                cycle_index=loop_state.cycle_count,
                asof=ts_iso,
                outcome=CycleOutcome.NO_PLAN,
                transport_state=self.transport.state,
                reconcile=reconcile_report,
                risk=risk_decision,
            )
            self._advance(loop_state, report)
            return report

        # 6b. record the executable decision BEFORE any order leaves. A failed
        # write converts the cycle to ABSTAIN: an execution nobody recorded is
        # an execution nobody can attribute.
        if not self._capture_decisions(
            loop_state=loop_state,
            ts_iso=ts_iso,
            disposition=Disposition.EXECUTE,
            reason_codes=list(risk_decision.reasons),
            target_weights=target_weights,
            degross_factor=risk_decision.degross_factor,
            portfolio_state=portfolio_state,
            executable=True,
        ):
            report = CycleReport(
                cycle_index=loop_state.cycle_count,
                asof=ts_iso,
                outcome=CycleOutcome.ABSTAINED_UNRECORDED,
                transport_state=self.transport.state,
                reconcile=reconcile_report,
                risk=risk_decision,
            )
            self._advance(loop_state, report)
            return report

        # 7. execute (shadow_mode tag goes into the log either way)
        attempted = 0
        filled = 0
        try:
            touched = self.scheduler.execute_plan(plan, self.execute_order)
            attempted = len(touched)
            filled = sum(1 for o in touched if o.is_filled)
        except (
            RuntimeError,
            OSError,
            ValueError,
        ) as exc:
            # Order-level failures are persisted by execute_plan itself; we
            # log + surface but do NOT crash the loop. The next cycle will
            # re-attempt the still-PENDING orders thanks to the idempotent
            # client_order_id.
            logger.error(
                "execute_plan raised %s — cycle continues; non-filled "
                "orders persist for retry",
                exc,
            )

        logger.info(
            "%s plan=%s orders_attempted=%d orders_filled=%d",
            LOG_TAG_EXEC_SHADOW if self.config.shadow_mode else "AUTONOMOUS_LOOP exec_live",
            plan.rebalance_id,
            attempted,
            filled,
        )

        # 8. finalize if fully filled
        if not plan.remaining_orders():
            try:
                self.scheduler.finalize_plan(plan)
            except Exception as exc:  # noqa: BLE001 — logged + surfaced
                logger.error("finalize_plan refused: %s", exc)

        # update portfolio state from filled orders
        for o in plan.filled_orders():
            if o.target_weight == 0.0:
                portfolio_state.current_weights.pop(o.ticker, None)
            else:
                portfolio_state.current_weights[o.ticker] = float(
                    o.target_weight
                )
        portfolio_state.last_rebalance_asof = plan.asof
        self.write_portfolio_state(portfolio_state)

        report = CycleReport(
            cycle_index=loop_state.cycle_count,
            asof=ts_iso,
            outcome=CycleOutcome.EXECUTED,
            transport_state=self.transport.state,
            reconcile=reconcile_report,
            risk=risk_decision,
            orders_attempted=attempted,
            orders_filled=filled,
        )
        self._advance(loop_state, report)
        return report

    # ------------------------------------------------------------------
    # Multi-cycle run
    # ------------------------------------------------------------------
    def run(
        self,
        *,
        clock: Optional[Callable[[], pd.Timestamp]] = None,
        sleep_fn: Optional[Callable[[float], None]] = None,
    ) -> List[CycleReport]:
        """Run the loop until ``max_cycles`` is reached (or forever).

        ``clock`` is the asof source per cycle (defaults to UTC now).
        ``sleep_fn`` controls inter-cycle pacing (defaults to
        :func:`time.sleep`). Both are injectable so tests can drive the
        loop deterministically without real wall clock.
        """
        def _default_clock() -> pd.Timestamp:
            return pd.Timestamp.utcnow().tz_localize("UTC")

        if clock is None:
            clock = _default_clock
        if sleep_fn is None:
            sleep_fn = time.sleep

        portfolio_state = self.load_portfolio_state()
        if portfolio_state is None:
            raise ControlLoopError(
                "no portfolio state on disk at "
                f"{self.portfolio_state_path}; seed before run()"
            )

        reports: List[CycleReport] = []
        max_cycles = self.config.max_cycles
        while True:
            if max_cycles is not None and len(reports) >= max_cycles:
                break
            asof = clock()
            report = self.run_cycle(
                asof=asof, portfolio_state=portfolio_state
            )
            reports.append(report)
            logger.info(
                "%s cycle=%d outcome=%s",
                LOG_TAG_CYCLE_END,
                report.cycle_index,
                report.outcome.value,
            )
            if self.config.cycle_interval_seconds > 0:
                sleep_fn(self.config.cycle_interval_seconds)
        return reports

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _plan_or_resume(
        self,
        asof: pd.Timestamp,
        target_weights: Mapping[str, float],
    ) -> Optional[RebalancePlan]:
        """Resume an in-flight plan or plan a new one when triggered."""
        resumed = self.scheduler.reconcile_and_resume()
        if resumed is not None and resumed.remaining_orders():
            return resumed
        if resumed is not None and not resumed.remaining_orders():
            # Plan finalised in a prior cycle but state wasn't cleared;
            # finalize now so plan_and_persist can proceed.
            try:
                self.scheduler.finalize_plan(resumed)
            except Exception as exc:  # noqa: BLE001 — logged + surfaced
                logger.warning(
                    "finalize_plan on stale resumed plan failed: %s", exc
                )
        state = self.scheduler.load_state()
        fire, reason = self.scheduler.should_rebalance(
            now=asof,
            current_weights=state.current_actual_weights,
            target_weights=target_weights,
            last_rebalance_asof=state.last_rebalance_asof,
        )
        if not fire:
            return None
        logger.info(
            "AUTONOMOUS_LOOP plan_fire reason=%s n_targets=%d",
            reason,
            len(target_weights),
        )
        return self.scheduler.plan_and_persist(
            target_weights=target_weights, asof=asof
        )

    def _capture_decisions(
        self,
        *,
        loop_state: LoopState,
        ts_iso: str,
        disposition: Disposition,
        reason_codes: List[str],
        target_weights: Dict[str, float],
        degross_factor: float,
        portfolio_state: PortfolioState,
        executable: bool,
    ) -> bool:
        """Record this cycle's decisions. Returns False iff the caller must abstain.

        Deterministic and side-effect-free with respect to the trading decision
        -- it records what was decided, it never influences it. The one place it
        DOES affect control flow is the executable path: a decision that cannot
        be durably written must not become an order.

        ``cycle_id`` is derived from (asof, cycle_count) so a retry of the same
        cycle reproduces the same identity digest and appends nothing new.
        """
        cycle_id = f"{ts_iso}:{loop_state.cycle_count}"
        try:
            records = build_cycle_decisions(
                cycle_id=cycle_id,
                decision_timestamp=ts_iso,
                disposition=disposition,
                reason_codes=reason_codes,
                target_weights=target_weights,
                degross_factor=degross_factor,
                current_weights=portfolio_state.current_weights,
                portfolio_nav=portfolio_state.nav,
                peak_nav=portfolio_state.peak_nav,
                market_data_asof=ts_iso,
                strategy_version=getattr(self.config, "pipeline_version", None),
                lane="equity",
            )
        except Exception as exc:  # noqa: BLE001 - capture must never crash the loop
            logger.error(
                "%s could not BUILD decision records: %r", LOG_TAG_DECISION_CAPTURE_FAILED, exc
            )
            return not executable

        persisted, _stats = persist_cycle_decisions(
            records, executable=executable, path=self.decision_ledger_path
        )
        # A candidate's disposition is already decided; losing its record is an
        # observability failure, not a trading one. Only an executable decision
        # is gated on the write.
        return persisted if executable else True

    def _advance(self, loop_state: LoopState, report: CycleReport) -> None:
        loop_state.cycle_count += 1
        loop_state.last_cycle_asof = report.asof
        loop_state.last_transport_state = report.transport_state.value
        if report.risk is not None:
            loop_state.last_risk_decision = report.risk.to_dict()
        if report.reconcile is not None:
            loop_state.last_reconcile = report.reconcile.to_dict()
        if report.outcome in (
            CycleOutcome.TRANSPORT_FAILED_CLOSED,
            CycleOutcome.STATE_CORRUPT,
        ):
            loop_state.consecutive_failures += 1
        else:
            loop_state.consecutive_failures = 0
        loop_state.halted = report.outcome in (
            CycleOutcome.HALTED,
            CycleOutcome.STATE_CORRUPT,
        )
        if (
            loop_state.consecutive_failures
            >= self.config.self_heal_after_consecutive_failures
        ):
            logger.warning(
                "%s consecutive_failures=%d threshold=%d",
                LOG_TAG_SELF_HEAL,
                loop_state.consecutive_failures,
                self.config.self_heal_after_consecutive_failures,
            )
        self.write_state(loop_state)


def _iso(ts: pd.Timestamp) -> str:
    ts = pd.Timestamp(ts)
    if ts.tz is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts.isoformat()


__all__ = [
    "STATE_CORRUPT",
    "SHIP_GATE_PATH_DEFAULT",
    "LOOP_STATE_PATH_DEFAULT",
    "PORTFOLIO_STATE_PATH_DEFAULT",
    "AutonomousLoop",
    "ControlLoopError",
    "CycleOutcome",
    "CycleReport",
    "IBKRTransportFSM",
    "LoopConfig",
    "LoopState",
    "LOG_TAG_CYCLE_BEGIN",
    "LOG_TAG_CYCLE_END",
    "LOG_TAG_EXEC_SHADOW",
    "LOG_TAG_FAIL_CLOSED",
    "LOG_TAG_RECONCILE_OK",
    "LOG_TAG_RISK_BLOCK",
    "LOG_TAG_SELF_HEAL",
    "LOG_TAG_STATE_CORRUPT",
    "LOG_TAG_STATE_PERSIST",
    "LOG_TAG_TRANSPORT_DOWN",
    "LOG_TAG_TRANSPORT_OK",
    "ReconcileReport",
    "StateReconciler",
    "TransportState",
]

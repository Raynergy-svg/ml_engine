"""US-017 — Tests for :mod:`src.equity.control_loop`.

No mocks. Real :class:`RebalanceScheduler`, real :class:`RiskAgentLayer`,
real on-disk ship-gate / loop-state / portfolio-state via ``tmp_path``.
The smoke test attaches a FileHandler pointed at
``tmp_path/logs/buddy_debug.log`` so the >=10-cycle grep AC is satisfied
without touching the project-root log file.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List, Mapping, Optional, Tuple

import pandas as pd
import pytest

from src.equity.control_loop import (
    LOG_TAG_CYCLE_BEGIN,
    LOG_TAG_CYCLE_END,
    LOG_TAG_RECONCILE_OK,
    LOG_TAG_STATE_PERSIST,
    LOG_TAG_TRANSPORT_OK,
    STATE_CORRUPT,
    AutonomousLoop,
    ControlLoopError,
    CycleOutcome,
    IBKRTransportFSM,
    LoopConfig,
    StateReconciler,
    TransportState,
)
from src.equity.rebalance import (
    DEFAULT_NO_TRADE_BAND,
    Order,
    RebalanceScheduler,
)
from src.equity.risk_agents import (
    PortfolioState,
    RiskAgentConfig,
    RiskAgentLayer,
)


UNIVERSE_HASH = "abc12345" * 8


# ---------------------------------------------------------------------------
# Fixtures (real classes, real disk, no mocks)
# ---------------------------------------------------------------------------


def _write_ship_gate(
    path: Path,
    *,
    gate_pass: bool = True,
    universe_hash: str = UNIVERSE_HASH,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "gate_pass": bool(gate_pass),
        "net_sharpe": 0.55,
        "max_dd": 0.20,
        "positive_years": 11,
        "total_years": 16,
        "universe_hash": universe_hash,
        "asof": "2026-06-19",
        "recommendation": "PASS — test fixture",
        "criteria_source": "test",
        "thresholds": {},
        "pipeline_version": "2026-06-18-eq1",
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return path


class _StubBroker:
    """Real-class stub broker: no unittest.mock.

    Records every send and accepts every order. The whole class is
    plain Python — no mock library involved.
    """

    def __init__(self) -> None:
        self.sent: List[Order] = []

    def __call__(self, order: Order) -> bool:
        self.sent.append(order)
        return True


class _LiveBookFetcher:
    """Returns a fixed (cash, shares, prices) tuple — pure data."""

    def __init__(
        self,
        *,
        cash: float,
        shares: Mapping[str, float],
        prices: Mapping[str, float],
    ) -> None:
        self.cash = float(cash)
        self.shares = dict(shares)
        self.prices = dict(prices)

    def __call__(self) -> Tuple[float, Mapping[str, float], Mapping[str, float]]:
        return self.cash, dict(self.shares), dict(self.prices)


class _Heartbeat:
    """Toggleable heartbeat — alive flag controlled by the test."""

    def __init__(self, alive: bool = True) -> None:
        self.alive = bool(alive)

    def __call__(self) -> bool:
        return self.alive


def _build_loop(
    tmp_path: Path,
    *,
    target_weights: Mapping[str, float],
    initial_weights: Optional[Mapping[str, float]] = None,
    nav: float = 1_000_000.0,
    alive: bool = True,
    shadow_mode: bool = True,
    max_cycles: Optional[int] = 12,
) -> "tuple[AutonomousLoop, _StubBroker, _Heartbeat, _LiveBookFetcher]":
    gate = tmp_path / "backtests" / "SHIP_GATE.json"
    _write_ship_gate(gate)
    sched_state = tmp_path / "state" / "rebalance.json"
    loop_state = tmp_path / "state" / "loop.json"
    pf_state = tmp_path / "state" / "portfolio.json"

    initial = dict(initial_weights or {})

    scheduler = RebalanceScheduler.create(
        universe_hash=UNIVERSE_HASH,
        ship_gate_path=gate,
        state_path=sched_state,
        no_trade_band=DEFAULT_NO_TRADE_BAND,
        rebalance_freq_days=30,
    )
    risk = RiskAgentLayer.create(
        config=RiskAgentConfig(),
        ship_gate_path=gate,
        universe_hash=UNIVERSE_HASH,
    )

    broker = _StubBroker()
    heartbeat = _Heartbeat(alive=alive)
    fsm = IBKRTransportFSM(heartbeat_fn=heartbeat)

    # Synthesize a live book consistent with the initial weights.
    prices = {t: 100.0 for t in target_weights}
    shares = {t: (initial.get(t, 0.0) * nav) / 100.0 for t in target_weights}
    cash = nav - sum(initial.get(t, 0.0) for t in target_weights) * nav
    fetcher = _LiveBookFetcher(cash=cash, shares=shares, prices=prices)
    reconciler = StateReconciler(
        fetcher=fetcher, equity_drift_bps=50.0, cash_drift_bps=200.0
    )

    config = LoopConfig(
        universe_hash=UNIVERSE_HASH,
        cycle_interval_seconds=0.0,
        max_cycles=max_cycles,
        shadow_mode=shadow_mode,
    )

    portfolio = PortfolioState(
        nav=nav,
        peak_nav=nav,
        current_weights=dict(initial),
    )
    pf_state.parent.mkdir(parents=True, exist_ok=True)
    pf_state.write_text(json.dumps(portfolio.to_dict()))

    # ADV map: 1e9 USD per name is comfortably above 5% participation
    # cap on a $1M NAV (5% * $1M = $50k worst-case notional).
    adv_map = {t: 1.0e9 for t in target_weights}
    loop = AutonomousLoop.create(
        config=config,
        scheduler=scheduler,
        risk_layer=risk,
        transport=fsm,
        reconciler=reconciler,
        get_target_weights=lambda asof: target_weights,
        execute_order=broker,
        state_path=loop_state,
        portfolio_state_path=pf_state,
        ship_gate_path=gate,
        get_adv=lambda asof: adv_map,
    )
    return loop, broker, heartbeat, fetcher


# ---------------------------------------------------------------------------
# AC: Ship-gate guard
# ---------------------------------------------------------------------------


def test_create_refuses_when_ship_gate_missing(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    cfg = LoopConfig(universe_hash=UNIVERSE_HASH)
    with pytest.raises(ControlLoopError, match="file not found"):
        AutonomousLoop.create(
            config=cfg,
            scheduler=object(),  # type: ignore[arg-type]
            risk_layer=object(),  # type: ignore[arg-type]
            transport=IBKRTransportFSM(heartbeat_fn=lambda: True),
            reconciler=StateReconciler(
                fetcher=lambda: (0.0, {}, {})
            ),
            get_target_weights=lambda asof: {},
            execute_order=lambda o: True,
            state_path=state_dir / "loop.json",
            portfolio_state_path=state_dir / "pf.json",
            ship_gate_path=tmp_path / "nope.json",
        )


def test_create_refuses_when_gate_pass_false(tmp_path: Path) -> None:
    gate = tmp_path / "backtests" / "SHIP_GATE.json"
    _write_ship_gate(gate, gate_pass=False)
    cfg = LoopConfig(universe_hash=UNIVERSE_HASH)
    with pytest.raises(ControlLoopError, match="gate_pass"):
        AutonomousLoop.create(
            config=cfg,
            scheduler=object(),  # type: ignore[arg-type]
            risk_layer=object(),  # type: ignore[arg-type]
            transport=IBKRTransportFSM(heartbeat_fn=lambda: True),
            reconciler=StateReconciler(
                fetcher=lambda: (0.0, {}, {})
            ),
            get_target_weights=lambda asof: {},
            execute_order=lambda o: True,
            state_path=tmp_path / "loop.json",
            portfolio_state_path=tmp_path / "pf.json",
            ship_gate_path=gate,
        )


def test_create_refuses_when_universe_hash_mismatched(tmp_path: Path) -> None:
    gate = tmp_path / "backtests" / "SHIP_GATE.json"
    _write_ship_gate(gate, universe_hash="WRONG" * 16)
    cfg = LoopConfig(universe_hash=UNIVERSE_HASH)
    with pytest.raises(ControlLoopError, match="universe_hash mismatch"):
        AutonomousLoop.create(
            config=cfg,
            scheduler=object(),  # type: ignore[arg-type]
            risk_layer=object(),  # type: ignore[arg-type]
            transport=IBKRTransportFSM(heartbeat_fn=lambda: True),
            reconciler=StateReconciler(
                fetcher=lambda: (0.0, {}, {})
            ),
            get_target_weights=lambda asof: {},
            execute_order=lambda o: True,
            state_path=tmp_path / "loop.json",
            portfolio_state_path=tmp_path / "pf.json",
            ship_gate_path=gate,
        )


# ---------------------------------------------------------------------------
# AC: Transport FSM
# ---------------------------------------------------------------------------


def test_transport_fsm_connected_when_heartbeat_alive() -> None:
    fsm = IBKRTransportFSM(heartbeat_fn=lambda: True)
    state = fsm.update()
    assert state is TransportState.CONNECTED
    assert fsm.is_healthy()


def test_transport_fsm_progresses_degraded_reconnecting_down() -> None:
    misses = {"count": 0}

    def hb() -> bool:
        misses["count"] += 1
        return False  # always dead

    reconnect_calls = {"n": 0}

    def reconnect() -> None:
        reconnect_calls["n"] += 1

    fsm = IBKRTransportFSM(
        heartbeat_fn=hb,
        reconnect_fn=reconnect,
        reconnect_after=2,
        down_after=4,
    )
    assert fsm.update() is TransportState.DEGRADED  # 1 miss
    assert fsm.update() is TransportState.RECONNECTING  # 2 misses
    assert reconnect_calls["n"] == 1
    fsm.update()  # 3
    assert fsm.update() is TransportState.DOWN  # 4
    assert not fsm.is_healthy()


def test_transport_recovers_on_alive_heartbeat() -> None:
    flag = {"alive": False}
    fsm = IBKRTransportFSM(heartbeat_fn=lambda: flag["alive"])
    fsm.update()
    flag["alive"] = True
    assert fsm.update() is TransportState.CONNECTED


# ---------------------------------------------------------------------------
# AC: State reconciler — SHARES + CASH, not just NAV
# ---------------------------------------------------------------------------


def test_reconciler_returns_per_name_drift_below_band() -> None:
    fetcher = _LiveBookFetcher(
        cash=500_000.0,
        shares={"AAPL": 5_000.0},  # 5000 * $100 = $500k
        prices={"AAPL": 100.0},
    )
    rc = StateReconciler(fetcher=fetcher, equity_drift_bps=50.0)
    report = rc.reconcile(
        expected_weights={"AAPL": 0.50},
        asof=pd.Timestamp("2026-06-21", tz="UTC"),
    )
    assert report.nav == pytest.approx(1_000_000.0)
    assert report.live_weights["AAPL"] == pytest.approx(0.5)
    assert report.drift_breaches == []


def test_reconciler_flags_per_name_drift_breach() -> None:
    fetcher = _LiveBookFetcher(
        cash=300_000.0,  # mismatched cash
        shares={"AAPL": 7_000.0},  # 7000 * $100 = $700k → 70% live, expect 50
        prices={"AAPL": 100.0},
    )
    rc = StateReconciler(
        fetcher=fetcher, equity_drift_bps=50.0, cash_drift_bps=100.0
    )
    report = rc.reconcile(
        expected_weights={"AAPL": 0.50},
        asof=pd.Timestamp("2026-06-21", tz="UTC"),
    )
    assert "AAPL" in report.drift_breaches
    assert "__CASH__" in report.drift_breaches


def test_reconciler_refuses_non_positive_nav() -> None:
    fetcher = _LiveBookFetcher(
        cash=0.0,
        shares={"AAPL": 0.0},
        prices={"AAPL": 100.0},
    )
    rc = StateReconciler(fetcher=fetcher)
    with pytest.raises(ControlLoopError, match="non-positive NAV"):
        rc.reconcile(
            expected_weights={},
            asof=pd.Timestamp("2026-06-21", tz="UTC"),
        )


# ---------------------------------------------------------------------------
# AC: Corrupt state -> log STATE_CORRUPT, skip execute, stay halted
# ---------------------------------------------------------------------------


def test_corrupt_state_emits_no_order_and_stays_halted(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    loop, broker, _hb, _f = _build_loop(
        tmp_path,
        target_weights={"AAPL": 0.10, "MSFT": 0.10},
        max_cycles=1,
    )
    # Inject a corrupt loop-state file BEFORE the cycle runs.
    loop.state_path.parent.mkdir(parents=True, exist_ok=True)
    loop.state_path.write_text("{this is not json")

    caplog.set_level(logging.ERROR, logger="src.equity.control_loop")
    portfolio = loop.load_portfolio_state()
    assert portfolio is not None
    report = loop.run_cycle(
        asof=pd.Timestamp("2026-06-21", tz="UTC"),
        portfolio_state=portfolio,
    )
    assert report.outcome is CycleOutcome.STATE_CORRUPT
    assert broker.sent == [], "no order emitted when state is corrupt"

    # State on disk must reflect halted=true
    state_disk = json.loads(loop.state_path.read_text())
    assert state_disk["halted"] is True

    # STATE_CORRUPT must appear in the log records
    assert any(
        STATE_CORRUPT in rec.getMessage() for rec in caplog.records
    )


# ---------------------------------------------------------------------------
# AC: Transport down -> FAIL CLOSED, no blind rebalance
# ---------------------------------------------------------------------------


def test_transport_down_fails_closed_no_orders(tmp_path: Path) -> None:
    loop, broker, hb, _f = _build_loop(
        tmp_path,
        target_weights={"AAPL": 0.10, "MSFT": 0.10},
        max_cycles=1,
    )
    # Trip the FSM to DOWN by misses.
    hb.alive = False
    loop.transport.down_after = 1
    portfolio = loop.load_portfolio_state()
    assert portfolio is not None
    report = loop.run_cycle(
        asof=pd.Timestamp("2026-06-21", tz="UTC"),
        portfolio_state=portfolio,
    )
    assert report.outcome is CycleOutcome.TRANSPORT_FAILED_CLOSED
    assert broker.sent == []


# ---------------------------------------------------------------------------
# AC: Graceful restart resumes from persisted state
# ---------------------------------------------------------------------------


def test_restart_loads_cycle_count(tmp_path: Path) -> None:
    loop, _b, _hb, _f = _build_loop(
        tmp_path,
        target_weights={"AAPL": 0.10, "MSFT": 0.10},
        max_cycles=1,
    )
    portfolio = loop.load_portfolio_state()
    assert portfolio is not None
    loop.run_cycle(
        asof=pd.Timestamp("2026-06-21", tz="UTC"),
        portfolio_state=portfolio,
    )
    state, corrupt = loop.load_state()
    assert not corrupt
    assert state.cycle_count == 1


# ---------------------------------------------------------------------------
# AC: Loop runs >=10 cycles end-to-end with stub executor
# ---------------------------------------------------------------------------


def test_loop_runs_full_pipeline_for_10_cycles_and_logs_substrings(
    tmp_path: Path,
) -> None:
    """The 'smoke test' AC: 10+ cycles + grep logs/buddy_debug.log."""
    # Attach a real FileHandler at the canonical path under tmp_path so
    # the grep target is reachable to the test without polluting the
    # project-root log.
    log_path = tmp_path / "logs" / "buddy_debug.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
    )
    target_logger = logging.getLogger("src.equity.control_loop")
    prior_level = target_logger.level
    target_logger.setLevel(logging.DEBUG)
    target_logger.addHandler(handler)
    try:
        loop, broker, _hb, _f = _build_loop(
            tmp_path,
            target_weights={"AAPL": 0.10, "MSFT": 0.10},
            initial_weights={"AAPL": 0.10, "MSFT": 0.10},
            max_cycles=10,
        )
        # Drive the loop with a deterministic clock.
        ticks = {"i": 0}

        def clk() -> pd.Timestamp:
            ticks["i"] += 1
            return pd.Timestamp("2026-06-21", tz="UTC") + pd.Timedelta(
                seconds=ticks["i"]
            )

        reports = loop.run(clock=clk, sleep_fn=lambda _s: None)
        assert len(reports) == 10
    finally:
        handler.close()
        target_logger.removeHandler(handler)
        target_logger.setLevel(prior_level)

    text = log_path.read_text(encoding="utf-8")
    for substring in (
        LOG_TAG_CYCLE_BEGIN,
        LOG_TAG_CYCLE_END,
        LOG_TAG_TRANSPORT_OK,
        LOG_TAG_RECONCILE_OK,
        LOG_TAG_STATE_PERSIST,
    ):
        assert substring in text, f"missing log substring: {substring}"
    # Cycle count must show 10 cycle_end lines exactly.
    assert text.count(LOG_TAG_CYCLE_END) == 10


# ---------------------------------------------------------------------------
# AC: Shadow execution + plan finalization
# ---------------------------------------------------------------------------


def test_shadow_execution_records_and_finalizes_plan(tmp_path: Path) -> None:
    loop, broker, _hb, _f = _build_loop(
        tmp_path,
        target_weights={"AAPL": 0.10, "MSFT": 0.10},
        initial_weights={},  # empty book → plan fires
        max_cycles=1,
    )
    portfolio = loop.load_portfolio_state()
    assert portfolio is not None
    report = loop.run_cycle(
        asof=pd.Timestamp("2026-06-21", tz="UTC"),
        portfolio_state=portfolio,
    )
    assert report.outcome is CycleOutcome.EXECUTED
    assert report.orders_attempted == 2
    assert report.orders_filled == 2
    assert len(broker.sent) == 2
    assert {o.ticker for o in broker.sent} == {"AAPL", "MSFT"}


# ---------------------------------------------------------------------------
# AC: Atomic writes — verify .tmp does not survive
# ---------------------------------------------------------------------------


def test_state_writes_atomically_no_tmp_residue(tmp_path: Path) -> None:
    loop, _b, _hb, _f = _build_loop(
        tmp_path,
        target_weights={"AAPL": 0.10},
        max_cycles=1,
    )
    portfolio = loop.load_portfolio_state()
    assert portfolio is not None
    loop.run_cycle(
        asof=pd.Timestamp("2026-06-21", tz="UTC"),
        portfolio_state=portfolio,
    )
    leftover = list(loop.state_path.parent.glob(".*.tmp"))
    assert leftover == []

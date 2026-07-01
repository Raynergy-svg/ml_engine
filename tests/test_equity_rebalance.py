"""US-007 — Tests for :mod:`src.equity.rebalance`.

No mocks. Real :class:`RebalanceScheduler`, real on-disk ship-gate JSON,
real on-disk state file via ``tmp_path``. The crash-test exercises the
load-bearing AC: kill after 2 of 5 orders → restart sends exactly the
missing 3, no double-fill.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Dict, List

import pandas as pd
import pytest

from src.equity.rebalance import (
    DEFAULT_NO_TRADE_BAND,
    Order,
    OrderStatus,
    RebalanceError,
    RebalancePlan,
    RebalanceScheduler,
    RebalanceState,
    plan_rebalance,
)


UNIVERSE_HASH = "abc12345" * 8  # 64-char placeholder
OTHER_HASH = "fedc6789" * 8


# ---------------------------------------------------------------------------
# Fixtures
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


def _make_scheduler(
    tmp_path: Path,
    *,
    no_trade_band: float = DEFAULT_NO_TRADE_BAND,
    rebalance_freq_days: int = 30,
    gate_pass: bool = True,
    universe_hash: str = UNIVERSE_HASH,
) -> RebalanceScheduler:
    gate = tmp_path / "backtests" / "SHIP_GATE.json"
    _write_ship_gate(gate, gate_pass=gate_pass, universe_hash=universe_hash)
    state = tmp_path / "state" / "rebalance.json"
    return RebalanceScheduler.create(
        universe_hash=UNIVERSE_HASH,
        ship_gate_path=gate,
        state_path=state,
        no_trade_band=no_trade_band,
        rebalance_freq_days=rebalance_freq_days,
    )


class _Broker:
    """A tiny in-memory broker that records fills and (optionally) raises.

    Used by execute_plan() tests in place of real network IO. NOT a mock:
    it is the real ``Callable[[Order], bool]`` contract the scheduler
    consumes, with no patching / no MagicMock.
    """

    def __init__(self, *, raise_after: int | None = None) -> None:
        self.calls: List[str] = []
        self.fills: Dict[str, float] = {}
        self.raise_after = raise_after

    def __call__(self, order: Order) -> bool:
        self.calls.append(order.client_order_id)
        if (
            self.raise_after is not None
            and len(self.calls) > self.raise_after
        ):
            raise RuntimeError("broker connection lost")
        self.fills[order.client_order_id] = order.target_weight
        return True


def _broker_that_dies_after(n: int) -> Callable[[Order], bool]:
    """Broker that fills the first ``n`` orders and then raises.

    Each fill DOES get recorded by the scheduler (it persists after each
    send). The next attempt raises — simulating a kill mid-execute.
    """
    state = {"count": 0, "fills": {}, "calls": []}

    def _send(order: Order) -> bool:
        state["calls"].append(order.client_order_id)
        if state["count"] >= n:
            raise RuntimeError("simulated crash mid-rebalance")
        state["count"] += 1
        state["fills"][order.client_order_id] = order.target_weight
        return True

    _send.state = state  # type: ignore[attr-defined]
    return _send


# ---------------------------------------------------------------------------
# Ship-gate guard — AC: refuses unless gate_pass==true AND hash matches
# ---------------------------------------------------------------------------


def test_ship_gate_guard_blocks_when_missing(tmp_path: Path) -> None:
    missing = tmp_path / "no_such_dir" / "SHIP_GATE.json"
    with pytest.raises(RebalanceError, match="file not found"):
        RebalanceScheduler.create(
            universe_hash=UNIVERSE_HASH,
            ship_gate_path=missing,
            state_path=tmp_path / "state.json",
        )


def test_ship_gate_guard_blocks_when_gate_pass_false(tmp_path: Path) -> None:
    gate = tmp_path / "SHIP_GATE.json"
    _write_ship_gate(gate, gate_pass=False)
    with pytest.raises(RebalanceError, match="gate_pass=False"):
        RebalanceScheduler.create(
            universe_hash=UNIVERSE_HASH,
            ship_gate_path=gate,
            state_path=tmp_path / "state.json",
        )


def test_ship_gate_guard_blocks_on_hash_mismatch(tmp_path: Path) -> None:
    gate = tmp_path / "SHIP_GATE.json"
    _write_ship_gate(gate, gate_pass=True, universe_hash=OTHER_HASH)
    with pytest.raises(RebalanceError, match="universe_hash mismatch"):
        RebalanceScheduler.create(
            universe_hash=UNIVERSE_HASH,
            ship_gate_path=gate,
            state_path=tmp_path / "state.json",
        )


def test_ship_gate_guard_blocks_on_malformed_json(tmp_path: Path) -> None:
    gate = tmp_path / "SHIP_GATE.json"
    gate.write_text("not-json{")
    with pytest.raises(RebalanceError, match="cannot read"):
        RebalanceScheduler.create(
            universe_hash=UNIVERSE_HASH,
            ship_gate_path=gate,
            state_path=tmp_path / "state.json",
        )


def test_ship_gate_guard_accepts_matching_pass(tmp_path: Path) -> None:
    sched = _make_scheduler(tmp_path)
    assert sched.universe_hash == UNIVERSE_HASH
    assert Path(sched.ship_gate_path).exists()


# ---------------------------------------------------------------------------
# plan_rebalance — order delta + no-trade band + closures
# ---------------------------------------------------------------------------


def test_plan_skips_names_inside_no_trade_band() -> None:
    plan = plan_rebalance(
        current_weights={"AAA": 0.10, "BBB": 0.10},
        target_weights={"AAA": 0.101, "BBB": 0.15},
        asof="2026-06-20T00:00:00+00:00",
        universe_hash=UNIVERSE_HASH,
        no_trade_band=0.005,
    )
    tickers = {o.ticker for o in plan.orders}
    # AAA delta = 0.001 < band — skipped.
    # BBB delta = 0.05 > band — included.
    assert tickers == {"BBB"}
    assert plan.orders[0].side == "BUY"


def test_plan_always_closes_a_dropped_name() -> None:
    # Drop BBB to 0 even though delta (0.003) is inside the band — closure
    # is unconditional.
    plan = plan_rebalance(
        current_weights={"AAA": 0.20, "BBB": 0.003},
        target_weights={"AAA": 0.20, "BBB": 0.0},
        asof="2026-06-20T00:00:00+00:00",
        universe_hash=UNIVERSE_HASH,
        no_trade_band=0.005,
    )
    tickers = {o.ticker for o in plan.orders}
    assert tickers == {"BBB"}
    assert plan.orders[0].side == "SELL"
    assert plan.orders[0].target_weight == 0.0


def test_plan_emits_stable_client_order_ids() -> None:
    targets = {"AAA": 0.25, "BBB": 0.30}
    plan_a = plan_rebalance(
        current_weights={},
        target_weights=targets,
        asof="2026-06-20T00:00:00+00:00",
        universe_hash=UNIVERSE_HASH,
    )
    plan_b = plan_rebalance(
        current_weights={},
        target_weights=targets,
        asof="2026-06-20T00:00:00+00:00",
        universe_hash=UNIVERSE_HASH,
    )
    ids_a = sorted(o.client_order_id for o in plan_a.orders)
    ids_b = sorted(o.client_order_id for o in plan_b.orders)
    assert ids_a == ids_b
    # Distinct per-ticker.
    assert len(set(ids_a)) == len(ids_a)


def test_plan_rejects_negative_band() -> None:
    with pytest.raises(RebalanceError, match="no_trade_band"):
        plan_rebalance(
            current_weights={},
            target_weights={"AAA": 0.1},
            asof="2026-06-20T00:00:00+00:00",
            universe_hash=UNIVERSE_HASH,
            no_trade_band=-0.001,
        )


# ---------------------------------------------------------------------------
# should_rebalance — calendar + drift triggers
# ---------------------------------------------------------------------------


def test_should_rebalance_initial_fires(tmp_path: Path) -> None:
    sched = _make_scheduler(tmp_path)
    fire, reason = sched.should_rebalance(
        now=pd.Timestamp("2026-06-20", tz="UTC"),
        current_weights={},
        target_weights={"AAA": 0.5},
        last_rebalance_asof=None,
    )
    assert fire is True
    assert reason == "initial"


def test_should_rebalance_drift_fires(tmp_path: Path) -> None:
    sched = _make_scheduler(tmp_path, no_trade_band=0.005)
    fire, reason = sched.should_rebalance(
        now=pd.Timestamp("2026-06-20", tz="UTC"),
        current_weights={"AAA": 0.10, "BBB": 0.10},
        target_weights={"AAA": 0.10, "BBB": 0.50},  # 40pp drift
        last_rebalance_asof="2026-06-19T00:00:00+00:00",
    )
    assert fire is True
    assert reason == "drift"


def test_should_rebalance_calendar_fires_after_freq_days(tmp_path: Path) -> None:
    sched = _make_scheduler(tmp_path, rebalance_freq_days=30)
    fire, reason = sched.should_rebalance(
        now=pd.Timestamp("2026-07-20", tz="UTC"),
        current_weights={"AAA": 0.50},
        target_weights={"AAA": 0.50},  # zero drift
        last_rebalance_asof="2026-06-19T00:00:00+00:00",
    )
    assert fire is True
    assert reason == "calendar"


def test_should_rebalance_quiet_within_band_and_window(tmp_path: Path) -> None:
    sched = _make_scheduler(tmp_path, no_trade_band=0.005, rebalance_freq_days=30)
    fire, reason = sched.should_rebalance(
        now=pd.Timestamp("2026-06-25", tz="UTC"),
        current_weights={"AAA": 0.50},
        target_weights={"AAA": 0.501},  # 0.1pp drift, inside band
        last_rebalance_asof="2026-06-19T00:00:00+00:00",
    )
    assert fire is False
    assert reason == "none"


# ---------------------------------------------------------------------------
# State persistence — survives restart
# ---------------------------------------------------------------------------


def test_state_persists_across_reload(tmp_path: Path) -> None:
    sched = _make_scheduler(tmp_path)
    state = sched.load_state()
    state.last_rebalance_asof = "2026-06-19T00:00:00+00:00"
    state.current_actual_weights = {"AAA": 0.25, "BBB": 0.25}
    sched.write_state(state)

    # New scheduler instance reading the same files.
    sched2 = RebalanceScheduler.create(
        universe_hash=UNIVERSE_HASH,
        ship_gate_path=sched.ship_gate_path,
        state_path=sched.state_path,
    )
    loaded = sched2.load_state()
    assert loaded.last_rebalance_asof == "2026-06-19T00:00:00+00:00"
    assert loaded.current_actual_weights == {"AAA": 0.25, "BBB": 0.25}


def test_state_atomic_write_no_tmp_residue(tmp_path: Path) -> None:
    sched = _make_scheduler(tmp_path)
    state = sched.load_state()
    state.current_actual_weights = {"AAA": 0.5}
    sched.write_state(state)
    state_dir = Path(sched.state_path).parent
    # No leftover .tmp file.
    leftovers = [p for p in state_dir.iterdir() if ".tmp" in p.name]
    assert leftovers == []


# ---------------------------------------------------------------------------
# Full happy path — plan, execute, finalize
# ---------------------------------------------------------------------------


def test_plan_and_execute_happy_path(tmp_path: Path) -> None:
    sched = _make_scheduler(tmp_path)
    targets = {f"T{i:02d}": 0.10 for i in range(5)}
    plan = sched.plan_and_persist(
        target_weights=targets,
        asof=pd.Timestamp("2026-06-20", tz="UTC"),
    )
    assert len(plan.orders) == 5
    broker = _Broker()
    touched = sched.execute_plan(plan, broker)
    assert len(touched) == 5
    assert all(o.is_filled for o in plan.orders)
    sched.finalize_plan(plan)

    final = sched.load_state()
    assert final.active_plan is None
    assert final.last_rebalance_asof == plan.asof
    assert final.current_actual_weights == targets


def test_plan_and_persist_refuses_overwrite(tmp_path: Path) -> None:
    sched = _make_scheduler(tmp_path)
    sched.plan_and_persist(
        target_weights={"AAA": 0.5},
        asof=pd.Timestamp("2026-06-20", tz="UTC"),
    )
    with pytest.raises(RebalanceError, match="active_plan already exists"):
        sched.plan_and_persist(
            target_weights={"AAA": 0.5},
            asof=pd.Timestamp("2026-06-21", tz="UTC"),
        )


def test_finalize_refuses_unfilled_plan(tmp_path: Path) -> None:
    sched = _make_scheduler(tmp_path)
    plan = sched.plan_and_persist(
        target_weights={"AAA": 0.5, "BBB": 0.5},
        asof=pd.Timestamp("2026-06-20", tz="UTC"),
    )
    # Fill only one of the two orders.
    plan.orders[0].status = OrderStatus.FILLED.value
    plan.orders[0].fill_weight = plan.orders[0].target_weight
    with pytest.raises(RebalanceError, match="non-filled orders"):
        sched.finalize_plan(plan)


# ---------------------------------------------------------------------------
# Idempotent restart — load-bearing AC test
# ---------------------------------------------------------------------------


def test_restart_after_two_of_five_sends_exactly_missing_three(
    tmp_path: Path,
) -> None:
    """AC: kill after 2 of 5 orders → restart sends exactly the missing 3,
    no double-fill.
    """
    sched = _make_scheduler(tmp_path)
    targets = {f"T{i:02d}": 0.10 for i in range(5)}
    plan = sched.plan_and_persist(
        target_weights=targets,
        asof=pd.Timestamp("2026-06-20", tz="UTC"),
    )
    assert len(plan.orders) == 5
    first_two_ids = [o.client_order_id for o in plan.orders[:2]]
    last_three_ids = [o.client_order_id for o in plan.orders[2:]]

    # First process: dies on the 3rd send (after 2 successful fills).
    dying_broker = _broker_that_dies_after(2)
    with pytest.raises(RuntimeError, match="simulated crash"):
        sched.execute_plan(plan, dying_broker)
    assert len(dying_broker.state["fills"]) == 2  # type: ignore[attr-defined]

    # Restart: NEW scheduler, NEW broker, only the state file connects them.
    sched2 = RebalanceScheduler.create(
        universe_hash=UNIVERSE_HASH,
        ship_gate_path=sched.ship_gate_path,
        state_path=sched.state_path,
    )
    resumed = sched2.reconcile_and_resume()
    assert resumed is not None
    assert resumed.rebalance_id == plan.rebalance_id
    # Two filled, three remaining.
    assert len(resumed.filled_orders()) == 2
    assert len(resumed.remaining_orders()) == 3
    filled_ids = {o.client_order_id for o in resumed.filled_orders()}
    assert filled_ids == set(first_two_ids)

    # The resumed plan re-sends only the missing three.
    new_broker = _Broker()
    touched = sched2.execute_plan(resumed, new_broker)
    assert [o.client_order_id for o in touched] == last_three_ids
    assert new_broker.calls == last_three_ids
    # No double-fill: the broker never saw the first two IDs.
    assert set(new_broker.calls).isdisjoint(set(first_two_ids))

    sched2.finalize_plan(resumed)
    final = sched2.load_state()
    assert final.active_plan is None
    assert final.last_rebalance_asof == resumed.asof
    assert final.current_actual_weights == targets


def test_client_order_ids_stable_across_restart_replan(tmp_path: Path) -> None:
    """Same (asof, targets) on restart produce the same client_order_ids.

    Guarantees the broker can dedupe across restarts even if state was
    lost: the deterministic ID is the safety net under the state file.
    """
    sched = _make_scheduler(tmp_path)
    targets = {"AAA": 0.25, "BBB": 0.30}
    asof = pd.Timestamp("2026-06-20", tz="UTC")
    plan_a = sched.plan_and_persist(target_weights=targets, asof=asof)
    ids_a = sorted(o.client_order_id for o in plan_a.orders)

    # Force clear state and replan against an empty actual book.
    state = sched.load_state()
    state.active_plan = None
    sched.write_state(state)

    plan_b = sched.plan_and_persist(target_weights=targets, asof=asof)
    ids_b = sorted(o.client_order_id for o in plan_b.orders)
    assert ids_a == ids_b


def test_reconcile_returns_none_when_no_active_plan(tmp_path: Path) -> None:
    sched = _make_scheduler(tmp_path)
    assert sched.reconcile_and_resume() is None


def test_reconcile_clears_corrupt_active_plan(tmp_path: Path) -> None:
    sched = _make_scheduler(tmp_path)
    state = sched.load_state()
    state.active_plan = {"this": "is", "not": "a plan"}
    sched.write_state(state)
    resumed = sched.reconcile_and_resume()
    assert resumed is None
    assert sched.load_state().active_plan is None


# ---------------------------------------------------------------------------
# Round-trip serialisation
# ---------------------------------------------------------------------------


def test_plan_round_trip_via_dict() -> None:
    plan = plan_rebalance(
        current_weights={},
        target_weights={"AAA": 0.4, "BBB": 0.6},
        asof="2026-06-20T00:00:00+00:00",
        universe_hash=UNIVERSE_HASH,
    )
    plan.orders[0].status = OrderStatus.FILLED.value
    plan.orders[0].fill_weight = plan.orders[0].target_weight
    payload = plan.to_dict()
    revived = RebalancePlan.from_dict(payload)
    assert revived.rebalance_id == plan.rebalance_id
    assert [o.client_order_id for o in revived.orders] == [
        o.client_order_id for o in plan.orders
    ]
    assert revived.orders[0].is_filled
    assert not revived.orders[1].is_filled


def test_state_round_trip_via_dict() -> None:
    state = RebalanceState(
        last_rebalance_asof="2026-06-20T00:00:00+00:00",
        current_actual_weights={"AAA": 0.25},
    )
    revived = RebalanceState.from_dict(state.to_dict())
    assert revived.last_rebalance_asof == state.last_rebalance_asof
    assert revived.current_actual_weights == state.current_actual_weights
    assert revived.active_plan is None

"""Shadow-only entry glue for the equity-beta harvester (N2, 2026-06-24).

The harvester library (``harvester_strategy`` + ``rebalance`` + ``ship_gate``)
is complete but orphaned — nothing in production instantiates it. This module
is the missing, NON-hot-path bootstrapper for a single SHADOW rebalance tick:

    UniverseSnapshot + prices
      -> HarvesterStrategy.create(...)            (ship-gate guarded)
      -> compute_target_weights(asof)
      -> RebalanceScheduler.plan_and_persist(...) (writes rebalance_state.json)
      -> execute_plan(..., shadow execute_order)  (deterministic shadow fills)

It writes ``trained_data/equity/rebalance_state.json`` so the TUI "HARVESTER
REBALANCE PLAN" panel (``DataProvider.get_rebalance_status``) renders a real
plan. It NEVER contacts a broker and NEVER places a real order — the shadow
``execute_order`` callback marks each order filled in simulation only.

Fail-closed guards (both mirror the FX halt discipline):
  * refuse unless ``config.enable_equity_harvester`` is True;
  * refuse while the global ``.claude/state.json`` ``halted`` flag is True.

Hot-path isolation: this module imports nothing from ``src.scanner.execution``,
``main``, or the FX scanner runtime. Driving it from a live process (a
``scripts/`` entrypoint or a ``main.py`` subcommand) is the separate,
operator-authorized H1 step — it is intentionally NOT wired here.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from src.equity.decision_gate import decide_cycle
from src.equity.harvester_strategy import HarvesterStrategy
from src.equity.rebalance import (
    STATE_PATH_DEFAULT as _REBALANCE_STATE_REL,
    Order,
    RebalancePlan,
    RebalanceScheduler,
)

if TYPE_CHECKING:  # pragma: no cover - typing only, avoids heavy/circular imports
    import pandas as pd

    from src.equity.universe import UniverseSnapshot
    from src.scanner.config import ScannerConfig

logger = logging.getLogger(__name__)

_DEFAULT_SHIP_GATE_REL = "trained_data/backtests/SHIP_GATE.json"
_EQUITY_STATE_DIR_REL = "trained_data/equity"


@dataclass(frozen=True)
class ShadowRunResult:
    """Observable outcome of one shadow tick.

    ``ran`` is False for a fail-closed refusal; ``reason`` is one of
    ``"disabled"`` / ``"halted"`` / ``"executed"``. ``plan`` is the persisted
    :class:`RebalancePlan` on an executed tick, else ``None``.
    """

    ran: bool
    reason: str
    plan: Optional[RebalancePlan]


def _shadow_execute_order(order: Order) -> bool:
    """Deterministic SHADOW fill: no broker, every order treated as filled.

    Returning True tells :meth:`RebalanceScheduler.execute_plan` to mark the
    order FILLED. This is a simulation policy, not a broker call — switching to
    real fills is the operator-authorized live step (H3), never reachable here.
    """
    logger.info(
        "AUTONOMOUS_LOOP exec_shadow ticker=%s side=%s target_weight=%.4f",
        order.ticker,
        order.side,
        order.target_weight,
    )
    return True


def run_shadow_rebalance(
    *,
    config: "ScannerConfig",
    snapshot: "UniverseSnapshot",
    prices: "pd.DataFrame",
    asof: "pd.Timestamp",
    project_root: Path = Path("."),
    now: Optional[datetime] = None,
) -> ShadowRunResult:
    """Run one shadow rebalance tick; persist the plan for the TUI panel.

    Fail-closed via the Pillar-3 decision gate (:func:`decide_cycle`): refuses
    (``ran=False``) unless the harvester is enabled AND the gate returns
    CONTINUE — global halt -> REFUSE, drawdown breach -> HALT, ship-gate
    fail/mismatch -> NO_ACT, stale data -> ABSTAIN. On CONTINUE it builds the
    ship-gate-guarded strategy + scheduler, computes target weights, persists
    the plan, and shadow-fills it — never touching a broker.
    """
    root = Path(project_root)
    ship_gate_path = root / getattr(
        config, "equity_harvester_ship_gate_path", _DEFAULT_SHIP_GATE_REL
    )
    state_path = root / _REBALANCE_STATE_REL

    # Guard 1 — master enable flag (default OFF, fail-closed).
    if not getattr(config, "enable_equity_harvester", False):
        logger.warning(
            "equity harvester shadow tick REFUSED — enable_equity_harvester is "
            "False (set the flag to run the shadow lane)"
        )
        return ShadowRunResult(ran=False, reason="disabled", plan=None)

    # Guard 2 — objective decision gate, re-derived from disk (Pillar 3).
    # Subsumes the global-halt check and adds drawdown / ship-gate / staleness
    # rails. Self-report is not trusted; only CONTINUE is actionable.
    index = getattr(prices, "index", None)
    data_asof = index.max() if (index is not None and len(index) > 0) else None
    decision = decide_cycle(
        config=config,
        project_root=root,
        data_asof=data_asof,
        now=now,
        snapshot_universe_hash=snapshot.universe_hash,
    )
    if not decision.actionable:
        logger.warning(
            "equity harvester shadow tick %s — reasons=%s",
            decision.decision.value.upper(),
            ",".join(decision.reasons),
        )
        return ShadowRunResult(ran=False, reason=decision.decision.value, plan=None)

    (root / _EQUITY_STATE_DIR_REL).mkdir(parents=True, exist_ok=True)

    strategy = HarvesterStrategy.create(
        snapshot=snapshot, prices=prices, ship_gate_path=ship_gate_path
    )
    target_weights = strategy.compute_target_weights(asof)

    scheduler = RebalanceScheduler.create(
        universe_hash=snapshot.universe_hash,
        ship_gate_path=ship_gate_path,
        state_path=state_path,
    )
    # Re-invocable: clear a prior FULLY-FILLED plan so a repeated tick (the
    # shadow "loop") re-plans instead of crashing on the active_plan guard. A
    # partially-filled prior plan is left intact (fail-closed) — only the
    # operator-authorized H1 driver owns partial-fill resume.
    prior = scheduler.reconcile_and_resume()
    if prior is not None and not prior.remaining_orders():
        scheduler.finalize_plan(prior)
    plan = scheduler.plan_and_persist(target_weights=target_weights, asof=asof)
    scheduler.execute_plan(plan, _shadow_execute_order)
    logger.info(
        "equity harvester shadow tick EXECUTED — orders=%d state=%s",
        len(plan.orders),
        state_path,
    )
    return ShadowRunResult(ran=True, reason="executed", plan=plan)

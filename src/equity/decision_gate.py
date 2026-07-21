"""Pillar 3 — objective per-cycle decision gate for the equity harvester.

`decide_cycle` re-derives the harvester's per-cycle action purely from disk, so
the decision is driven by observable reality rather than the runner's
self-report (anti-falsification discipline, cf. L-007). It returns one of:

    REFUSE   — global ``.claude/state.json`` halt is set (do not even consider)
    HALT     — portfolio drawdown breached ``dd_hard`` (or risk state unreadable)
    NO_ACT   — ship gate missing / failing / universe-hash mismatch
    ABSTAIN  — market data is stale (older than the freshness window)
    CONTINUE — none of the above; the cycle may plan/execute (shadow)

Precedence is most-severe-first (REFUSE > HALT > NO_ACT > ABSTAIN > CONTINUE):
the gate short-circuits on the first applicable rail and reports its reason.
Every disk read is fail-closed — an unreadable risk file resolves to the
restrictive side, never silently to CONTINUE.

Shadow-only / NON-hot-path: imports nothing from ``src.scanner.execution``,
``main``, or ``embedded_scanner``; the FX lane is untouched.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Tuple

from src.equity.data_loader import DEFAULT_FRESHNESS_DAYS
from src.equity.risk_agents import PortfolioState, RiskAgentError

if TYPE_CHECKING:  # pragma: no cover - typing only
    import pandas as pd

    from src.scanner.config import ScannerConfig

logger = logging.getLogger(__name__)

_DEFAULT_SHIP_GATE_REL = "trained_data/backtests/SHIP_GATE.json"
_PORTFOLIO_STATE_REL = "trained_data/equity/portfolio_state.json"
_DEFAULT_DD_HARD = 0.20


class EquityDecision(str, Enum):
    """Per-cycle decision, ordered most-severe-first."""

    REFUSE = "refuse"
    HALT = "halt"
    NO_ACT = "no_act"
    ABSTAIN = "abstain"
    CONTINUE = "continue"


@dataclass(frozen=True)
class CycleDecision:
    """Disk-derived verdict for one harvester cycle."""

    decision: EquityDecision
    reasons: Tuple[str, ...]

    @property
    def actionable(self) -> bool:
        """Only CONTINUE permits planning/execution."""
        return self.decision is EquityDecision.CONTINUE

    def to_dict(self) -> dict:
        return {
            "decision": self.decision.value,
            "reasons": list(self.reasons),
            "actionable": self.actionable,
        }


def _global_halt(root: Path) -> Tuple[bool, bool]:
    """Return ``(halted, readable)`` from ``<root>/.claude/state.json``.

    Read directly (not via the fail-OPEN ``StateEngine.get_halted``) so a
    present-but-corrupt halt file is reported NOT readable -> the caller fails
    CLOSED (REFUSE). An absent file is a readable "no halt set".

    Unchanged by per-lane support: this is the exact global-only check every
    existing caller (``decide_cycle(lane=None)``) still gets.
    """
    path = root / ".claude" / "state.json"
    if not path.exists():
        return (False, True)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return (False, False)
    if not isinstance(payload, dict):
        return (False, False)
    return (bool(payload.get("halted", False)), True)


def _lane_halted(root: Path, lane: str) -> Tuple[bool, bool, str]:
    """Return ``(halted, readable, reason)`` for one lane at
    ``<root>/.claude/state.json``.

    Mirrors ``_global_halt``'s direct, fail-closed-on-corruption read (never
    goes through the fail-OPEN ``StateEngine``), but strengthens the
    "absent state" default: unlike ``_global_halt``'s "absent file -> not
    halted" bootstrap convention, a lane that was never explicitly told
    "not halted" — via a missing file, a missing ``halted_lanes`` dict, or a
    missing/non-bool entry for this lane — is treated as HALTED. A lane must
    be explicitly unhalted; it is never assumed safe to run by omission.
    """
    path = root / ".claude" / "state.json"
    if not path.exists():
        return (True, True, "lane_not_configured")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return (True, False, "state_unreadable")
    if not isinstance(payload, dict):
        return (True, False, "state_unreadable")
    if bool(payload.get("halted", False)):
        return (True, True, "global_halt")
    lanes = payload.get("halted_lanes")
    if not isinstance(lanes, dict) or not isinstance(lanes.get(lane), bool):
        return (True, True, "lane_not_configured")
    lane_value = bool(lanes[lane])
    return (lane_value, True, f"lane_halted:{lane}" if lane_value else "")


def _drawdown_breach(root: Path, dd_hard: float) -> Optional[str]:
    """Return a reason if portfolio drawdown breaches ``dd_hard``.

    Absent file -> no positions -> no drawdown risk (None). A present-but-corrupt
    file is fail-closed: we cannot verify risk, so we report a breach reason.
    """
    path = root / _PORTFOLIO_STATE_REL
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        state = PortfolioState.from_dict(payload)
    except (OSError, json.JSONDecodeError, RiskAgentError, TypeError) as exc:
        logger.warning("portfolio_state unreadable (%s) — fail-closed HALT", exc)
        return f"portfolio_state_unreadable:{type(exc).__name__}"
    dd = state.drawdown()
    if dd >= dd_hard - 1e-9:  # tolerance: an exact dd_hard must breach (IEEE-754)
        return f"drawdown_breach:{dd:.4f}>=dd_hard:{dd_hard:.4f}"
    return None


def _ship_gate_block(root: Path, ship_rel: str, expected_hash: Optional[str]) -> Optional[str]:
    """Return a reason if the ship gate is missing, failing, or hash-mismatched."""
    path = root / ship_rel
    if not path.exists():
        return "ship_gate_missing"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return f"ship_gate_unreadable:{type(exc).__name__}"
    if not isinstance(payload, dict):
        return "ship_gate_malformed"
    if payload.get("gate_pass") is not True:
        return "ship_gate_fail"
    if expected_hash is not None and str(payload.get("universe_hash")) != str(expected_hash):
        return "universe_hash_mismatch"
    return None


def _stale(data_asof: "Optional[pd.Timestamp]", now: datetime, freshness_days: int) -> Optional[str]:
    """Return a reason if data is stale, absent, or its asof is unparseable.

    Fail-closed: ``None`` data_asof, ``NaT``, or any unparseable timestamp
    resolves to ``no_data`` (-> ABSTAIN) rather than skipping the rail.
    """
    if data_asof is None:
        return "no_data"
    try:
        last = data_asof.to_pydatetime()
        if last.tzinfo is None:
            last = last.replace(tzinfo=timezone.utc)
        age_days = (now.astimezone(timezone.utc) - last.astimezone(timezone.utc)).days
    except (AttributeError, ValueError, TypeError):
        return "no_data"  # NaT / non-timestamp asof -> fail-closed abstain
    if age_days > freshness_days:
        return f"stale_data:{age_days}d>{freshness_days}d"
    return None


def decide_cycle(
    *,
    config: "ScannerConfig",
    project_root: Path = Path("."),
    data_asof: "Optional[pd.Timestamp]" = None,
    now: Optional[datetime] = None,
    snapshot_universe_hash: Optional[str] = None,
    lane: Optional[str] = None,
) -> CycleDecision:
    """Re-derive the cycle decision from disk (most-severe rail wins).

    ``lane`` (default ``None``): every existing caller keeps getting the
    exact legacy global-only halt check (``_global_halt``, unreadable/absent
    handled as documented there). Passing a lane name (e.g. ``"equity"``)
    switches the halt rail to the fail-closed per-lane check (``_lane_halted``)
    — see that function's docstring for the fail-closed defaults.
    """
    root = Path(project_root)
    now = now or datetime.now(timezone.utc)
    dd_hard = float(getattr(config, "equity_harvester_dd_hard", _DEFAULT_DD_HARD))
    freshness = int(
        getattr(config, "equity_harvester_freshness_days", DEFAULT_FRESHNESS_DAYS)
    )
    ship_rel = getattr(
        config, "equity_harvester_ship_gate_path", _DEFAULT_SHIP_GATE_REL
    )

    # 1. global/lane halt — most severe; fail CLOSED on a corrupt halt file.
    if lane is None:
        halted, readable = _global_halt(root)
        reason = "global_halt"
    else:
        halted, readable, reason = _lane_halted(root, lane)
    if not readable:
        return CycleDecision(EquityDecision.REFUSE, ("state_unreadable",))
    if halted:
        return CycleDecision(EquityDecision.REFUSE, (reason,))

    # 2. drawdown breach — emergency stop (operator must intervene).
    dd_reason = _drawdown_breach(root, dd_hard)
    if dd_reason is not None:
        return CycleDecision(EquityDecision.HALT, (dd_reason,))

    # 3. ship gate — cannot act on an unvalidated / failing strategy.
    sg_reason = _ship_gate_block(root, ship_rel, snapshot_universe_hash)
    if sg_reason is not None:
        return CycleDecision(EquityDecision.NO_ACT, (sg_reason,))

    # 4. stale data — skip this cycle, retry when fresh.
    stale_reason = _stale(data_asof, now, freshness)
    if stale_reason is not None:
        return CycleDecision(EquityDecision.ABSTAIN, (stale_reason,))

    return CycleDecision(EquityDecision.CONTINUE, ())

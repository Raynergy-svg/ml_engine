"""Per-invocation brain-loop orchestrator.

Order of operations, every cycle:
  1. ``get_halted(lane="brain")`` — if already halted (globally or the
     "brain" lane specifically), refuse everything. No hypothesis work
     happens while halted, full stop.
  2. ``monitor.evaluate_rails()`` — if a NEW drawdown breach is detected,
     de-risk immediately (``derisk.halt()``) and skip hypothesis work this
     cycle. De-risk always wins over research.
  3. If (and only if) neither of the above fired, and a hypothesis was
     supplied this cycle: register it (frozen), run it through the real gate,
     and — only on a CONTINUE verdict — propose SHADOW promotion (auto; no
     capital, no ARM). LIVE promotion is never proposed here — that decision
     belongs to a human reviewing a PASS result, not to this loop.

This module is deliberately the only place that composes the other five —
each of hypothesis_registry / gate_runner / promotion / monitor / derisk stays
independently testable and independently safe.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Mapping, Optional

from src.brain_loop import derisk, gate_runner, monitor, promotion
from src.brain_loop.hypothesis_registry import register_hypothesis
from src.scanner.automation.state_engine import StateEngine

if TYPE_CHECKING:  # pragma: no cover - typing only
    import pandas as pd

    from src.scanner.config import ScannerConfig


@dataclass(frozen=True)
class CycleResult:
    halted: bool
    breach_derisked: bool
    derisk_action: Optional[Dict[str, Any]]
    hypothesis_result: Optional[Dict[str, Any]]
    rails: Dict[str, Any]


def run_cycle(
    *,
    project_root: Path,
    config: "ScannerConfig",
    ledger_path: Path,
    requests_dir: Path,
    state_path: Optional[Path] = None,
    hypothesis: Optional[Mapping[str, Any]] = None,
    data_asof: "Optional[pd.Timestamp]" = None,
    now: Optional[datetime] = None,
    actor: str = "brain_loop",
) -> CycleResult:
    project_root = Path(project_root)
    state_path = Path(state_path) if state_path else project_root / ".claude" / "state.json"

    if StateEngine(state_path).get_halted(lane="brain"):
        return CycleResult(
            halted=True,
            breach_derisked=False,
            derisk_action=None,
            hypothesis_result=None,
            rails={"decision": "refuse", "reasons": ["already_halted"]},
        )

    rails = monitor.evaluate_rails(
        config=config, project_root=project_root, data_asof=data_asof, now=now
    )

    if rails.breach:
        action = derisk.halt(
            state_path,
            reason="; ".join(rails.breach_reasons) or "rail_breach",
            actor=actor,
        )
        return CycleResult(
            halted=True,
            breach_derisked=True,
            derisk_action=action,
            hypothesis_result=None,
            rails=rails.to_dict(),
        )

    hypothesis_result: Optional[Dict[str, Any]] = None
    if hypothesis is not None:
        hypothesis_id = str(hypothesis["hypothesis_id"])
        register_hypothesis(
            ledger_path,
            hypothesis_id=hypothesis_id,
            name=hypothesis["name"],
            mechanism=hypothesis["mechanism"],
            novelty_justification=hypothesis["novelty_justification"],
            params=hypothesis["params"],
        )
        gate_summary = gate_runner.run_and_gate(
            hypothesis_id=hypothesis_id,
            ledger_path=ledger_path,
            harness_cmd=hypothesis["harness_cmd"],
            config=config,
            project_root=project_root,
            data_asof=data_asof,
        )
        promotion_path: Optional[str] = None
        if gate_summary["decision"]["decision"] == "continue":
            path = promotion.propose_promotion(
                requests_dir,
                hypothesis_id=hypothesis_id,
                decision=gate_summary["decision"],
                target="shadow",
            )
            promotion_path = str(path)
        hypothesis_result = {**gate_summary, "promotion_path": promotion_path}

    return CycleResult(
        halted=False,
        breach_derisked=False,
        derisk_action=None,
        hypothesis_result=hypothesis_result,
        rails=rails.to_dict(),
    )

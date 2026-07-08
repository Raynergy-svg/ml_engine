"""Runs the existing backtest harness and defers every gate decision to the
existing, already-validated decision gate. This module computes NO statistics
itself — Sharpe/DSR/drawdown all come from the harness output and from
``src.equity.decision_gate.decide_cycle`` (Pillar 3), which already composes
halt / drawdown / ship-gate / freshness in fail-closed, most-severe-first
order. Reusing it here (rather than re-deriving gate logic) is the point: the
brain loop's job is orchestration and judgment, not re-implementing math that
is already gated and tested elsewhere.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional, Sequence

from src.brain_loop.hypothesis_registry import record_gate_verdict
from src.equity.decision_gate import CycleDecision, decide_cycle

if TYPE_CHECKING:  # pragma: no cover - typing only
    import pandas as pd

    from src.scanner.config import ScannerConfig

_DEFAULT_TIMEOUT_S = 1800.0
_STDOUT_TAIL_CHARS = 4000


class UnsafeHarnessCommand(ValueError):
    """Raised when a hypothesis-supplied harness_cmd fails the executable/path allowlist."""


def _validate_harness_cmd(harness_cmd: Sequence[str], *, cwd: Path) -> None:
    """Fail closed unless argv[0] is a Python interpreter and argv[1] resolves to a
    ``.py`` file inside this project's ``scripts/`` directory.

    ``harness_cmd`` ultimately traces back to hypothesis JSON that a scheduled Claude
    session (or, per PROMPT.md, eventually an operator) writes — it is not raw
    end-user input, but it is still data, not code the developer wrote. Passing it
    to ``subprocess.run`` unchecked would let a malformed/compromised hypothesis file
    run an arbitrary binary (CWE-78). This does not re-introduce the joint-fallback
    or ship-gate concerns — it only constrains *which command* can run, never what
    the backtest computes.
    """
    if len(harness_cmd) < 2:
        raise UnsafeHarnessCommand("harness_cmd must be [python_interpreter, script_path, ...args]")
    # Resolve argv[0] to a real filesystem path (via PATH lookup if relative/bare)
    # and require it to be the CURRENT interpreter, not merely basename-matched --
    # `Path(x).name in {"python", "python3", ...}` would accept any binary an
    # attacker names /tmp/evil/python3, defeating the whole allowlist.
    interpreter_arg = str(harness_cmd[0])
    interpreter_path = Path(interpreter_arg)
    if not interpreter_path.is_absolute():
        which = shutil.which(interpreter_arg)
        interpreter_path = Path(which) if which else interpreter_path
    if interpreter_path.resolve() != Path(sys.executable).resolve():
        raise UnsafeHarnessCommand(
            f"harness_cmd[0] must resolve to the current python interpreter "
            f"({sys.executable}), got {harness_cmd[0]!r}"
        )
    scripts_dir = (Path(cwd) / "scripts").resolve()
    script_path = (Path(cwd) / str(harness_cmd[1])).resolve()
    if script_path.suffix != ".py" or scripts_dir not in script_path.parents:
        raise UnsafeHarnessCommand(
            f"harness_cmd[1] must be a .py file inside {scripts_dir}, got {harness_cmd[1]!r}"
        )


def run_backtest(
    harness_cmd: Sequence[str],
    *,
    cwd: Path,
    timeout: float = _DEFAULT_TIMEOUT_S,
) -> subprocess.CompletedProcess:
    """Run the caller-supplied harness command as a real subprocess.

    ``harness_cmd`` is validated first (see ``_validate_harness_cmd``) so hypothesis
    data can only select *which script under scripts/* runs, never an arbitrary
    command. No mocking, no re-implementation of what the harness computes. Raises
    ``subprocess.TimeoutExpired`` on hang (fail loud, not silently) rather than
    swallowing a stuck backtest.
    """
    _validate_harness_cmd(harness_cmd, cwd=cwd)
    return subprocess.run(  # noqa: S603 - harness_cmd is validated above (fixed interpreter + scripts/ allowlist)
        list(harness_cmd),
        cwd=str(cwd),
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def evaluate_gate(
    *,
    config: "ScannerConfig",
    project_root: Path,
    snapshot_universe_hash: Optional[str] = None,
    data_asof: "Optional[pd.Timestamp]" = None,
    now: Optional[datetime] = None,
) -> CycleDecision:
    """Thin, explicit re-export of ``decide_cycle`` — the one canonical gate."""
    return decide_cycle(
        config=config,
        project_root=Path(project_root),
        data_asof=data_asof,
        now=now,
        snapshot_universe_hash=snapshot_universe_hash,
    )


def run_and_gate(
    *,
    hypothesis_id: str,
    ledger_path: Path,
    harness_cmd: Sequence[str],
    config: "ScannerConfig",
    project_root: Path,
    cwd: Optional[Path] = None,
    timeout: float = _DEFAULT_TIMEOUT_S,
    snapshot_universe_hash: Optional[str] = None,
    data_asof: "Optional[pd.Timestamp]" = None,
) -> Dict[str, Any]:
    """Run the harness, evaluate the canonical gate, record the verdict.

    Requires ``hypothesis_id`` to already be registered (raises
    ``HypothesisNotRegistered`` otherwise) — a gate verdict can never attach to
    a hypothesis that was never frozen.
    """
    proc = run_backtest(harness_cmd, cwd=cwd or Path(project_root), timeout=timeout)
    decision = evaluate_gate(
        config=config,
        project_root=project_root,
        snapshot_universe_hash=snapshot_universe_hash,
        data_asof=data_asof,
    )
    ledger_record = record_gate_verdict(
        ledger_path, hypothesis_id, decision=decision.decision.value, reasons=decision.reasons
    )
    return {
        "harness_returncode": proc.returncode,
        "harness_stdout_tail": proc.stdout[-_STDOUT_TAIL_CHARS:],
        "harness_stderr_tail": proc.stderr[-_STDOUT_TAIL_CHARS:],
        "decision": decision.to_dict(),
        "ledger_record": ledger_record,
    }

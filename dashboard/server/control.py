"""AXIOM Phase 2 — bounded control router (operator control path to the practice bot).

Mounted by app.py ONLY when ``AXIOM_CONTROL_ENABLED`` is truthy (default OFF → 404).
Every handler: requires an explicit per-action ``X-AXIOM-Confirm`` header, runs the
structural guards in ``control_safety.enforce`` BEFORE any effect, and audit-logs the
attempt (allowed AND denied). See dashboard/CONTROL_DESIGN.md.

Functional actions (all practice-pinned + bounded by control_safety):
  halt               — StateEngine.set_halted(True) (fail-safe; always allowed).
  unhalt             — StateEngine.set_halted(False) ONLY after assert_unhalt_eligible
                       passes (practice + drawdown<20% + gates GREEN + models fresh).
  set_gross_leverage — writes a clamped [0,15] override; the trend loop sizes to it
                       and OANDA scanner execution enforces it as a gross cap.
  start_loop/stop_loop — fixed-WHITELIST process control (no arbitrary exec): trend ->
                       run_oanda_trend.py --loop, tier7 -> run_tier7_loop.py; stop by pid.
"""
from __future__ import annotations

import logging
import os
import shlex
import signal
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel

from dashboard.server import control_safety as cs

logger = logging.getLogger("axiom.control")
router = APIRouter(prefix="/api/control", tags=["control"])

REPO_ROOT = Path(__file__).resolve().parents[2]
LOG_DIR = REPO_ROOT / "trained_data" / "axiom"

# FIXED command + process-needle per whitelisted loop. argv is fully fixed by the loop
# name (validated against LOOP_WHITELIST) — NO user input reaches the process args, NO
# shell. This is the only place the control layer spawns a process.
LOOP_CMDS: Dict[str, list[str]] = {
    "trend": [sys.executable, str(REPO_ROOT / "scripts" / "run_oanda_trend.py"), "--loop", "3600"],
    "tier7": [sys.executable, str(REPO_ROOT / "scripts" / "run_tier7_loop.py")],
}
LOOP_NEEDLES: Dict[str, str] = {
    "trend": "run_oanda_trend.py",
    "tier7": "run_tier7_loop.py",
}


class ActionBody(BaseModel):
    params: Dict[str, Any] = {}


def _confirm(action: str, header_val: Optional[str]) -> None:
    if header_val != action:
        raise HTTPException(status_code=403, detail="missing/incorrect X-AXIOM-Confirm header")


def _loop_pids(loop: str) -> list[int]:
    """Return exact loop process pids for a whitelisted loop.

    Avoid ``pgrep -f`` substring matches: shell commands, tests, or grep/rg probes
    can mention ``run_tier7_loop.py`` without being the loop the control panel can
    actually start/stop.
    """
    script_name = Path(LOOP_NEEDLES[loop]).name
    try:
        out = subprocess.run(["ps", "-axo", "pid=,command="], capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.SubprocessError):
        return []
    pids = []
    for line in out.stdout.splitlines():
        try:
            pid_txt, command = line.strip().split(None, 1)
            pid = int(pid_txt)
        except ValueError:
            continue
        if pid == os.getpid():
            continue
        try:
            argv = shlex.split(command)
        except ValueError:
            argv = command.split()
        exe = (Path(argv[0]).name if argv else "").lower()
        exe_is_python = exe.startswith("python")
        exe_is_script = exe == script_name
        script_arg = any(Path(arg).name == script_name for arg in argv[1:])
        if exe_is_script or (exe_is_python and script_arg):
            pids.append(pid)
    return pids


def _start_loop(loop: str) -> Dict[str, Any]:
    if _loop_pids(loop):
        return {"result": "already_running", "pids": _loop_pids(loop)}
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logf = open(LOG_DIR / f"{loop}_loop.out", "a")  # noqa: SIM115 — handed to the child
    proc = subprocess.Popen(  # noqa: S603 — fixed argv from LOOP_CMDS, no shell, no user input
        LOOP_CMDS[loop], cwd=str(REPO_ROOT), stdout=logf, stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    return {"result": "started", "pid": proc.pid}


def _stop_loop(loop: str) -> Dict[str, Any]:
    pids = _loop_pids(loop)
    for pid in pids:
        try:
            os.kill(pid, signal.SIGTERM)
        except OSError:
            pass
    return {"result": "stopped" if pids else "not_running", "pids": pids}


def _state() -> Dict[str, Any]:
    """Current persisted control state for dashboard hydration/readback."""
    from src.scanner.automation.state_engine import StateEngine

    overrides = cs.read_overrides()
    loops = {}
    for loop in sorted(LOOP_CMDS):
        pids = _loop_pids(loop)
        loops[loop] = {"running": bool(pids), "pids": pids}
    return {
        "ok": True,
        "environment": cs.assert_practice(),
        "halted": StateEngine().get_halted(),
        "gross_leverage": overrides.get("gross_leverage"),
        "override_updated_at": overrides.get("_updated_at"),
        "leverage_cap": cs.LEVERAGE_CAP,
        "loops": loops,
    }


def _run(action: str, params: Dict[str, Any]):
    """Guard → audit → effect. Denials are audited and surfaced as 403."""
    try:
        normalized = cs.enforce(action, params)
        if action == "unhalt":
            normalized["eligibility"] = cs.assert_unhalt_eligible()  # extra gate; raises if ineligible
    except cs.ControlDenied as exc:
        cs.audit({"action": action, "params": params, "allowed": False, "reason": str(exc), "result": "denied"})
        raise HTTPException(status_code=403, detail=str(exc))

    # --- effects (only reached after ALL immutables + eligibility passed) ---
    from src.scanner.automation.state_engine import StateEngine
    if action == "halt":
        StateEngine().set_halted(True)
        result: Dict[str, Any] = {"result": "halted"}
    elif action == "unhalt":
        StateEngine().set_halted(False)  # eligibility already enforced above
        result = {"result": "unhalted", "eligibility": normalized.get("eligibility")}
    elif action == "set_gross_leverage":
        ov = cs.set_override("gross_leverage", normalized["gross_leverage"])
        result = {"result": "leverage_set", "gross_leverage": ov["gross_leverage"]}
    elif action == "start_loop":
        result = _start_loop(normalized["loop"])
    elif action == "stop_loop":
        result = _stop_loop(normalized["loop"])
    else:  # unreachable — enforce() allowlist guarantees one of the above
        raise HTTPException(status_code=500, detail="unhandled action")

    cs.audit({"action": action, "params": normalized, "allowed": True, "reason": "guards passed", **result})
    logger.warning("AXIOM CONTROL executed: %s -> %s", action, result.get("result"))
    return {"ok": True, "action": action, **result, "state": _state()}


@router.get("/state")
def state():
    """Read-only control state: persisted dials + loop/halt readback."""
    return _state()


@router.get("/audit")
def audit_log(limit: int = 50):
    """Read-only tail of the control audit trail (most recent first)."""
    path = cs.AUDIT_PATH
    if not path.exists():
        return {"entries": [], "count": 0}
    import json
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            rows.append(json.loads(line))
        except ValueError:
            continue
    rows.reverse()
    return {"entries": rows[: int(limit)], "count": len(rows)}


@router.post("/halt")
def halt(body: Optional[ActionBody] = None, *, x_axiom_confirm: Optional[str] = Header(default=None)):
    _confirm("halt", x_axiom_confirm)
    return _run("halt", body.params if body else {})


@router.post("/unhalt")
def unhalt(body: Optional[ActionBody] = None, *, x_axiom_confirm: Optional[str] = Header(default=None)):
    _confirm("unhalt", x_axiom_confirm)
    return _run("unhalt", body.params if body else {})


@router.post("/set_gross_leverage")
def set_gross_leverage(body: Optional[ActionBody] = None, *, x_axiom_confirm: Optional[str] = Header(default=None)):
    _confirm("set_gross_leverage", x_axiom_confirm)
    return _run("set_gross_leverage", body.params if body else {})


@router.post("/start_loop")
def start_loop(body: Optional[ActionBody] = None, *, x_axiom_confirm: Optional[str] = Header(default=None)):
    _confirm("start_loop", x_axiom_confirm)
    return _run("start_loop", body.params if body else {})


@router.post("/stop_loop")
def stop_loop(body: Optional[ActionBody] = None, *, x_axiom_confirm: Optional[str] = Header(default=None)):
    _confirm("stop_loop", x_axiom_confirm)
    return _run("stop_loop", body.params if body else {})

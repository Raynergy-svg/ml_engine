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
  flatten            — closes EVERY open position on the practice account. The ONLY
                       action that calls a broker-mutating endpoint directly (all others
                       either write local state/files or spawn a whitelisted process).
                       Uses the same practice-pinned OandaPracticeClient the trend loop
                       itself uses (no live URL constant exists in that client — there is
                       no path to real money regardless of this action). Each position's
                       close is audited individually so a partial failure is visible.
"""
from __future__ import annotations

import fcntl
import logging
import os
import shlex
import signal
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import APIRouter, Header, HTTPException, Request
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


def _actor(request: Request, x_axiom_actor: Optional[str]) -> str:
    """Attribution (investigation finding, 2026-07-01: previously NO actor field
    at all). Prefers an explicit ``X-AXIOM-Actor`` header (the frontend can send
    a stable per-browser-session id); falls back to the client IP the ASGI
    server sees; "unknown" only if neither is available (never raises —
    attribution is best-effort, must never itself block a control action)."""
    if x_axiom_actor:
        return str(x_axiom_actor)[:128]  # bounded length — this goes straight to disk
    try:
        if request.client and request.client.host:
            return f"ip:{request.client.host}"
    except Exception:  # noqa: BLE001 — attribution must never crash the request
        pass
    return "unknown"


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
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    # C-B3: serialize check-and-spawn under a per-loop file lock. Without it, two
    # concurrent start_loop requests (a double-click within the ~3s poll-lag window)
    # both pass the pid check and spawn TWO trading loops on the same account. The
    # lock fd is close-on-exec by default (PEP 446), so the child never holds it.
    lock_path = LOG_DIR / f"{loop}.start.lock"
    with open(lock_path, "w") as lockf:  # noqa: SIM115
        try:
            fcntl.flock(lockf, fcntl.LOCK_EX)
        except OSError:
            pass  # lock unavailable -> proceed; pid check below still guards best-effort
        existing = _loop_pids(loop)
        if existing:
            return {"result": "already_running", "pids": existing}
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


def _flatten_all() -> Dict[str, Any]:
    """Close every open position on the practice account.

    The ONLY control action that calls a broker-mutating endpoint directly. Uses the
    SAME practice-pinned client the trend loop itself uses for execution — no live URL
    constant exists in OandaPracticeClient, so there is no path to real money regardless
    of this action running. Each position's close is attempted + audited individually:
    one failure must not hide (or block) the others, and the caller sees exactly which
    instruments closed and which didn't.
    """
    from src.utils.oanda_practice import OandaPracticeClient

    client = OandaPracticeClient.from_env()
    positions_resp = client.get_open_positions() or {}
    open_positions = positions_resp.get("positions") or []
    results = []
    for p in open_positions:
        inst = p.get("instrument")
        if not inst:
            continue
        try:
            client.close_position(instrument=inst)
            results.append({"instrument": inst, "ok": True})
        except Exception as exc:  # noqa: BLE001 — one failed close must not hide the rest
            results.append({"instrument": inst, "ok": False, "error": str(exc)})
    all_ok = all(r["ok"] for r in results) if results else True
    return {"result": "flattened" if all_ok else "flatten_partial",
            "count": len(results), "positions": results}


def _state() -> Dict[str, Any]:
    """Current persisted control state for dashboard hydration/readback."""
    from src.scanner.automation.state_engine import StateEngine

    overrides = cs.read_overrides()
    loops = {}
    for loop in sorted(LOOP_CMDS):
        pids = _loop_pids(loop)
        loops[loop] = {"running": bool(pids), "pids": pids}
    arm_state = cs.read_arm_state()
    return {
        "ok": True,
        "environment": cs.assert_practice(),
        "halted": StateEngine().get_halted(),
        "gross_leverage": overrides.get("gross_leverage"),
        "override_updated_at": overrides.get("_updated_at"),
        "leverage_cap": cs.LEVERAGE_CAP,
        "loops": loops,
        "armed": cs.is_armed(arm_state),
        "arm_expires_at": arm_state.get("expires_at"),
        "armed_by": arm_state.get("actor"),
    }


def _run(action: str, params: Dict[str, Any], *, actor: str):
    """Guard → audit → effect. Denials are audited and surfaced as 403."""
    try:
        normalized = cs.enforce(action, params)
        if action == "unhalt":
            normalized["eligibility"] = cs.assert_unhalt_eligible()  # extra gate; raises if ineligible
    except cs.ControlDenied as exc:
        cs.audit({"action": action, "params": params, "allowed": False, "reason": str(exc),
                  "result": "denied"}, actor=actor)
        raise HTTPException(status_code=403, detail=str(exc))

    # --- effects (only reached after ALL immutables + eligibility passed) ---
    # C-B1: the effect is wrapped so a failure AFTER the guards pass (Popen OSError,
    # disk-full on set_halted/override) is AUDITED, not silently lost. Previously audit
    # ran only after a successful effect, so a mutating attempt that then raised left no
    # record. Every attempt is now audited: denied (above), effect_error (here), or ok.
    from src.scanner.automation.state_engine import StateEngine
    try:
        if action == "halt":
            # C-B2: idempotent check-and-set — report already_halted instead of blindly
            # re-flipping on a double-submit (UI disable lags the 3s poll).
            eng = StateEngine()
            if eng.get_halted():
                result: Dict[str, Any] = {"result": "already_halted", "changed": False}
            else:
                eng.set_halted(True)
                result = {"result": "halted", "changed": True}
        elif action == "unhalt":
            eng = StateEngine()  # eligibility already enforced above
            if not eng.get_halted():
                result = {"result": "already_unhalted", "changed": False,
                          "eligibility": normalized.get("eligibility")}
            else:
                eng.set_halted(False)
                result = {"result": "unhalted", "changed": True,
                          "eligibility": normalized.get("eligibility")}
        elif action == "set_gross_leverage":
            ov = cs.set_override("gross_leverage", normalized["gross_leverage"])
            result = {"result": "leverage_set", "gross_leverage": ov["gross_leverage"]}
        elif action == "start_loop":
            result = _start_loop(normalized["loop"])
        elif action == "stop_loop":
            result = _stop_loop(normalized["loop"])
        elif action == "flatten":
            result = _flatten_all()
        elif action == "arm":
            state = cs.set_arm_state(True, actor)
            result = {"result": "armed", "expires_at": state.get("expires_at")}
        elif action == "disarm":
            cs.set_arm_state(False, actor)
            result = {"result": "disarmed"}
        else:  # unreachable — enforce() allowlist guarantees one of the above
            raise HTTPException(status_code=500, detail="unhandled action")
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001 — guards passed but the effect failed
        cs.audit({"action": action, "params": normalized, "allowed": True,
                  "reason": "guards passed", "result": "effect_error", "error": repr(exc)},
                 actor=actor)
        logger.error("AXIOM CONTROL effect FAILED: %s -> %r", action, exc)
        raise HTTPException(status_code=500, detail=f"control effect failed: {type(exc).__name__}")

    cs.audit({"action": action, "params": normalized, "allowed": True, "reason": "guards passed",
             **result}, actor=actor)
    logger.warning("AXIOM CONTROL executed: %s -> %s (actor=%s)", action, result.get("result"), actor)
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
def halt(request: Request, body: Optional[ActionBody] = None, *,
         x_axiom_confirm: Optional[str] = Header(default=None),
         x_axiom_actor: Optional[str] = Header(default=None)):
    _confirm("halt", x_axiom_confirm)
    return _run("halt", body.params if body else {}, actor=_actor(request, x_axiom_actor))


@router.post("/unhalt")
def unhalt(request: Request, body: Optional[ActionBody] = None, *,
           x_axiom_confirm: Optional[str] = Header(default=None),
           x_axiom_actor: Optional[str] = Header(default=None)):
    _confirm("unhalt", x_axiom_confirm)
    return _run("unhalt", body.params if body else {}, actor=_actor(request, x_axiom_actor))


@router.post("/set_gross_leverage")
def set_gross_leverage(request: Request, body: Optional[ActionBody] = None, *,
                       x_axiom_confirm: Optional[str] = Header(default=None),
                       x_axiom_actor: Optional[str] = Header(default=None)):
    _confirm("set_gross_leverage", x_axiom_confirm)
    return _run("set_gross_leverage", body.params if body else {}, actor=_actor(request, x_axiom_actor))


@router.post("/start_loop")
def start_loop(request: Request, body: Optional[ActionBody] = None, *,
               x_axiom_confirm: Optional[str] = Header(default=None),
               x_axiom_actor: Optional[str] = Header(default=None)):
    _confirm("start_loop", x_axiom_confirm)
    return _run("start_loop", body.params if body else {}, actor=_actor(request, x_axiom_actor))


@router.post("/stop_loop")
def stop_loop(request: Request, body: Optional[ActionBody] = None, *,
              x_axiom_confirm: Optional[str] = Header(default=None),
              x_axiom_actor: Optional[str] = Header(default=None)):
    _confirm("stop_loop", x_axiom_confirm)
    return _run("stop_loop", body.params if body else {}, actor=_actor(request, x_axiom_actor))


@router.post("/flatten")
def flatten(request: Request, body: Optional[ActionBody] = None, *,
            x_axiom_confirm: Optional[str] = Header(default=None),
            x_axiom_actor: Optional[str] = Header(default=None)):
    _confirm("flatten", x_axiom_confirm)
    return _run("flatten", body.params if body else {}, actor=_actor(request, x_axiom_actor))


@router.post("/arm")
def arm(request: Request, body: Optional[ActionBody] = None, *,
        x_axiom_confirm: Optional[str] = Header(default=None),
        x_axiom_actor: Optional[str] = Header(default=None)):
    """Deliberately ARM controls (investigation finding, 2026-07-01): flatten,
    set_gross_leverage, start_loop, and unhalt are refused as structural no-ops
    until this is called — arming is never automatic and always expires
    (see control_safety.ARM_TTL_SECONDS)."""
    _confirm("arm", x_axiom_confirm)
    return _run("arm", body.params if body else {}, actor=_actor(request, x_axiom_actor))


@router.post("/disarm")
def disarm(request: Request, body: Optional[ActionBody] = None, *,
           x_axiom_confirm: Optional[str] = Header(default=None),
           x_axiom_actor: Optional[str] = Header(default=None)):
    _confirm("disarm", x_axiom_confirm)
    return _run("disarm", body.params if body else {}, actor=_actor(request, x_axiom_actor))

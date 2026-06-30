"""AXIOM Phase 2 — server-side control IMMUTABLES (structural, pre-effect guards).

This is the load-bearing safety layer for the (disabled) control surface. Every control
action passes through ``enforce()`` BEFORE any mutation. The guarantees here are
structural, not UI-hidden: a crafted request cannot reach live / real-money / a Hard-NO,
because there is simply no parameter or code path to get there.

Immutables (see dashboard/CONTROL_DESIGN.md):
  I1 practice-only   — assert_practice() re-derives env each call; no env/url/account param.
  I2 no real-money   — only the practice-pinned client is ever used; no URL is constructed.
  I3 no Hard-NO relax — only the bounded action set; leverage clamped; unhalt needs practice.
  I4 no promotion    — there is no promote action at all.
"""
from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

REPO_ROOT = Path(__file__).resolve().parents[2]
AUDIT_PATH = REPO_ROOT / "trained_data" / "axiom" / "control_audit.jsonl"
# Control-override file the OANDA trend loop and scanner execution path read.
OVERRIDES_PATH = REPO_ROOT / "trained_data" / "axiom" / "control_overrides.json"

# The ONLY actions that exist. Anything else is unknown -> denied.
ALLOWED_ACTIONS = frozenset({"halt", "unhalt", "set_gross_leverage", "start_loop", "stop_loop"})
LEVERAGE_CAP = 15.0          # hard cap; requests above this are refused, never clamped-up
LOOP_WHITELIST = frozenset({"trend", "tier7"})  # named loops only; no arbitrary exec


class ControlDenied(RuntimeError):
    """Raised when a control request violates an immutable / bound. Maps to HTTP 403."""


def assert_practice() -> str:
    """I1/I2: re-derive oanda_environment from ScannerConfig; refuse unless practice."""
    from src.scanner.config import ScannerConfig

    env = getattr(ScannerConfig(), "oanda_environment", "practice")
    if env != "practice":
        raise ControlDenied(f"HARD LINE: control refused — oanda_environment={env!r} (practice-only).")
    return env


def validate_leverage(value: Any) -> float:
    """I3: leverage must be a finite number in [0, LEVERAGE_CAP]; else refuse."""
    try:
        x = float(value)
    except (TypeError, ValueError):
        raise ControlDenied("leverage must be a number")
    if x != x or x in (float("inf"), float("-inf")):
        raise ControlDenied("leverage must be finite")
    if x < 0 or x > LEVERAGE_CAP:
        raise ControlDenied(f"leverage {x} outside [0, {LEVERAGE_CAP}] cap")
    return x


def validate_loop(name: Any) -> str:
    if name not in LOOP_WHITELIST:
        raise ControlDenied(f"unknown loop {name!r}; allowed: {sorted(LOOP_WHITELIST)}")
    return str(name)


UNHALT_DD_HARD = 0.20            # refuse unhalt if NAV drawdown from peak >= 20%
UNHALT_MAX_AGE_DAYS = 7.0        # refuse unhalt if routed models older than this
UNHALT_VERDICT_MAX_AGE_S = 86400.0  # refuse unhalt if the verify-gate verdict is >24h stale


def _read_json_safe(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return None


def assert_unhalt_eligible() -> Dict[str, Any]:
    """Refuse unhalt unless it is SAFE to resume PRACTICE trading.

    Composite eligibility (all must pass): practice (re-asserted), NAV drawdown below
    the 20% rail, verify_gate verdict GREEN (Hard-NOs intact), and routed models not
    stale (>7d). Returns the checks dict on success; raises ControlDenied with the
    blocking reasons otherwise. Resuming a bleeding/broken/ungated system is the exact
    risk the deferred gate guarded against — this makes unhalt safe to expose.
    """
    assert_practice()  # never resume on a non-practice env
    checks: Dict[str, Any] = {}
    reasons = []

    # 0) lane data recency (INFORMATIONAL — surfaced, not blocked). A hard block here
    # would deadlock the unhalt→start_loop resume flow (a stopped lane has a stale
    # snapshot precisely when you want to unhalt then restart it). Surfaced so the
    # operator sees lane liveness at the moment of unhalt.
    import time as _time
    acct_path = REPO_ROOT / "trained_data" / "oanda" / "account_state.json"
    try:
        checks["account_snapshot_age_s"] = round(_time.time() - acct_path.stat().st_mtime, 1)
    except OSError:
        checks["account_snapshot_age_s"] = None

    # 1) NAV drawdown rail (peak_nav vs current NAV) — FAIL-CLOSED (S-A1).
    # If drawdown CANNOT be verified (missing/zero/corrupt nav or peak_nav), REFUSE.
    # Previously dd stayed None and NO reason was appended, so a missing peak_nav.json
    # silently removed the entire drawdown rail. Unverifiable rail => block.
    acct = _read_json_safe(acct_path) or {}
    peak = (_read_json_safe(REPO_ROOT / "trained_data" / "oanda" / "peak_nav.json") or {}).get("peak_nav")
    nav = acct.get("nav")
    dd = None
    if isinstance(nav, (int, float)) and isinstance(peak, (int, float)) and peak > 0 and nav > 0:
        dd = max(0.0, (peak - nav) / peak)
        if dd >= UNHALT_DD_HARD:
            reasons.append(f"NAV drawdown {dd:.1%} >= {UNHALT_DD_HARD:.0%} rail")
    else:
        reasons.append("NAV drawdown rail UNVERIFIABLE (missing/invalid nav or peak_nav) — fail-closed")
    checks["drawdown_pct"] = dd

    # 2) verify_gate verdict GREEN + FRESH (all Hard-NO checks ok) — FAIL-CLOSED (S-A1).
    # A missing/empty verdict, ANY failed check, an unknown age, or a >24h-stale verdict
    # ALL block unhalt with an explicit reason. Previously an empty/missing verdict set
    # gates_green=False but appended NO reason (the append was guarded by `if vchecks`),
    # so unhalt PROCEEDED while reporting gates_green:False. Re-run /verify-task to refresh
    # verdict.json before unhalting; a stale "GREEN" must not authorize resuming trading.
    verdict_path = REPO_ROOT / ".claude" / "loop" / "verdict.json"
    verdict = _read_json_safe(verdict_path) or {}
    vchecks = verdict.get("checks") or []
    gates_green = bool(vchecks) and all(c.get("ok") for c in vchecks)
    checks["gates_green"] = gates_green
    verdict_age_s = None
    try:
        verdict_age_s = round(_time.time() - verdict_path.stat().st_mtime, 1)
    except OSError:
        verdict_age_s = None
    checks["verdict_age_s"] = verdict_age_s
    if not vchecks:
        reasons.append("verify_gate verdict missing/empty — fail-closed (cannot confirm Hard-NOs)")
    elif not gates_green:
        failed = [c.get("name") for c in vchecks if not c.get("ok")]
        reasons.append(f"verify_gate not GREEN: {failed}")
    elif verdict_age_s is None:
        reasons.append("verify_gate verdict age unknown — fail-closed")
    elif verdict_age_s > UNHALT_VERDICT_MAX_AGE_S:
        reasons.append(f"verify_gate verdict stale ({int(verdict_age_s)}s > {int(UNHALT_VERDICT_MAX_AGE_S)}s) — re-run verify")

    # 3) routed model freshness — INFORMATIONAL only, does NOT block.
    # The live strategy is the non-ML TREND lane; the FX/ML transformers are RETIRED
    # (L-016) and permanently stale by design, so blocking unhalt on their age would
    # make unhalt impossible forever. We surface the age for visibility but never gate on it.
    try:
        from src.scanner.automation.model_freshness import get_model_freshness_for_pairs
        fr = get_model_freshness_for_pairs()
        checks["oldest_model_age_days"] = fr.get("oldest_age_days")
    except Exception:  # noqa: BLE001
        checks["oldest_model_age_days"] = None

    if reasons:
        raise ControlDenied("unhalt blocked — " + "; ".join(reasons))
    return checks


def enforce(action: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """Run ALL pre-effect guards. Returns normalized params or raises ControlDenied.

    Rejects any attempt to smuggle an environment/url/account override (no such field is
    ever honored), re-asserts practice, validates the action + its bounded params.
    """
    if action not in ALLOWED_ACTIONS:
        raise ControlDenied(f"unknown action {action!r}")
    # Refuse loudly if a request tries to smuggle a live/env/url/account override.
    forbidden = {"environment", "env", "url", "base_url", "account", "account_id", "oanda_environment"}
    smuggled = forbidden.intersection(params or {})
    if smuggled:
        raise ControlDenied(f"forbidden parameter(s): {sorted(smuggled)} — environment is immutable")

    assert_practice()  # I1/I2 — every action, no exceptions

    out: Dict[str, Any] = {}
    if action == "set_gross_leverage":
        out["gross_leverage"] = validate_leverage((params or {}).get("gross_leverage"))
    elif action in ("start_loop", "stop_loop"):
        out["loop"] = validate_loop((params or {}).get("loop"))
    # halt / unhalt take no params; unhalt's practice re-assert already happened above.
    return out


def set_override(key: str, value: Any) -> Dict[str, Any]:
    """Atomically merge a clamped control override into OVERRIDES_PATH (the trend loop
    reads it each cycle). Only known, bounded keys are accepted."""
    if key != "gross_leverage":
        raise ControlDenied(f"unknown override {key!r}")
    OVERRIDES_PATH.parent.mkdir(parents=True, exist_ok=True)
    current = _read_json_safe(OVERRIDES_PATH) or {}
    current[key] = value
    current["_updated_at"] = datetime.now(timezone.utc).isoformat()
    current["_source"] = "axiom_control"
    fd, tmp = tempfile.mkstemp(dir=str(OVERRIDES_PATH.parent), suffix=".tmp")
    with os.fdopen(fd, "w") as fh:
        json.dump(current, fh, indent=2, sort_keys=True)
    os.replace(tmp, OVERRIDES_PATH)
    return current


def read_overrides() -> Dict[str, Any]:
    """Read persisted operator overrides. Missing/corrupt files mean no override."""
    data = _read_json_safe(OVERRIDES_PATH)
    return data if isinstance(data, dict) else {}


def audit(entry: Dict[str, Any]) -> None:
    """Append-only, atomic audit line. EVERY attempt (allowed or denied) is recorded."""
    entry = {"ts": datetime.now(timezone.utc).isoformat(), **entry}
    AUDIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(entry, sort_keys=True) + "\n"
    # atomic-ish append: write tmp + concatenate is overkill for a log; append+fsync is fine.
    with open(AUDIT_PATH, "a", encoding="utf-8") as fh:
        fh.write(line)
        fh.flush()
        os.fsync(fh.fileno())

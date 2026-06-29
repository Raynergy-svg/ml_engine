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

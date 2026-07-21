"""The ONLY brain_loop module allowed to touch live-affecting state.

Exactly two capabilities exist here, both de-escalation-only:

  * :func:`halt` — sets ``halted=True`` via the real ``StateEngine``. This is
    NOT a new capability: autonomous ``set_halted(True)`` circuit breakers
    already exist at ``src/scanner/execution.py:6307`` and
    ``src/scanner/engine.py:5626``. This function is a new, audited CALLER of
    that precedented pattern — it has no ``value`` parameter, so it cannot be
    used to unhalt even by mistake.
  * :func:`tighten_risk_config` / :func:`apply_risk_tightening` — moves only
    the pre-clamped ``risk_pct`` knob down, reusing
    ``src.equity.trend_risk_gates.clamp_risk_pct`` so the result can never
    fall outside the existing [MIN_RISK_PCT, MAX_RISK_PCT] bounds and can
    never increase risk (only ``abs(delta)`` is ever subtracted).

This module contains NO reference to flatten / set_gross_leverage / start_loop
/ unhalt / arming — not as a policy choice but because the functions simply
are not imported or defined here (see
tests/test_brain_loop_capability_absence.py).
"""
from __future__ import annotations

import json
import math
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from src.equity.trend_risk_gates import DEFAULT_RISK_PCT, MIN_RISK_PCT, clamp_risk_pct
from src.scanner.automation.state_engine import StateEngine

_DEFAULT_AUDIT_REL = ".claude/brain_loop_audit.jsonl"


def _read_json_safe(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return None


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True)
        os.replace(tmp, path)
    except OSError:
        tmp.unlink(missing_ok=True)
        raise


def _audit(entry: Dict[str, Any], *, actor: str, audit_path: Path) -> None:
    """Append-only audit line. ``actor`` is a REQUIRED keyword argument —
    mirrors dashboard/server/control_safety.py's audit() contract: a missing
    actor is a TypeError at call time, not a silent gap."""
    line = {"ts": datetime.now(timezone.utc).isoformat(), "actor": str(actor), **entry}
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    with open(audit_path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(line, sort_keys=True) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def halt(
    state_path: Path,
    *,
    reason: str,
    actor: str,
    audit_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """De-escalate: set halted=True. Cannot unhalt — no such parameter exists.

    Reuses the exact autonomous circuit-breaker pattern already precedented in
    this codebase (execution.py / engine.py auto-halt), so this is a new
    caller of an existing, already-safe capability, not a new one.
    """
    engine = StateEngine(Path(state_path))
    engine.set_halted(True)
    engine.set_last_actor(actor)
    entry = {"action": "halt", "reason": str(reason)}
    _audit(entry, actor=actor, audit_path=Path(audit_path) if audit_path else Path(state_path).parent / "brain_loop_audit.jsonl")
    return {**entry, "actor": actor}


def tighten_risk_config(current_risk_pct: float, *, delta: float) -> float:
    """Return a risk_pct that is strictly <= ``current_risk_pct`` (modulo the
    existing clamp floor), reusing the SAME bounds the live risk gate already
    enforces. ``delta`` is always treated as a reduction magnitude — a
    negative delta cannot be used to sneak an increase.

    ``clamp_risk_pct`` treats any non-positive OR non-finite value as "invalid
    input, use DEFAULT_RISK_PCT" (0.01) — correct for its normal caller (a
    config override), but WRONG for a de-risk tightening: a large enough delta
    on an already-at-floor risk_pct would land on 0 or below and get bounced
    UP to the 0.01 default instead of DOWN to the 0.005 floor. A NaN delta or
    NaN current_risk_pct is a second way to reach that same fallback — under
    IEEE 754, ``nan <= 0`` is ALWAYS False, so a naive "reduced <= 0" guard
    lets NaN slip through to ``clamp_risk_pct(nan)`` -> 0.01 (security-review
    finding, 2026-07-01: this is a real, exploitable risk-INCREASE via a
    NaN-producing delta, e.g. an LLM-computed ratio that divides by zero).
    Explicitly floor at MIN_RISK_PCT on ANY non-finite or non-positive result,
    so tightening can never increase risk regardless of malformed input.
    """
    reduced = float(current_risk_pct) - abs(float(delta))
    if not math.isfinite(reduced) or reduced <= 0:
        return MIN_RISK_PCT
    return clamp_risk_pct(reduced)


def apply_risk_tightening(
    risk_overrides_path: Path,
    *,
    delta: float,
    actor: str,
    current_risk_pct: Optional[float] = None,
    audit_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Persist a bounded risk_pct tightening, atomically, audited.

    Reads the current override (or DEFAULT_RISK_PCT if none/unreadable/
    ``current_risk_pct`` not supplied), tightens it, and writes back.
    """
    path = Path(risk_overrides_path)
    current = _read_json_safe(path) or {}
    base = (
        float(current_risk_pct)
        if current_risk_pct is not None
        else float(current.get("risk_pct", DEFAULT_RISK_PCT))
    )
    new_value = tighten_risk_config(base, delta=delta)
    current["risk_pct"] = new_value
    current["_updated_at"] = datetime.now(timezone.utc).isoformat()
    current["_actor"] = str(actor)
    _atomic_write_json(path, current)
    _audit(
        {"action": "tighten_risk_config", "risk_pct": new_value, "prior_risk_pct": base},
        actor=actor,
        audit_path=Path(audit_path) if audit_path else path.parent / "brain_loop_audit.jsonl",
    )
    return current

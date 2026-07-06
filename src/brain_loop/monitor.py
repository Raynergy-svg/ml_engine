"""Live-book monitoring for the brain loop — read-only. Evaluates rails using
the EXISTING, already-tested thresholds (``decide_cycle``'s drawdown/ship-gate/
halt precedence); does not invent new risk math. Tier 7's own state
(``.claude/tier7_state.json``) is read for informational context only — the
brain loop never competes with Tier 7's bounded self-heal, so a stale or
missing tier7_state is reported as unknown, never treated as a rail itself.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple

from src.equity.decision_gate import EquityDecision, decide_cycle

if TYPE_CHECKING:  # pragma: no cover - typing only
    import pandas as pd

    from src.scanner.config import ScannerConfig

_TIER7_STATE_REL = ".claude/tier7_state.json"
# Same freshness convention as .claude/loop/running_status.py's HEARTBEAT_FRESH_S.
_TIER7_STALE_S = 90.0


@dataclass(frozen=True)
class RailReport:
    """Everything the brain loop needs to decide whether to de-risk."""

    decision: str
    reasons: Tuple[str, ...]
    breach: bool
    breach_reasons: Tuple[str, ...]
    tier7_running: Optional[bool]
    tier7_halted: Optional[bool]
    tier7_state_age_s: Optional[float]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "decision": self.decision,
            "reasons": list(self.reasons),
            "breach": self.breach,
            "breach_reasons": list(self.breach_reasons),
            "tier7_running": self.tier7_running,
            "tier7_halted": self.tier7_halted,
            "tier7_state_age_s": self.tier7_state_age_s,
        }


def _read_tier7_state(
    project_root: Path, now: datetime
) -> Tuple[Optional[bool], Optional[bool], Optional[float]]:
    """Return ``(running, halted, age_s)`` from tier7_state.json IF fresh.

    A stale or unreadable file returns ``(None, None, age_s)`` — fail-closed to
    "unknown", never to a stale "running" claim.
    """
    path = Path(project_root) / _TIER7_STATE_REL
    if not path.exists():
        return (None, None, None)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return (None, None, None)
    if not isinstance(payload, dict):
        return (None, None, None)

    generated_at = payload.get("generated_at")
    age_s: Optional[float] = None
    if isinstance(generated_at, str):
        try:
            ts = datetime.fromisoformat(generated_at)
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            age_s = (now.astimezone(timezone.utc) - ts.astimezone(timezone.utc)).total_seconds()
        except ValueError:
            age_s = None

    if age_s is None or age_s > _TIER7_STALE_S:
        return (None, None, age_s)
    return (payload.get("running"), payload.get("halted"), age_s)


def evaluate_rails(
    *,
    config: "ScannerConfig",
    project_root: Path,
    data_asof: "Optional[pd.Timestamp]" = None,
    now: Optional[datetime] = None,
    snapshot_universe_hash: Optional[str] = None,
) -> RailReport:
    """Compose the canonical decision gate with tier7 informational context.

    ``breach=True`` only for a NEW drawdown emergency (``EquityDecision.HALT``)
    — an already-``REFUSE``d (halted) state needs no additional de-risk action.
    """
    now = now or datetime.now(timezone.utc)
    decision = decide_cycle(
        config=config,
        project_root=Path(project_root),
        data_asof=data_asof,
        now=now,
        snapshot_universe_hash=snapshot_universe_hash,
    )
    breach_reasons = decision.reasons if decision.decision is EquityDecision.HALT else ()
    tier7_running, tier7_halted, tier7_age = _read_tier7_state(project_root, now)
    return RailReport(
        decision=decision.decision.value,
        reasons=decision.reasons,
        breach=bool(breach_reasons),
        breach_reasons=tuple(breach_reasons),
        tier7_running=tier7_running,
        tier7_halted=tier7_halted,
        tier7_state_age_s=tier7_age,
    )

"""
ScanDistributionStats — dry_run validation telemetry.

Emits one JSONL row per pair per scan cycle to
``trained_data/dry_run_validation.jsonl``. Used by
``scripts/analyze_dry_run.py`` to verify gate firing distributions and
execution-path proportions during the 48h first-contact validation
window (US-604).

Design notes:
- Write-only from the scan loop; no OANDA execution side effects.
- Atomic per-row append (open in 'a' mode, one json.dumps line).
- Schema is stable: consumers (analyze script + tests) pin exact keys.
- 0-candidate cycles (empty analyses / all errors) are silently skipped
  so the file does not bloat with null rows.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

logger = logging.getLogger(__name__)

DEFAULT_LOG_PATH = Path("trained_data/dry_run_validation.jsonl")

# Stable schema keys (exposed for tests + analyzer to validate against).
ROW_SCHEMA_KEYS = frozenset({
    "ts",
    "pair",
    "regime",
    "confidence",
    "wvs",
    "disagreement",
    "tradeable",
    "block_reasons",
    "would_submit",
    "gates_fired",
})

GATES_FIRED_KEYS = frozenset({
    "trend_veto",
    "mr_veto",
    "staleness_veto",
    "confidence_gate",
    "momentum_gate",
    "risk_gate",
    "correlation_blocked",
})


@dataclass
class ScanDistributionStats:
    """Append-only JSONL writer for dry_run validation telemetry.

    Parameters
    ----------
    log_path:
        Destination file. Parent dir is created on first write.
    enabled:
        Master toggle — when False all :meth:`record_cycle` calls are no-ops.
    """

    log_path: Path = DEFAULT_LOG_PATH
    enabled: bool = True

    def __post_init__(self) -> None:
        self.log_path = Path(self.log_path)

    def record_cycle(
        self,
        analyses: Iterable[Any],
        correlation_blocked_pairs: Optional[Iterable[str]] = None,
        *,
        ts: Optional[datetime] = None,
    ) -> int:
        """Record one row per candidate in ``analyses``.

        Returns the number of rows written. Cycles with zero directional
        candidates (HOLD-only or all errors) write no rows.
        """

        if not self.enabled:
            return 0

        analyses_list: List[Any] = [a for a in (analyses or []) if a is not None]
        if not analyses_list:
            return 0

        corr_blocked = set(correlation_blocked_pairs or [])
        ts_str = (ts or datetime.now(timezone.utc)).isoformat()
        rows: List[Dict[str, Any]] = []
        for a in analyses_list:
            row = self._build_row(a, ts_str, corr_blocked)
            if row is not None:
                rows.append(row)

        if not rows:
            return 0

        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(self.log_path, "a", encoding="utf-8") as fh:
                for row in rows:
                    fh.write(json.dumps(row, sort_keys=True) + "\n")
        except OSError as exc:  # disk full, permission, etc. — log & continue
            logger.warning("validation_stats write failed: %s", exc)
            return 0
        return len(rows)

    def _build_row(
        self,
        analysis: Any,
        ts_str: str,
        corr_blocked: set,
    ) -> Optional[Dict[str, Any]]:
        pair = getattr(analysis, "pair", None)
        if not pair:
            return None
        # Skip pure HOLD / errored pairs — they're not candidates.
        direction = getattr(analysis, "direction", "HOLD")
        error = getattr(analysis, "error", None)
        if error is not None or direction not in {"LONG", "SHORT"}:
            return None

        confidence = float(getattr(analysis, "confidence", 0.0) or 0.0)
        wvs = float(getattr(analysis, "weighted_vote_score", 0.0) or 0.0)
        disagreement = float(getattr(analysis, "model_disagreement", 0.0) or 0.0)
        regime = str(getattr(analysis, "volatility_regime", "UNKNOWN") or "UNKNOWN")
        tradeable = bool(getattr(analysis, "is_tradeable", False))

        gates_fired = _derive_gates_fired(analysis, pair in corr_blocked)
        block_reasons = _derive_block_reasons(analysis, gates_fired, tradeable)

        row: Dict[str, Any] = {
            "ts": ts_str,
            "pair": pair,
            "regime": regime,
            "confidence": round(confidence, 6),
            "wvs": round(wvs, 6),
            "disagreement": round(disagreement, 6),
            "tradeable": tradeable,
            "block_reasons": block_reasons,
            "would_submit": tradeable,  # dry_run: if tradeable, LIVE would submit
            "gates_fired": gates_fired,
        }
        return row


def _derive_gates_fired(analysis: Any, correlation_blocked: bool) -> Dict[str, bool]:
    """Map PairAnalysis + context to the stable gates_fired schema.

    A gate is considered "fired" (True) when it BLOCKED the trade. This keeps
    histogram semantics consistent: higher count = more often the limiting
    constraint.
    """
    reason_codes = set(getattr(analysis, "agent_reason_codes", []) or [])
    circuit_breakers = set(getattr(analysis, "circuit_breakers_triggered", []) or [])

    # Trend veto: trend agent failed (hard veto) — rule from 2026-04-15.
    trend_veto = (
        "trend_veto" in reason_codes
        or "trend_failed" in reason_codes
        or "trend" in circuit_breakers
    )
    # Mean reversion veto (2026-04-17 rule).
    mr_veto = (
        "mean_reversion_veto" in reason_codes
        or "mr_veto" in reason_codes
        or "mean_reversion" in circuit_breakers
    )
    # Staleness veto (2026-04-15 rule — component age > 7d).
    staleness_veto = (
        "staleness_veto" in reason_codes
        or "stale_model" in reason_codes
        or "staleness" in circuit_breakers
    )

    confidence_passed = bool(
        getattr(analysis, "confidence_passed", False)
        or getattr(analysis, "confidence_gate_passed", False)
    )
    momentum_passed = bool(
        getattr(analysis, "momentum_passed", False)
        or getattr(analysis, "momentum_gate_passed", False)
    )
    risk_passed = bool(
        getattr(analysis, "risk_passed", False)
        or getattr(analysis, "risk_gate_passed", False)
    )

    return {
        "trend_veto": bool(trend_veto),
        "mr_veto": bool(mr_veto),
        "staleness_veto": bool(staleness_veto),
        "confidence_gate": not confidence_passed,
        "momentum_gate": not momentum_passed,
        "risk_gate": not risk_passed,
        "correlation_blocked": bool(correlation_blocked),
    }


def _derive_block_reasons(
    analysis: Any,
    gates_fired: Dict[str, bool],
    tradeable: bool,
) -> List[str]:
    """Flat list of reason strings for this candidate."""
    if tradeable:
        return []

    reasons: List[str] = []
    for key, fired in gates_fired.items():
        if fired:
            reasons.append(key)

    kill_reason = getattr(analysis, "kill_reason", None)
    if kill_reason and kill_reason not in reasons:
        reasons.append(str(kill_reason))

    if getattr(analysis, "blocked_by_circuit_breaker", False):
        reasons.append("circuit_breaker")

    if not bool(getattr(analysis, "execution_quality_passed", True)):
        reasons.append("execution_quality")

    # agent_passed=False with no soft penalty applied
    if not bool(getattr(analysis, "agent_passed", False)) and not bool(
        getattr(analysis, "agent_soft_penalty_applied", False)
    ):
        reasons.append("agent_consensus")

    # Always ensure non-empty when not tradeable so downstream histograms
    # don't show mysterious "no reason" rows.
    if not reasons:
        reasons.append("other")
    return reasons


def _default_instance() -> ScanDistributionStats:
    """Construct a ScanDistributionStats with env-override for log path."""
    env_path = os.environ.get("BUDDY_DRY_RUN_VALIDATION_PATH")
    log_path = Path(env_path) if env_path else DEFAULT_LOG_PATH
    return ScanDistributionStats(log_path=log_path, enabled=True)


__all__ = [
    "ScanDistributionStats",
    "ROW_SCHEMA_KEYS",
    "GATES_FIRED_KEYS",
    "DEFAULT_LOG_PATH",
]

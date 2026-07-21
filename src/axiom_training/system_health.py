"""Consolidated AXIOM runtime, operator-attention, and promotion status.

The scopes are intentionally separate: a shadow-model promotion failure must
remain visible without claiming that the deterministic practice trading lane is
offline.  This module is read-only and never changes broker or operator state.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.axiom_training.promotion_criteria import PromotionVerdict, evaluate_promotion
from src.axiom_training.runtime_health import RuntimeHealthReport, evaluate_runtime_health

REPO_ROOT = Path(__file__).resolve().parents[2]
MAINTENANCE_PATH = REPO_ROOT / ".claude" / "maintenance_report.json"
ONLINE_LEARNING_PATH = REPO_ROOT / "trained_data" / "axiom" / "online_risk_learning.json"
SAFETY_MONITOR_LOG_PATH = REPO_ROOT / "trained_data" / "axiom" / "safety_monitor.jsonl"
# Online learning fires per closed trade; 72h tolerates a weekend market close.
LEARNING_LOOP_STALE_HOURS = 72.0
# The supervised monitor polls every 60s; 15 minutes = generous grace.
SAFETY_MONITOR_STALE_MINUTES = 15.0


@dataclass(frozen=True)
class ConsolidatedBlocker:
    scope: str
    code: str
    severity: str
    detail: str
    blocks_live_runtime: bool = False


@dataclass(frozen=True)
class AxiomSystemHealth:
    status: str
    live: bool
    environment: str
    policy_mode: str
    runtime: RuntimeHealthReport
    promotion_ready: bool
    blockers: Dict[str, List[ConsolidatedBlocker]] = field(default_factory=dict)
    summary: Dict[str, Any] = field(default_factory=dict)
    schema_version: str = "2026-07-10-v1"


def _read_json(path: Path) -> Dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _parse_time(value: Any) -> Optional[datetime]:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _last_jsonl_timestamp(path: Path, key: str = "ts") -> Optional[datetime]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return None
    for line in reversed(lines):
        try:
            payload = json.loads(line)
        except (TypeError, ValueError):
            continue
        if isinstance(payload, dict):
            return _parse_time(payload.get(key))
    return None


def _liveness_blockers(
    *,
    now: Optional[datetime] = None,
    online_learning_path: Path = ONLINE_LEARNING_PATH,
    safety_monitor_log_path: Path = SAFETY_MONITOR_LOG_PATH,
) -> List["ConsolidatedBlocker"]:
    """Watchdog-of-the-watchdogs: surface a dead learning loop or dead safety
    monitor as dashboard blockers. A stale champion keeps sizing forever with
    live inputs — the only way to notice the LOOP died is here."""
    current = now or datetime.now(timezone.utc)
    blockers: List[ConsolidatedBlocker] = []

    learning = _read_json(online_learning_path)
    learned_at = _parse_time(learning.get("completed_at"))
    if learned_at is None:
        blockers.append(ConsolidatedBlocker(
            scope="attention", code="learning_loop_never_ran", severity="warning",
            detail="online risk learning has no recorded cycle (online_risk_learning.json missing/invalid)",
        ))
    else:
        age_h = (current - learned_at).total_seconds() / 3600.0
        if age_h > LEARNING_LOOP_STALE_HOURS:
            blockers.append(ConsolidatedBlocker(
                scope="attention", code="learning_loop_stale", severity="warning",
                detail=(
                    f"last online learning cycle {age_h:.0f}h ago (> {LEARNING_LOOP_STALE_HOURS:.0f}h): "
                    "champion keeps sizing with live inputs, but nothing is retraining"
                ),
            ))

    monitor_at = _last_jsonl_timestamp(safety_monitor_log_path)
    monitor_age_min = (
        (current - monitor_at).total_seconds() / 60.0 if monitor_at is not None else None
    )
    if monitor_age_min is None or monitor_age_min > SAFETY_MONITOR_STALE_MINUTES:
        detail = (
            "safety monitor has never logged"
            if monitor_age_min is None
            else f"safety monitor last logged {monitor_age_min:.0f}m ago (> {SAFETY_MONITOR_STALE_MINUTES:.0f}m)"
        )
        blockers.append(ConsolidatedBlocker(
            scope="attention", code="safety_monitor_stale", severity="warning",
            detail=f"{detail}: restart com.axiom.safety_monitor (scripts/axiom_launchd/load.sh)",
        ))
    return blockers


def evaluate_axiom_system_health(
    *,
    runtime: Optional[RuntimeHealthReport] = None,
    promotion: Optional[PromotionVerdict] = None,
    maintenance_path: Path = MAINTENANCE_PATH,
    online_learning_path: Path = ONLINE_LEARNING_PATH,
    safety_monitor_log_path: Path = SAFETY_MONITOR_LOG_PATH,
) -> AxiomSystemHealth:
    runtime = runtime or evaluate_runtime_health()
    promotion = promotion or evaluate_promotion(runtime_health=runtime)
    maintenance = _read_json(maintenance_path)

    runtime_blocks = [
        ConsolidatedBlocker(
            scope="runtime",
            code=issue.check,
            severity=issue.severity,
            detail=issue.detail,
            blocks_live_runtime=True,
        )
        for issue in runtime.issues
    ]
    promotion_blocks = [
        ConsolidatedBlocker(
            scope="promotion",
            code="shadow_policy_promotion",
            severity="warning",
            detail=reason,
            blocks_live_runtime=False,
        )
        for reason in promotion.reasons
        if not reason.startswith("runtime health is ")
    ]
    needs_attention = maintenance.get("needs_attention")
    attention_blocks = [
        ConsolidatedBlocker(
            scope="attention",
            code="operator_attention",
            severity="info",
            detail=str(detail),
            blocks_live_runtime=False,
        )
        for detail in (needs_attention if isinstance(needs_attention, list) else [])
    ]
    attention_blocks.extend(_liveness_blockers(
        online_learning_path=online_learning_path,
        safety_monitor_log_path=safety_monitor_log_path,
    ))

    live = runtime.status == "HEALTHY" and bool(runtime.metrics.get("tier7_running"))
    return AxiomSystemHealth(
        status=runtime.status,
        live=live,
        environment="practice",
        policy_mode="shadow",
        runtime=runtime,
        promotion_ready=promotion.ready,
        blockers={
            "runtime": runtime_blocks,
            "promotion": promotion_blocks,
            "attention": attention_blocks,
        },
        summary={
            "runtime_blockers": len(runtime_blocks),
            "promotion_blockers": len(promotion_blocks),
            "attention_items": len(attention_blocks),
            "open_positions": runtime.metrics.get("account_open_trade_count", 0),
            "protected_positions": (
                int(runtime.metrics.get("account_open_trade_count", 0) or 0)
                - int(runtime.metrics.get("open_trade_missing_brackets", 0) or 0)
            ),
            "feedback_coverage": runtime.metrics.get("feedback_coverage"),
        },
    )


def axiom_system_health_to_dict(report: AxiomSystemHealth) -> Dict[str, Any]:
    return asdict(report)


__all__ = [
    "AxiomSystemHealth",
    "ConsolidatedBlocker",
    "axiom_system_health_to_dict",
    "evaluate_axiom_system_health",
]
